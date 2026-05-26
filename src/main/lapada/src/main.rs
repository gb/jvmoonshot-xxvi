//! L4 FD-passing forwarder: TCP :9999 → 2 API instances via SCM_RIGHTS.
//! No HTTP parse, no logs, no checks, no metrics, no TLS. Linux-only.
//!
//! For each accepted client TCP socket, lapada round-robins to one of the two
//! APIs and hands the raw fd over a persistent Unix control socket. The API
//! reads/writes directly to the client socket; lapada is off the data path.

#[cfg(target_os = "linux")]
mod linux {

use std::env;
use std::fs;
use std::io;
use std::net::{AddrParseError, SocketAddr};
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokio::net::{TcpListener, TcpSocket};
use tokio::time::sleep;

use std::os::fd::AsRawFd;

const LISTEN_BACKLOG: u32 = 4096;
const BACKEND_COOLDOWN_MS: u64 = 1_000;
const FD_PRECONNECT_INTERVAL: Duration = Duration::from_millis(25);

static DOWN_UNTIL_MS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static START: OnceLock<Instant> = OnceLock::new();

/// FD upstream paths, parsed once from FD_UPSTREAMS env var.
static FD_UPSTREAMS: OnceLock<[String; 2]> = OnceLock::new();

/// Raw fds for the persistent Unix control sockets to each API's FD receiver.
/// -1 means disconnected. Single-threaded (current_thread tokio) so plain AtomicI32
/// with Relaxed ordering is sufficient — no contention, just preventing the compiler
/// from treating the values as constants across the reconnect path.
static FD_CTRL_FDS: [AtomicI32; 2] = [AtomicI32::new(-1), AtomicI32::new(-1)];

fn set_int_sockopt(
    fd: std::os::fd::RawFd,
    level: libc::c_int,
    opt: libc::c_int,
    value: libc::c_int,
) {
    unsafe {
        libc::setsockopt(
            fd,
            level,
            opt,
            std::ptr::from_ref(&value).cast::<libc::c_void>(),
            std::mem::size_of_val(&value) as libc::socklen_t,
        );
    }
}

fn now_ms() -> u64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

fn backend_available(idx: usize) -> bool {
    let down_until = DOWN_UNTIL_MS[idx].load(Ordering::Relaxed);
    if down_until == 0 {
        return true;
    }
    if now_ms() >= down_until {
        DOWN_UNTIL_MS[idx].store(0, Ordering::Relaxed);
        return true;
    }
    false
}

fn mark_backend_failed(idx: usize) {
    DOWN_UNTIL_MS[idx].store(now_ms() + BACKEND_COOLDOWN_MS, Ordering::Relaxed);
}

fn parse_listen_addr(addr: &str) -> io::Result<SocketAddr> {
    addr.parse::<SocketAddr>()
        .map_err(|err: AddrParseError| io::Error::new(io::ErrorKind::InvalidInput, err))
}

fn bind_listener(listen_addr: &str) -> io::Result<TcpListener> {
    let addr = parse_listen_addr(listen_addr)?;
    let socket = match addr {
        SocketAddr::V4(_) => TcpSocket::new_v4()?,
        SocketAddr::V6(_) => TcpSocket::new_v6()?,
    };
    socket.set_reuseaddr(true)?;
    let fd = socket.as_raw_fd();
    set_int_sockopt(fd, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT, 1);
    set_int_sockopt(
        fd,
        libc::IPPROTO_TCP,
        libc::TCP_FASTOPEN,
        LISTEN_BACKLOG as libc::c_int,
    );
    socket.bind(addr)?;
    socket.listen(LISTEN_BACKLOG)
}

fn ready_file_path() -> String {
    env::var("LAPADA_READY_FILE").unwrap_or_else(|_| "/lapada.ready".into())
}

fn clear_ready_file() {
    let _ = fs::remove_file(ready_file_path());
}

fn write_ready_file() -> io::Result<()> {
    fs::write(ready_file_path(), b"ready")
}

/// Discriminate sendmsg failures so a transient kernel hiccup doesn't tear the
/// persistent control socket. Returns true if the error is fatal to the conn.
fn is_ctrl_conn_fatal(err: &io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(libc::EPIPE)
            | Some(libc::ECONNRESET)
            | Some(libc::ENOTCONN)
            | Some(libc::EBADF)
            | Some(libc::ENOTSOCK)
            | Some(libc::ESHUTDOWN)
    )
    // EAGAIN/EWOULDBLOCK, ENOBUFS, ENOMEM, EINTR (already handled inside send_fd):
    // treat as transient — caller may retry on the next request without tearing the
    // persistent control conn.
}

/// Pass the client FD to the API at `backend_idx` via the persistent control socket.
/// Fully synchronous: sendmsg is a raw syscall, no async overhead needed.
/// Reconnects only on fatal errors; transient errors propagate without closing.
fn pass_fd_to_api(client_fd: std::os::fd::RawFd, backend_idx: usize) -> io::Result<()> {
    use std::os::fd::IntoRawFd;
    let mut ctrl_fd = FD_CTRL_FDS[backend_idx].load(Ordering::Relaxed);
    if ctrl_fd < 0 {
        let path = &FD_UPSTREAMS.get().unwrap()[backend_idx];
        ctrl_fd = std::os::unix::net::UnixStream::connect(path)?.into_raw_fd();
        FD_CTRL_FDS[backend_idx].store(ctrl_fd, Ordering::Relaxed);
    }
    match send_fd(ctrl_fd, client_fd) {
        Ok(()) => Ok(()),
        Err(e) => {
            if is_ctrl_conn_fatal(&e) {
                unsafe { libc::close(ctrl_fd); }
                FD_CTRL_FDS[backend_idx].store(-1, Ordering::Relaxed);
            }
            Err(e)
        }
    }
}

fn accept_client_fd(listener_fd: std::os::fd::RawFd) -> io::Result<std::os::fd::RawFd> {
    loop {
        let fd = unsafe {
            libc::accept4(
                listener_fd,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            )
        };
        if fd >= 0 {
            return Ok(fd);
        }
        let err = io::Error::last_os_error();
        if err.kind() != io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
}

async fn preconnect_fd_upstreams() -> io::Result<()> {
    use std::os::unix::io::IntoRawFd;

    let fd_upstreams = FD_UPSTREAMS.get().unwrap();
    let mut last_error = String::new();

    loop {
        let mut connected = 0;
        for i in 0..2 {
            if FD_CTRL_FDS[i].load(Ordering::Relaxed) >= 0 {
                connected += 1;
                continue;
            }
            match std::os::unix::net::UnixStream::connect(&fd_upstreams[i]) {
                Ok(s) => {
                    FD_CTRL_FDS[i].store(s.into_raw_fd(), Ordering::Relaxed);
                    connected += 1;
                    eprintln!("[lapada] fd ctrl {i} pre-connected");
                }
                Err(e) => {
                    last_error = format!("fd ctrl {i} {}: {e}", fd_upstreams[i]);
                }
            }
        }
        if connected == 2 {
            return Ok(());
        }
        let _ = &last_error;
        sleep(FD_PRECONNECT_INTERVAL).await;
    }
}

fn send_fd(ctrl_fd: std::os::fd::RawFd, client_fd: std::os::fd::RawFd) -> io::Result<()> {
    unsafe {
        let mut data: u8 = 0;
        let mut iov = libc::iovec {
            iov_base: &mut data as *mut u8 as *mut libc::c_void,
            iov_len: 1,
        };

        let fd_size = std::mem::size_of::<libc::c_int>() as u32;
        let cmsg_space = libc::CMSG_SPACE(fd_size) as usize;
        let mut cmsg_buf = [0u8; 64];
        if cmsg_space > cmsg_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SCM_RIGHTS buffer too small",
            ));
        }

        let cmsg = cmsg_buf.as_mut_ptr() as *mut libc::cmsghdr;
        (*cmsg).cmsg_len = libc::CMSG_LEN(fd_size) as _;
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        std::ptr::write(libc::CMSG_DATA(cmsg) as *mut libc::c_int, client_fd);

        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov as *mut libc::iovec;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_space as _;

        loop {
            let sent = libc::sendmsg(ctrl_fd, &msg as *const libc::msghdr, libc::MSG_NOSIGNAL);
            if sent == 1 {
                return Ok(());
            }
            if sent < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(err);
            }
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short SCM_RIGHTS send",
            ));
        }
    }
}

#[tokio::main(flavor = "current_thread")]
pub async fn run() -> std::io::Result<()> {
    unsafe {
        libc::prctl(libc::PR_SET_TIMERSLACK, 0_u64);
    }

    clear_ready_file();

    // FD_UPSTREAMS is mandatory: "path1,path2". The proxy fallback was removed
    // because production has always run FD-mode and per-conn heap + tokio::spawn
    // was an unbounded foot-gun if it ever fired under contest pressure.
    let val = env::var("FD_UPSTREAMS").map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "FD_UPSTREAMS must be set to 'path1,path2'",
        )
    })?;
    let parts: Vec<&str> = val.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("FD_UPSTREAMS must be 'path1,path2', got: {val}"),
        ));
    }
    let _ = FD_UPSTREAMS.set([parts[0].trim().to_string(), parts[1].trim().to_string()]);

    let listen_addr = env::var("LAPADA_LISTEN").unwrap_or_else(|_| "0.0.0.0:9999".into());
    let listener = bind_listener(&listen_addr)?;

    // Bind early so Docker's published host port does not reset early test
    // connections, but do not accept anything until both API FD receivers are
    // connected. Any premature client requests wait in the kernel backlog.
    preconnect_fd_upstreams().await?;
    write_ready_file()?;

    let upstreams = FD_UPSTREAMS.get().unwrap();
    eprintln!(
        "[lapada] listening on {} fd_upstreams=[{},{}]",
        listen_addr, upstreams[0], upstreams[1]
    );

    // Blocking accept4 removes epoll_wait and the tokio task scheduler from the
    // per-connection hot path. The accepted child fd stays nonblocking so Java
    // receives the same socket mode used by older non-FD tags.
    let std_listener = listener.into_std()?;
    std_listener.set_nonblocking(false)?;
    let listener_fd = std_listener.as_raw_fd();
    let mut rr_idx: usize = 0;
    loop {
        let client_fd = match accept_client_fd(listener_fd) {
            Ok(fd) => fd,
            Err(e) => {
                eprintln!("[lapada] accept failed: {e}");
                continue;
            }
        };
        // TCP_NODELAY here saves one setsockopt syscall on the CPU-constrained
        // Java IO thread. The flag is preserved across SCM_RIGHTS.
        // TCP_QUICKACK is intentionally NOT set: it is one-shot and clears long
        // before Java writes the response, so the prior per-accept call was waste.
        set_int_sockopt(client_fd, libc::IPPROTO_TCP, libc::TCP_NODELAY, 1);
        let idx = rr_idx;
        rr_idx ^= 1;
        let second_idx = (idx + 1) % 2;
        let mut passed = false;
        if backend_available(idx) {
            match pass_fd_to_api(client_fd, idx) {
                Ok(()) => passed = true,
                Err(_) => mark_backend_failed(idx),
            }
        }
        if !passed && backend_available(second_idx) && pass_fd_to_api(client_fd, second_idx).is_err() {
            mark_backend_failed(second_idx);
        }
        // Java holds its own dup via SCM_RIGHTS.
        unsafe { libc::close(client_fd); }
    }
}

pub fn healthcheck() -> std::io::Result<()> {
    Path::new(&ready_file_path()).try_exists().and_then(|exists| {
        if exists {
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, "lapada not ready"))
        }
    })
}

}

#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--healthcheck") {
        return linux::healthcheck();
    }
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "lapada is Linux-only (SCM_RIGHTS FD passing)",
    ))
}
