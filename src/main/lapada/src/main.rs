//! L4 round-robin proxy: TCP :9999 → 2 UDS backends.
//! No HTTP parse, no logs, no checks, no metrics, no TLS.
//! Exception: GET /ready is intercepted and answered only when BOTH backends are ready.
//!
//! FD-passing mode (set FD_UPSTREAMS env var):
//!   Instead of proxying data, lapada accepts the TCP socket then passes the raw FD to the
//!   chosen API instance via sendmsg SCM_RIGHTS over a persistent Unix control socket.  The
//!   API reads directly from the client socket, eliminating the lapada data-copy path.
//!   BACKENDS (api1.sock/api2.sock) are still used for /ready health probes.

use std::env;
use std::io;
use std::net::{AddrParseError, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpSocket, TcpStream, UnixStream};
use tokio::time::timeout;

#[cfg(target_os = "linux")]
use tokio::sync::Mutex as AsyncMutex;

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

const BACKENDS: [&str; 2] = ["/sockets/api1.sock", "/sockets/api2.sock"];
const LISTEN_BACKLOG: u32 = 4096;
const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);
const COPY_BUF: usize = 1024;
const BACKEND_COOLDOWN_MS: u64 = 1_000;
const READY_PROBE_TIMEOUT: Duration = Duration::from_millis(500);
const READY_REQUEST: &[u8] = b"GET /ready HTTP/1.1\r\nHost: lapada\r\nConnection: close\r\n\r\n";
const READY_PREFIX: &[u8] = b"GET /ready "; // 11 bytes

static ALL_READY: AtomicBool = AtomicBool::new(false);
static RR: AtomicU8 = AtomicU8::new(0);
static DOWN_UNTIL_MS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static START: OnceLock<Instant> = OnceLock::new();

/// FD upstream paths, parsed once from FD_UPSTREAMS env var.
/// None = proxy mode (default).  Some = FD-passing mode.
static FD_UPSTREAMS: OnceLock<[String; 2]> = OnceLock::new();

/// Persistent Unix control sockets to each API's FD receiver.
/// connect-per-client is replaced by one long-lived connection per backend.
/// On sendmsg failure the slot is set to None; the next pass_fd_to_api call reconnects.
#[cfg(target_os = "linux")]
static FD_CTRL_SOCKS: OnceLock<[AsyncMutex<Option<UnixStream>>; 2]> = OnceLock::new();

#[cfg(target_os = "linux")]
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

async fn connect_backend(idx: usize) -> io::Result<UnixStream> {
    if !backend_available(idx) {
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "backend cooling down",
        ));
    }
    match timeout(CONNECT_TIMEOUT, UnixStream::connect(BACKENDS[idx])).await {
        Ok(Ok(stream)) => {
            DOWN_UNTIL_MS[idx].store(0, Ordering::Relaxed);
            Ok(stream)
        }
        Ok(Err(err)) => {
            mark_backend_failed(idx);
            Err(err)
        }
        Err(_) => {
            mark_backend_failed(idx);
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "backend connect timeout",
            ))
        }
    }
}

async fn connect_upstream(first: usize) -> io::Result<UnixStream> {
    match connect_backend(first).await {
        Ok(s) => Ok(s),
        Err(_) => connect_backend((first + 1) % BACKENDS.len()).await,
    }
}

async fn pump<R, W>(reader: &mut R, writer: &mut W, buf: &mut [u8]) -> io::Result<()>
where
    R: AsyncRead + Unpin + ?Sized,
    W: AsyncWrite + Unpin + ?Sized,
{
    loop {
        let n = match reader.read(buf).await {
            Ok(0) => {
                let _ = writer.shutdown().await;
                return Ok(());
            }
            Ok(n) => n,
            Err(err) => {
                let _ = writer.shutdown().await;
                return Err(err);
            }
        };
        if let Err(err) = writer.write_all(&buf[..n]).await {
            let _ = writer.shutdown().await;
            return Err(err);
        }
    }
}

async fn check_backend_ready(idx: usize) -> bool {
    let mut stream = match timeout(CONNECT_TIMEOUT, UnixStream::connect(BACKENDS[idx])).await {
        Ok(Ok(s)) => s,
        _ => return false,
    };
    if stream.write_all(READY_REQUEST).await.is_err() {
        return false;
    }
    let mut head = [0u8; 12];
    match timeout(READY_PROBE_TIMEOUT, stream.read_exact(&mut head)).await {
        Ok(Ok(_)) => head.starts_with(b"HTTP/1.1 200"),
        _ => false,
    }
}

async fn handle_ready_probe(mut client: TcpStream) {
    let (a, b) = tokio::join!(check_backend_ready(0), check_backend_ready(1));
    let resp: &[u8] = if a && b {
        ALL_READY.store(true, Ordering::Release);
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
    } else {
        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    };
    let _ = client.write_all(resp).await;
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
    #[cfg(target_os = "linux")]
    {
        let fd = socket.as_raw_fd();
        set_int_sockopt(fd, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT, 1);
        set_int_sockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_FASTOPEN,
            LISTEN_BACKLOG as libc::c_int,
        );
    }
    socket.bind(addr)?;
    socket.listen(LISTEN_BACKLOG)
}

/// Pass the client FD to the API at `backend_idx` via the persistent control socket.
/// Reconnects automatically if the control connection has dropped.
/// Linux-only: FD passing requires SCM_RIGHTS which is not portable.
#[cfg(target_os = "linux")]
async fn pass_fd_to_api(client: &TcpStream, backend_idx: usize) -> io::Result<()> {
    let fd_upstreams = FD_UPSTREAMS.get().unwrap();
    let ctrl_socks = FD_CTRL_SOCKS.get().unwrap();
    let client_fd = client.as_raw_fd();

    let mut guard = ctrl_socks[backend_idx].lock().await;
    if guard.is_none() {
        match timeout(
            CONNECT_TIMEOUT,
            UnixStream::connect(&fd_upstreams[backend_idx]),
        )
        .await
        {
            Ok(Ok(s)) => *guard = Some(s),
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "fd ctrl connect timeout",
                ))
            }
        }
    }

    match send_fd_on_ctrl(guard.as_ref().unwrap(), client_fd).await {
        Ok(()) => Ok(()),
        Err(e) => {
            // Persistent connection broken; Java side will re-accept after seeing EOF.
            *guard = None;
            Err(e)
        }
    }
}

#[cfg(target_os = "linux")]
async fn send_fd_on_ctrl(ctrl: &UnixStream, client_fd: std::os::fd::RawFd) -> io::Result<()> {
    loop {
        match send_fd(ctrl.as_raw_fd(), client_fd) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                ctrl.writable().await?;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(not(target_os = "linux"))]
async fn pass_fd_to_api(_client: &TcpStream, _backend_idx: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "FD passing is Linux-only",
    ))
}

#[cfg(target_os = "linux")]
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
async fn main() -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_TIMERSLACK, 0_u64);
    }

    // Parse FD_UPSTREAMS once: "path1,path2"
    let fd_mode = if let Ok(val) = env::var("FD_UPSTREAMS") {
        let parts: Vec<&str> = val.splitn(2, ',').collect();
        if parts.len() == 2 {
            let arr = [parts[0].trim().to_string(), parts[1].trim().to_string()];
            let _ = FD_UPSTREAMS.set(arr);
            true
        } else {
            eprintln!("[lapada] FD_UPSTREAMS must be 'path1,path2', got: {val}");
            false
        }
    } else {
        false
    };

    // Initialize persistent control socket slots (connect lazily on first client).
    #[cfg(target_os = "linux")]
    if fd_mode {
        let _ = FD_CTRL_SOCKS.set([AsyncMutex::new(None), AsyncMutex::new(None)]);
    }

    let listen_addr = env::var("LAPADA_LISTEN").unwrap_or_else(|_| "0.0.0.0:9999".into());
    let listener = bind_listener(&listen_addr)?;

    eprintln!(
        "[lapada] listening on {} backends={:?} fd_mode={}",
        listen_addr, BACKENDS, fd_mode
    );

    loop {
        let (mut client, _peer) = match listener.accept().await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lapada] accept failed: {e}");
                continue;
            }
        };
        let _ = client.set_nodelay(true);
        #[cfg(target_os = "linux")]
        set_int_sockopt(client.as_raw_fd(), libc::IPPROTO_TCP, libc::TCP_QUICKACK, 1);

        // Intercept /ready probes (both proxy and FD mode).
        if !ALL_READY.load(Ordering::Acquire) {
            let mut peek_buf = [0u8; 11];
            if let Ok(n) = client.peek(&mut peek_buf).await {
                if n >= 11 && &peek_buf[..11] == READY_PREFIX {
                    tokio::spawn(handle_ready_probe(client));
                    continue;
                }
            }
        }

        let idx = RR.fetch_xor(1, Ordering::Relaxed) as usize;
        let second_idx = (idx + 1) % 2;

        if fd_mode {
            tokio::spawn(async move {
                let mut passed = false;
                if backend_available(idx) {
                    match pass_fd_to_api(&client, idx).await {
                        Ok(()) => passed = true,
                        Err(_) => mark_backend_failed(idx),
                    }
                }
                if !passed && backend_available(second_idx) {
                    if pass_fd_to_api(&client, second_idx).await.is_err() {
                        mark_backend_failed(second_idx);
                    }
                }
            });
        } else {
            // Proxy path: copy bidirectionally through lapada.
            tokio::spawn(async move {
                let mut upstream = match connect_upstream(idx).await {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = client.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
                        return;
                    }
                };
                let (mut cr, mut cw) = client.split();
                let (mut ur, mut uw) = upstream.split();
                let mut c2u = [0u8; COPY_BUF];
                let mut u2c = [0u8; COPY_BUF];
                let _ = tokio::join!(
                    pump(&mut cr, &mut uw, &mut c2u),
                    pump(&mut ur, &mut cw, &mut u2c)
                );
            });
        }
    }
}
