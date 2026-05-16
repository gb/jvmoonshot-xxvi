package com.github.gb.moonshot.http;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Linux-only FD-passing primitives via Panama Foreign Function & Memory API.
 *
 * <p>{@link #receive(int)} wraps {@code recvmsg()} with {@code SCM_RIGHTS} to receive a file
 * descriptor sent by lapada over a control Unix socket.  Each call uses a thread-confined
 * {@link Arena} that is allocated once per thread and reused across calls — no per-call
 * allocation on the Java heap.
 *
 * <p>{@link #setTcpNoDelay(int)} wraps {@code setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)} so the
 * received TCP socket behaves the same as one accepted by {@link NioHttpServer}.
 *
 * <p>All native memory layouts are for Linux x86-64 (glibc 64-bit).
 * {@code struct msghdr}: 56 bytes  (msg_name 8, msg_namelen 4, pad 4, msg_iov 8,
 * msg_iovlen 8, msg_control 8, msg_controllen 8, msg_flags 4, pad 4).
 * {@code struct iovec}: 16 bytes  (iov_base 8, iov_len 8).
 * {@code struct cmsghdr}: cmsg_len (size_t=8) + cmsg_level (int=4) + cmsg_type (int=4) = 16 bytes.
 * FD data at CMSG_DATA offset 16 (4 bytes).  CMSG_SPACE(4) = CMSG_ALIGN(20) = 24 bytes.
 */
public final class FdPassing {

    // ---- setsockopt constants ----
    private static final int IPPROTO_TCP = 6;
    private static final int TCP_NODELAY = 1;

    // ---- SOL_SOCKET / SCM_RIGHTS ----
    private static final int SOL_SOCKET = 1;
    private static final int SCM_RIGHTS = 1;

    // ---- msghdr field offsets (Linux x86-64) ----
    private static final long MSGHDR_SIZE = 56;
    private static final long OFF_MSG_IOV = 16;
    private static final long OFF_MSG_IOVLEN = 24;
    private static final long OFF_MSG_CONTROL = 32;
    private static final long OFF_MSG_CONTROLLEN = 40;

    // ---- iovec field offsets ----
    private static final long IOVEC_SIZE = 16;
    private static final long OFF_IOV_BASE = 0;
    private static final long OFF_IOV_LEN = 8;

    // ---- cmsghdr + data layout ----
    private static final long CMSG_SPACE = 24; // CMSG_SPACE(sizeof(int)) on glibc 64-bit
    private static final long OFF_CMSG_LEN = 0;  // size_t (8 bytes)
    private static final long OFF_CMSG_LEVEL = 8;  // int (4 bytes)
    private static final long OFF_CMSG_TYPE = 12; // int (4 bytes)
    private static final long OFF_CMSG_DATA = 16; // first data byte (FD as int32)
    private static final long CMSG_LEN_VALUE = 20; // sizeof(cmsghdr) + sizeof(int) = 16 + 4

    // ---- native method handles (found once, cached) ----
    private static final MethodHandle RECVMSG;
    private static final MethodHandle SETSOCKOPT;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = linker.defaultLookup();

        RECVMSG = linker.downcallHandle(
                lookup.find("recvmsg").orElseThrow(() -> new RuntimeException("recvmsg not found")),
                FunctionDescriptor.of(
                        ValueLayout.JAVA_LONG,   // ssize_t
                        ValueLayout.JAVA_INT,    // int sockfd
                        ValueLayout.ADDRESS,     // struct msghdr*
                        ValueLayout.JAVA_INT     // int flags
                )
        );

        SETSOCKOPT = linker.downcallHandle(
                lookup.find("setsockopt").orElseThrow(() -> new RuntimeException("setsockopt not found")),
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,  // int return
                        ValueLayout.JAVA_INT,  // int sockfd
                        ValueLayout.JAVA_INT,  // int level
                        ValueLayout.JAVA_INT,  // int optname
                        ValueLayout.ADDRESS,   // const void* optval
                        ValueLayout.JAVA_INT   // socklen_t optlen
                )
        );
    }

    /**
     * Per-thread arena + pre-allocated native buffers, reused across calls.
     */
    private static final ThreadLocal<ThreadBuffers> BUFS = ThreadLocal.withInitial(ThreadBuffers::new);

    private static final class ThreadBuffers {
        final Arena arena;
        final MemorySegment msghdr;   // struct msghdr
        final MemorySegment iovec;    // struct iovec
        final MemorySegment dataByte; // 1-byte iov_base payload
        final MemorySegment cmsgBuf;  // cmsghdr + FD data
        final MemorySegment intVal;   // 4-byte int for setsockopt optval

        ThreadBuffers() {
            arena = Arena.ofConfined();
            msghdr = arena.allocate(MSGHDR_SIZE, 8);
            iovec = arena.allocate(IOVEC_SIZE, 8);
            dataByte = arena.allocate(1, 1);
            cmsgBuf = arena.allocate(CMSG_SPACE, 8);
            intVal = arena.allocate(4, 4);

            // Wire up iovec → dataByte (constant across calls).
            iovec.set(ValueLayout.ADDRESS, OFF_IOV_BASE, dataByte);
            iovec.set(ValueLayout.JAVA_LONG, OFF_IOV_LEN, 1L);

            // Wire up msghdr → iovec (constant).
            msghdr.set(ValueLayout.ADDRESS, OFF_MSG_IOV, iovec);
            msghdr.set(ValueLayout.JAVA_LONG, OFF_MSG_IOVLEN, 1L);
            msghdr.set(ValueLayout.ADDRESS, OFF_MSG_CONTROL, cmsgBuf);
            msghdr.set(ValueLayout.JAVA_LONG, OFF_MSG_CONTROLLEN, CMSG_SPACE);

            // cmsghdr header is fixed for receiving one FD.
            cmsgBuf.set(ValueLayout.JAVA_LONG, OFF_CMSG_LEN, CMSG_LEN_VALUE);
            cmsgBuf.set(ValueLayout.JAVA_INT, OFF_CMSG_LEVEL, SOL_SOCKET);
            cmsgBuf.set(ValueLayout.JAVA_INT, OFF_CMSG_TYPE, SCM_RIGHTS);
        }
    }

    /**
     * Receive one file descriptor from a lapada control socket via {@code recvmsg(SCM_RIGHTS)}.
     *
     * @param controlSocketFd the raw fd of the accepted lapada control connection
     * @return the received file descriptor (≥0), or -1 if the connection was closed / error
     */
    public static int receive(int controlSocketFd) {
        ThreadBuffers b = BUFS.get();
        b.dataByte.set(ValueLayout.JAVA_BYTE, 0, (byte) 0);
        b.cmsgBuf.set(ValueLayout.JAVA_INT, OFF_CMSG_DATA, 0); // clear FD slot
        // Reset msg_controllen to the allocated size before the call; the kernel
        // writes the actual ancillary-data length back here after recvmsg.
        b.msghdr.set(ValueLayout.JAVA_LONG, OFF_MSG_CONTROLLEN, CMSG_SPACE);

        long received;
        try {
            received = (long) RECVMSG.invokeExact(controlSocketFd, b.msghdr, 0);
        } catch (Throwable t) {
            return -1;
        }
        if (received <= 0) return -1;

        // Check that the kernel actually wrote ancillary data (msg_controllen > 0).
        // If recvmsg returns > 0 but with no SCM_RIGHTS payload (data-only message or
        // connection-close race), msg_controllen is 0 and the cmsghdr fields are stale
        // from the constructor — reading them would silently return fd=0 (stdin).
        long actualCtrlLen = b.msghdr.get(ValueLayout.JAVA_LONG, OFF_MSG_CONTROLLEN);
        if (actualCtrlLen < CMSG_LEN_VALUE) return -1;

        // Validate cmsghdr.
        long cmsgLen = b.cmsgBuf.get(ValueLayout.JAVA_LONG, OFF_CMSG_LEN);
        int cmsgLevel = b.cmsgBuf.get(ValueLayout.JAVA_INT, OFF_CMSG_LEVEL);
        int cmsgType = b.cmsgBuf.get(ValueLayout.JAVA_INT, OFF_CMSG_TYPE);
        if (cmsgLen < CMSG_LEN_VALUE || cmsgLevel != SOL_SOCKET || cmsgType != SCM_RIGHTS) {
            return -1;
        }
        return b.cmsgBuf.get(ValueLayout.JAVA_INT, OFF_CMSG_DATA);
    }

    /**
     * Set {@code TCP_NODELAY=1} on a raw socket FD so pipelined keep-alive responses
     * are flushed immediately without Nagle delay.
     */
    public static void setTcpNoDelay(int fd) {
        ThreadBuffers b = BUFS.get();
        b.intVal.set(ValueLayout.JAVA_INT, 0, 1);
        try {
            SETSOCKOPT.invokeExact(fd, IPPROTO_TCP, TCP_NODELAY, b.intVal, 4);
        } catch (Throwable ignored) {
        }
    }

    private FdPassing() {
    }
}
