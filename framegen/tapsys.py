"""What the frame tap needs from the operating system: named shared memory, the clock its timestamps use, and
the reader's wake-up when a frame lands. Windows, Linux and macOS, matching chiaki_tap/frametap.c.

  Windows  a named file mapping, QueryPerformanceCounter, and a named auto-reset event ("<name>-frame")
  Linux    /dev/shm/<the name after its last backslash>, CLOCK_MONOTONIC in nanoseconds (what
           time.perf_counter() reads there), and a futex on the low 32 bits of the ring's write_seq
  macOS    shm_open("/<the name after its last backslash>"), CLOCK_UPTIME_RAW in nanoseconds (what
           time.perf_counter() reads there), and no futex, so the reader polls write_seq while it waits
"""
from __future__ import annotations

import ctypes
import errno
import mmap
import os
import struct
import sys
import time

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform == "darwin"
SUPPORTED = WINDOWS or LINUX or MACOS
SHM_NAME_MAX = 31  # the longest shared-memory name macOS takes, with the slash (PSHMNAMLEN)
POLL_S = 0.0005  # how often a waiting reader looks at write_seq on macOS

if WINDOWS:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateEventW.restype = ctypes.c_void_p
    _k32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    _k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k32.SetEvent.argtypes = [ctypes.c_void_p]
    _k32.CloseHandle.argtypes = [ctypes.c_void_p]

    def qpc() -> int:
        v = ctypes.c_int64()
        _k32.QueryPerformanceCounter(ctypes.byref(v))
        return v.value

    def qpc_freq() -> int:
        f = ctypes.c_int64()
        _k32.QueryPerformanceFrequency(ctypes.byref(f))
        return f.value

elif LINUX:
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.syscall.restype = ctypes.c_long
    _SYS_FUTEX = {"x86_64": 202, "aarch64": 98}.get(os.uname().machine)
    _FUTEX_WAIT, _FUTEX_WAKE = 0, 1

    class _Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    def qpc() -> int:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    def qpc_freq() -> int:
        return 1_000_000_000

elif MACOS:
    def qpc() -> int:
        # Not CLOCK_MONOTONIC, which on macOS also counts the time the Mac was asleep
        return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)

    def qpc_freq() -> int:
        return 1_000_000_000

else:  # elsewhere: there's no tap, but the module still imports, so the ring's other code and tests can
    def qpc() -> int:
        return time.perf_counter_ns()

    def qpc_freq() -> int:
        return 1_000_000_000


def unsupported() -> RuntimeError:
    return RuntimeError("the chiaki frame tap works on Windows, Linux and macOS only")


def shm_name(name: str) -> str:
    """The shared memory's name on macOS, "/" and what follows the last backslash. "Local\\x", "/x" and "x" are all
    "/x". ValueError if nothing is left, it has another "/", or it's longer than macOS allows."""
    leaf = name.rsplit("\\", 1)[-1].lstrip("/")
    path = "/" + leaf
    if not leaf or "/" in leaf or len(path.encode()) > SHM_NAME_MAX:
        raise ValueError(f"can't use {name!r} as shared memory. On macOS its name is \"/\" and what follows the last "
                         f"backslash, with no other \"/\" and {SHM_NAME_MAX} characters at most ({path!r} has "
                         f"{len(path.encode())})")
    return path


def _open_macos(name: str, size: int) -> mmap.mmap:
    import _posixshmem  # what multiprocessing.shared_memory uses, without its tracker that unlinks it at exit
    path = shm_name(name)
    fd = _posixshmem.shm_open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        # macOS sizes an object only while it's empty, so only the side that comes first sizes it. If both try at
        # once, the second gets EINVAL and can still see 0 bytes for a moment, so it waits up to 200 ms for the size.
        if os.fstat(fd).st_size == 0:
            try:
                os.ftruncate(fd, size)
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
        have, t_end = os.fstat(fd).st_size, time.perf_counter() + 0.2
        while have == 0 and time.perf_counter() < t_end:
            time.sleep(0.0002)
            have = os.fstat(fd).st_size
        if have < size:  # macOS rounds the size up to whole pages
            raise OSError(f"the shared memory {path} is {have} bytes, too small for the frame tap ({size})")
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)  # the mapping keeps the memory


def unlink(*names: str) -> None:
    """Remove the named shared memory on Linux and macOS, where it otherwise lasts until a restart (for tests).
    Memory still mapped stays usable. Windows frees a mapping with its last handle, so there's nothing to do."""
    for name in names:
        try:
            if MACOS:
                import _posixshmem
                _posixshmem.shm_unlink(shm_name(name))
            elif LINUX:
                os.unlink("/dev/shm/" + name.rsplit("\\", 1)[-1])
        except (FileNotFoundError, ValueError):
            pass


def open_map(name: str, size: int) -> mmap.mmap:
    """The named shared memory, created if neither side has made it yet."""
    if WINDOWS:
        return mmap.mmap(-1, size, tagname=name)
    if MACOS:
        return _open_macos(name, size)
    if not LINUX:
        raise unsupported()
    path = "/dev/shm/" + name.rsplit("\\", 1)[-1]
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(fd).st_size < size:
            os.ftruncate(fd, size)
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)  # the mapping keeps the memory


class Signal:
    """A frame landed. One reader waits; the writer sets it after every frame and at the stream's end.

    Behaves like Windows' auto-reset event: a wait() returns at once if the writer set it since the previous
    wait(), and otherwise sleeps until it does or the timeout passes. On Linux it is a futex on write_seq's low
    word (offset 32 of `mm`), so it needs nothing beyond the shared memory itself. macOS has no futex, so wait()
    looks at write_seq and the magic every POLL_S instead, and set() does nothing. The magic is there because the
    writer also sets the event when a stream ends."""

    def __init__(self, mm: mmap.mmap, event_name: str, seq_offset: int = 32) -> None:
        if WINDOWS:
            self._event = _k32.CreateEventW(None, 0, 0, event_name)
            return
        if MACOS:
            self._mm, self._seq_offset = mm, seq_offset
            self._seen = self._now()
            return
        if not LINUX or _SYS_FUTEX is None:
            raise unsupported()
        self._word = ctypes.c_uint32.from_buffer(mm, seq_offset)
        self._addr = ctypes.addressof(self._word)
        self._seen = self._word.value

    def _now(self) -> tuple[int, int]:
        """The magic and write_seq's low word (macOS), which change when a frame lands or a stream ends."""
        return struct.unpack_from("<I", self._mm, 0)[0], struct.unpack_from("<I", self._mm, self._seq_offset)[0]

    def wait(self, timeout_s: float) -> None:
        if WINDOWS:
            _k32.WaitForSingleObject(self._event, int(timeout_s * 1000))
            return
        if MACOS:
            t_end = time.perf_counter() + timeout_s
            while (cur := self._now()) == self._seen and (left := t_end - time.perf_counter()) > 0:
                time.sleep(min(POLL_S, left))
            self._seen = cur
            return
        cur = self._word.value
        if cur == self._seen:  # nothing new since the last wait: sleep (returns at once if it changes first)
            ts = _Timespec(int(timeout_s), int((timeout_s % 1) * 1e9))
            _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._addr), _FUTEX_WAIT, ctypes.c_uint32(cur),
                          ctypes.byref(ts), None, 0)
        self._seen = self._word.value

    def set(self) -> None:
        if WINDOWS:
            _k32.SetEvent(self._event)
        elif LINUX:
            _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._addr), _FUTEX_WAKE, 1, None, None, 0)

    def close(self) -> None:
        if WINDOWS:
            _k32.CloseHandle(self._event)
        elif LINUX:
            del self._word  # release the buffer export, so the mmap can close
