//! POSIX shared memory, as frametap.c and framegen/chiaki_tap.py's `_shm_map` make it: either side may come first.

use std::ffi::CString;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::layout::{AUDIO_SUFFIX, SHM_NAME_MAX};
use crate::{Error, Result};

/// A tap's name as a POSIX shared-memory name: what follows the last backslash, after a "/" (`Local\chiaki-frametap`
/// is `/chiaki-frametap`; "/name" and "name" are "/name" too). Err if nothing is left, it has a "/" in it, or it's
/// longer than SHM_NAME_MAX (31 characters with the "/").
pub fn posix_name(name: &str) -> Result<String> {
    let base = name.rsplit('\\').next().unwrap_or("").trim_start_matches('/');
    if base.is_empty() || base.contains('/') {
        return Err(Error::Name(format!(
            "the frame tap's name {name:?} isn't usable as shared memory here (try a plain name)"
        )));
    }
    let full = format!("/{base}");
    if full.len() > SHM_NAME_MAX {
        return Err(Error::Name(format!(
            "the frame tap's name {name:?} is too long here: as shared memory it's {full:?}, {} characters, and this \
             system allows {SHM_NAME_MAX}",
            full.len()
        )));
    }
    Ok(full)
}

/// Err unless the tap's name and its sound ring's (the name + "-audio") both fit, so a name that's too long fails at
/// once rather than when the sound starts.
pub fn check_name(name: &str) -> Result<()> {
    let base = posix_name(name)?;
    if posix_name(&format!("{name}{AUDIO_SUFFIX}")).is_err() {
        let room = SHM_NAME_MAX - 1 - AUDIO_SUFFIX.len();
        return Err(Error::Name(format!(
            "the frame tap's name {name:?} is too long here: its sound ring would be {:?} and this system allows \
             {SHM_NAME_MAX} characters, so the name can have at most {room} (after any Local\\)",
            format!("{base}{AUDIO_SUFFIX}")
        )));
    }
    Ok(())
}

/// Remove the tap `name` and its sound ring (memory already mapped stays usable until it's unmapped); for tests and
/// clean-up. Neither side does this in normal use: a reader that did would leave chiaki-ng writing into memory nobody
/// else can open.
pub fn unlink(name: &str) {
    for n in [name.to_string(), format!("{name}{AUDIO_SUFFIX}")] {
        if let Ok(p) = posix_name(&n) {
            let c = CString::new(p).unwrap();
            unsafe {
                libc::shm_unlink(c.as_ptr());
            }
        }
    }
}

/// `size` bytes of shared memory, mapped read-write. Its bytes are read and written with volatile and atomic
/// accesses at fixed little-endian offsets.
pub struct Shm {
    ptr: *mut u8,
    len: usize,
}

// The mapping is plain memory another process writes to as well; every access goes through volatile or atomic
// operations, so sharing it between threads is as safe as sharing it between processes.
unsafe impl Send for Shm {}
unsafe impl Sync for Shm {}

impl Shm {
    /// `size` bytes of the shared memory `name` (a tap name: [`posix_name`] applies), created if it isn't there.
    /// macOS sets a shared-memory object's size only once, while it's still empty, so the second side to try gets
    /// EINVAL and finds it already sized.
    pub fn open(name: &str, size: usize) -> Result<Shm> {
        let path = posix_name(name)?;
        let c = CString::new(path.clone()).unwrap();
        unsafe {
            let fd = libc::shm_open(c.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o600 as libc::c_uint);
            if fd < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            let close = |e: Error| {
                libc::close(fd);
                Err(e)
            };
            let mut st: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut st) != 0 {
                return close(std::io::Error::last_os_error().into());
            }
            if st.st_size == 0 && libc::ftruncate(fd, size as libc::off_t) != 0 {
                let e = std::io::Error::last_os_error();
                if e.raw_os_error() != Some(libc::EINVAL) {
                    // EINVAL: the other side sized it first
                    return close(e.into());
                }
            }
            // When both sides size it at once, the one that gets EINVAL can still see 0 bytes for a moment, until
            // the other's ftruncate shows (about 1 in 600 simultaneous opens here): give that up to 200 ms.
            let t_end = crate::clock::now() + 0.2;
            loop {
                if libc::fstat(fd, &mut st) != 0 {
                    return close(std::io::Error::last_os_error().into());
                }
                if st.st_size as u64 >= size as u64 || st.st_size != 0 || crate::clock::now() > t_end {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            if (st.st_size as u64) < size as u64 {
                // (macOS rounds the size up to whole pages)
                return close(Error::Io(std::io::Error::other(format!(
                    "the shared memory {path} is {} bytes, too small for the frame tap ({size}): something else made \
                     it. Remove it (sp_tap::unlink) or use another name",
                    st.st_size
                ))));
            }
            let p = libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
            libc::close(fd);
            if p == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(Shm { ptr: p as *mut u8, len: size })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn at(&self, off: usize, n: usize) -> *mut u8 {
        assert!(off + n <= self.len, "offset {off} past the mapping");
        unsafe { self.ptr.add(off) }
    }

    pub fn u32(&self, off: usize) -> u32 {
        unsafe { std::ptr::read_volatile(self.at(off, 4) as *const u32) }
    }
    pub fn i32(&self, off: usize) -> i32 {
        unsafe { std::ptr::read_volatile(self.at(off, 4) as *const i32) }
    }
    pub fn u64(&self, off: usize) -> u64 {
        unsafe { std::ptr::read_volatile(self.at(off, 8) as *const u64) }
    }
    pub fn set_u32(&self, off: usize, v: u32) {
        unsafe { std::ptr::write_volatile(self.at(off, 4) as *mut u32, v) }
    }
    pub fn set_i32(&self, off: usize, v: i32) {
        unsafe { std::ptr::write_volatile(self.at(off, 4) as *mut i32, v) }
    }
    pub fn set_u64(&self, off: usize, v: u64) {
        unsafe { std::ptr::write_volatile(self.at(off, 8) as *mut u64, v) }
    }
    pub fn atomic_u32(&self, off: usize) -> &AtomicU32 {
        assert!(off.is_multiple_of(4));
        unsafe { AtomicU32::from_ptr(self.at(off, 4) as *mut u32) }
    }
    pub fn atomic_u64(&self, off: usize) -> &AtomicU64 {
        assert!(off.is_multiple_of(8));
        unsafe { AtomicU64::from_ptr(self.at(off, 8) as *mut u64) }
    }
    /// frametap.c's XCHG: a full-barrier store.
    pub fn xchg_u32(&self, off: usize, v: u32) -> u32 {
        self.atomic_u32(off).swap(v, Ordering::SeqCst)
    }
    pub fn xchg_u64(&self, off: usize, v: u64) -> u64 {
        self.atomic_u64(off).swap(v, Ordering::SeqCst)
    }
    /// Copies `dst.len()` bytes from `off`.
    pub fn copy_out(&self, off: usize, dst: &mut [u8]) {
        let src = self.at(off, dst.len());
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) }
    }
    /// Copies `src` to `off`.
    pub fn copy_in(&self, off: usize, src: &[u8]) {
        let dst = self.at(off, src.len());
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) }
    }
}

impl Drop for Shm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// A full barrier where frametap.c has one (`__atomic_thread_fence(__ATOMIC_SEQ_CST)`) and Python calls
/// OSMemoryBarrier: Apple's chips may show another core one core's stores out of order.
#[inline]
pub fn fence() {
    std::sync::atomic::fence(Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(posix_name("Local\\chiaki-frametap").unwrap(), "/chiaki-frametap");
        assert_eq!(posix_name("Local\\smoovplay-tap").unwrap(), posix_name("/smoovplay-tap").unwrap());
        assert_eq!(posix_name("smoovplay-tap").unwrap(), "/smoovplay-tap");
        check_name("Local\\chiaki-frametap-selftest").unwrap(); // its sound ring is exactly 31
        let room = SHM_NAME_MAX - 1 - AUDIO_SUFFIX.len();
        check_name(&"x".repeat(room)).unwrap();
        let e = check_name(&format!("Local\\{}", "x".repeat(room + 1))).unwrap_err();
        assert!(e.to_string().contains("sound ring"), "{e}");
        for bad in ["", "Local\\", "a/b"] {
            assert!(posix_name(bad).is_err(), "{bad:?}");
        }
    }
}
