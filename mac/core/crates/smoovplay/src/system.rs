//! What run.py gets from framegen/winutil.py on a Mac: the pipeline thread's QoS, keeping the display awake while
//! playing (a controller isn't input to macOS), and the local time for the reports.

use std::ffi::{CString, c_char, c_int, c_void};

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(class: u32, relative_priority: c_int) -> c_int;
}

const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

/// The calling thread at QoS user-interactive (the scheduler keeps it on the performance cores and ahead of other
/// work), as winutil.boost_thread does on a Mac. True if it took.
pub fn boost_thread() -> bool {
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 }
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOPMAssertionCreateWithName(kind: *const c_void, level: u32, name: *const c_void, id: *mut u32) -> i32;
    fn IOPMAssertionRelease(id: u32) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(alloc: *const c_void, s: *const c_char, encoding: u32) -> *const c_void;
    fn CFRelease(cf: *const c_void);
}

const UTF8: u32 = 0x0800_0100;

fn cfstr(s: &str) -> *const c_void {
    let c = CString::new(s).unwrap();
    unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
}

/// The display kept on (IOPMAssertion PreventUserIdleDisplaySleep) while it's held.
pub struct Awake(u32);

impl Awake {
    pub fn hold() -> Option<Awake> {
        let kind = cfstr("PreventUserIdleDisplaySleep");
        let name = cfstr("SmoovPlay is playing video");
        let mut id = 0u32;
        let rc = unsafe { IOPMAssertionCreateWithName(kind, 255, name, &mut id) };
        unsafe {
            CFRelease(kind);
            CFRelease(name);
        }
        (rc == 0).then_some(Awake(id))
    }
}

impl Drop for Awake {
    fn drop(&mut self) {
        unsafe {
            IOPMAssertionRelease(self.0);
        }
    }
}

/// time.strftime(fmt) in local time.
pub fn strftime(fmt: &str) -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let f = CString::new(fmt).unwrap();
        let mut buf = [0u8; 128];
        let n = libc::strftime(buf.as_mut_ptr() as *mut c_char, buf.len(), f.as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

/// The user's home folder written as ~ (no user name in Copy diagnostics).
pub fn mask_home(s: &str) -> String {
    match std::env::var("HOME") {
        Ok(h) if !h.is_empty() => s.replace(&h, "~"),
        _ => s.to_string(),
    }
}
