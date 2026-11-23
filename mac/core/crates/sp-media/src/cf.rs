//! Small CoreFoundation helpers for the dictionaries VideoToolbox and CoreVideo take.

use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};

/// A CFDictionary of string keys and any values.
pub fn dict(pairs: &[(&CFString, &CFType)]) -> CFRetained<CFDictionary> {
    let keys: Vec<&CFString> = pairs.iter().map(|p| p.0).collect();
    let values: Vec<&CFType> = pairs.iter().map(|p| p.1).collect();
    let d: CFRetained<CFDictionary<CFString, CFType>> = CFDictionary::from_slices(&keys, &values);
    // (the same object, untyped)
    unsafe { CFRetained::cast_unchecked(d) }
}

pub fn num(v: i64) -> CFRetained<CFNumber> {
    CFNumber::new_i64(v)
}

pub fn yes() -> &'static CFBoolean {
    unsafe { objc2_core_foundation::kCFBooleanTrue.unwrap() }
}

/// A four-character code ('420v') as an OSType.
pub const fn fourcc(s: &[u8; 4]) -> u32 {
    ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | s[3] as u32
}

/// What an OSStatus from VideoToolbox or CoreMedia means, for messages.
pub fn status_name(status: i32) -> &'static str {
    match status {
        -12909 => "bad data (kVTVideoDecoderBadDataErr)",
        -12903 => "invalid session (kVTInvalidSessionErr)",
        -12911 => "decoder malfunction (kVTVideoDecoderMalfunctionErr)",
        -12906 => "no decoder for this format (kVTCouldNotFindVideoDecoderErr)",
        -12910 => "unsupported data (kVTVideoDecoderUnsupportedDataFormatErr)",
        -12902 => "a parameter is wrong (kVTParameterErr)",
        -12904 => "out of memory (kVTAllocationFailedErr)",
        -17694 => "a reference picture is missing (kVTVideoDecoderReferenceMissingErr)",
        -12712 => "bad parameter sets (kCMFormatDescriptionError_InvalidParameter)",
        _ => "error",
    }
}
