//! One Metal device and command queue for the whole core, so frames made by one piece (the decoder) can be read by
//! the next (RIFE, the window) without copies. Metal objects are safe to use from any thread.

use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};

use crate::{Error, Result};

pub type Device = ProtocolObject<dyn MTLDevice>;
pub type Queue = ProtocolObject<dyn MTLCommandQueue>;
pub type Library = ProtocolObject<dyn MTLLibrary>;
pub type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

pub struct Gpu {
    pub device: Retained<Device>,
    pub queue: Retained<Queue>,
}

// Metal's devices and queues are thread-safe (Apple: "MTLDevice ... can be used from any thread").
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static SHARED: OnceLock<Gpu> = OnceLock::new();

impl Gpu {
    /// The Mac's GPU and the core's command queue, made on first use. Panics only if the Mac has no Metal GPU
    /// (every Apple silicon Mac has one).
    pub fn shared() -> &'static Gpu {
        SHARED.get_or_init(|| {
            let device = MTLCreateSystemDefaultDevice().expect("no Metal GPU");
            let queue = device.newCommandQueue().expect("no Metal command queue");
            Gpu { device, queue }
        })
    }

    /// The GPU's name, e.g. "Apple M5 Max".
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Compiles Metal Shading Language source into a library.
    pub fn library(&self, source: &str) -> Result<Retained<Library>> {
        self.device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(|e| Error::Metal(format!("compiling shaders: {}", e.localizedDescription())))
    }

    /// A compute pipeline for `function` in `library`.
    pub fn pipeline(&self, library: &Library, function: &str) -> Result<Retained<Pipeline>> {
        let f = library
            .newFunctionWithName(&NSString::from_str(function))
            .ok_or_else(|| Error::Metal(format!("no shader function {function}")))?;
        self.device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| Error::Metal(format!("pipeline {function}: {}", e.localizedDescription())))
    }
}
