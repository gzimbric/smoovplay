//! The convolutions, as MPSGraph executables: one per IFBlock (its two strided convolutions, eight residual
//! convolutions and its last layer, a transposed convolution that net.rs turns into a plain one) and one for the
//! encoder's first three layers, each compiled once for its input size. MPSGraph is what PyTorch's MPS backend runs convolutions on, so the same kernels (Apple's, which
//! use the GPU's matrix hardware) do the work; what's gone is the per-op dispatch PyTorch adds around each.
//!
//! Tensors are NHWC fp16 in plain MTLBuffers the Metal kernels in kernels.rs read and write, so data never leaves the
//! GPU and nothing is copied between the two.

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLDevice};
use objc2_metal_performance_shaders::{MPSCommandBuffer, MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphConvolution2DOpDescriptor, MPSGraphDevice, MPSGraphExecutable,
    MPSGraphOptimization, MPSGraphOptions, MPSGraphPaddingStyle, MPSGraphShapedType, MPSGraphTensor,
    MPSGraphTensorData, MPSGraphTensorNamedDataLayout,
};

use half::f16;

pub type Buffer = ProtocolObject<dyn MTLBuffer>;

/// A convolution's weights (OIHW, as PyTorch stores them) and bias, already in fp16.
#[derive(Clone, Debug)]
pub struct Conv {
    pub cout: usize,
    pub cin: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub w: Vec<f16>,
    pub b: Vec<f16>,
}

/// One layer of a stack: a convolution and what follows it.
#[derive(Clone, Debug)]
pub enum Layer {
    /// conv + bias, then LeakyReLU(0.2)
    ConvLrelu(Conv),
    /// RIFE's ResConv with beta folded into the weights: LeakyReLU(conv(x) + bias + x, 0.2)
    Residual(Conv),
    /// conv + bias
    Conv(Conv),
    /// ConvTranspose2d (weights (in, out, k, k) as PyTorch stores them) + bias
    Deconv(Conv),
    /// A 4x4, stride 2, padding 1 ConvTranspose2d (weights as for Deconv) as four 2x2 convolutions, one per output
    /// pixel parity, side by side in the channels: output (h, w, 4 * out), channel (py * 2 + px) * out + o holds
    /// the transposed convolution's output o at (2y + py, 2x + px). The same sums, without MPSGraph's slower
    /// transposed convolution.
    Subpixel(Conv),
}

/// A Subpixel layer's weights for output parity (py, px) as a 2x2 convolution (OIHW) and its padding (left,
/// right, top, bottom): rows (y - 1, y) take kernel rows (3, 1) for py = 0, rows (y, y + 1) take (2, 0) for py = 1.
pub fn subpixel_weights(c: &Conv, py: usize, px: usize) -> (Vec<f16>, [usize; 4]) {
    assert!(c.k == 4 && c.stride == 2 && c.pad == 1, "subpixel takes 4x4 stride 2 padding 1");
    let mut w = vec![f16::ZERO; c.cout * c.cin * 4];
    for o in 0..c.cout {
        for i in 0..c.cin {
            for a in 0..2 {
                for b in 0..2 {
                    let (ky, kx) = (3 - py - 2 * a, 3 - px - 2 * b);
                    w[((o * c.cin + i) * 2 + a) * 2 + b] = c.w[((i * c.cout + o) * 4 + ky) * 4 + kx];
                }
            }
        }
    }
    let pad = [1 - px, px, 1 - py, py];
    (w, pad)
}

/// The same transposed convolution as one 3x3 convolution with 4 * out outputs (Subpixel's layout); 2.25 times the
/// multiplies (the zeros), for comparison.
pub fn deconv_as_conv3(c: &Conv) -> Conv {
    let mut w = vec![f16::ZERO; 4 * c.cout * c.cin * 9];
    let mut b = vec![f16::ZERO; 4 * c.cout];
    for py in 0..2 {
        for px in 0..2 {
            let p = py * 2 + px;
            for o in 0..c.cout {
                b[p * c.cout + o] = c.b[o];
                for i in 0..c.cin {
                    for r in 0..3 {
                        for q in 0..3 {
                            let (ky, kx) = (py as isize + 3 - 2 * r as isize, px as isize + 3 - 2 * q as isize);
                            if (0..4).contains(&ky) && (0..4).contains(&kx) {
                                w[(((p * c.cout + o) * c.cin + i) * 3 + r) * 3 + q] =
                                    c.w[((i * c.cout + o) * 4 + ky as usize) * 4 + kx as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    Conv { cout: 4 * c.cout, cin: c.cin, k: 3, stride: 1, pad: 1, w, b }
}

fn shape(dims: &[usize]) -> Retained<MPSShape> {
    let v: Vec<Retained<NSNumber>> = dims.iter().map(|&d| NSNumber::new_usize(d)).collect();
    NSArray::from_retained_slice(&v)
}

fn halves(v: &[f16]) -> Retained<NSData> {
    let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    NSData::from_vec(bytes)
}

/// A compiled stack of layers: NHWC fp16 in, NHWC fp16 out.
pub struct Stack {
    exe: Retained<MPSGraphExecutable>,
    pub in_shape: [usize; 4],
    pub out_shape: [usize; 4],
}

// MPSGraphExecutable is thread-safe once compiled (Apple: executables may be encoded from any thread).
unsafe impl Send for Stack {}
unsafe impl Sync for Stack {}

impl Stack {
    /// Builds and compiles `layers` for an input of (h, w, cin).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, layers: &[Layer], h: usize, w: usize, cin: usize) -> Stack {
        unsafe {
            let g = MPSGraph::new();
            let in_shape = [1, h, w, cin];
            let x = g.placeholderWithShape_dataType_name(Some(&shape(&in_shape)), MPSDataType::Float16, None);
            let mut t = x.clone();
            let mut cur = in_shape;
            for layer in layers {
                let (c, y) = match layer {
                    Layer::ConvLrelu(c) | Layer::Residual(c) | Layer::Conv(c) => {
                        assert_eq!(c.cin, cur[3], "layer input channels");
                        let desc = conv_desc(c.stride, c.pad);
                        let wt = g.constantWithData_shape_dataType(
                            &halves(&c.w),
                            &shape(&[c.cout, c.cin, c.k, c.k]),
                            MPSDataType::Float16,
                        );
                        let y = g.convolution2DWithSourceTensor_weightsTensor_descriptor_name(&t, &wt, &desc, None);
                        let oh = (cur[1] + 2 * c.pad - c.k) / c.stride + 1;
                        let ow = (cur[2] + 2 * c.pad - c.k) / c.stride + 1;
                        cur = [1, oh, ow, c.cout];
                        (c, y)
                    }
                    Layer::Deconv(c) => {
                        assert_eq!(c.cin, cur[3], "deconv input channels");
                        let desc = conv_desc(c.stride, c.pad);
                        // The forward convolution's weights: O = this layer's input channels, I = its outputs.
                        let wt = g.constantWithData_shape_dataType(
                            &halves(&c.w),
                            &shape(&[c.cin, c.cout, c.k, c.k]),
                            MPSDataType::Float16,
                        );
                        let oh = (cur[1] - 1) * c.stride + c.k - 2 * c.pad;
                        let ow = (cur[2] - 1) * c.stride + c.k - 2 * c.pad;
                        cur = [1, oh, ow, c.cout];
                        let y = g.convolutionTranspose2DWithSourceTensor_weightsTensor_outputShape_descriptor_name(
                            &t,
                            &wt,
                            &shape(&cur),
                            &desc,
                            None,
                        );
                        (c, y)
                    }
                    Layer::Subpixel(c) => {
                        assert_eq!(c.cin, cur[3], "deconv input channels");
                        let mut parts = Vec::new();
                        for py in 0..2 {
                            for px in 0..2 {
                                let (w, [l, r, tp, bt]) = subpixel_weights(c, py, px);
                                let wt = g.constantWithData_shape_dataType(
                                    &halves(&w),
                                    &shape(&[c.cout, c.cin, 2, 2]),
                                    MPSDataType::Float16,
                                );
                                let desc = conv_desc_pad(1, [l, r, tp, bt]);
                                let y = g.convolution2DWithSourceTensor_weightsTensor_descriptor_name(&t, &wt, &desc, None);
                                let bias = g.constantWithData_shape_dataType(
                                    &halves(&c.b),
                                    &shape(&[1, 1, 1, c.cout]),
                                    MPSDataType::Float16,
                                );
                                parts.push(g.additionWithPrimaryTensor_secondaryTensor_name(&y, &bias, None));
                            }
                        }
                        let refs: Vec<&MPSGraphTensor> = parts.iter().map(|p| &**p).collect();
                        t = g.concatTensors_dimension_name(&NSArray::from_slice(&refs), 3, None);
                        cur = [1, cur[1], cur[2], 4 * c.cout];
                        continue;
                    }
                };
                let bias = g.constantWithData_shape_dataType(&halves(&c.b), &shape(&[1, 1, 1, c.cout]), MPSDataType::Float16);
                let y = g.additionWithPrimaryTensor_secondaryTensor_name(&y, &bias, None);
                t = match layer {
                    Layer::ConvLrelu(_) => g.leakyReLUWithTensor_alpha_name(&y, 0.2, None),
                    Layer::Residual(_) => {
                        let s = g.additionWithPrimaryTensor_secondaryTensor_name(&y, &t, None);
                        g.leakyReLUWithTensor_alpha_name(&s, 0.2, None)
                    }
                    Layer::Conv(_) | Layer::Deconv(_) => y,
                    Layer::Subpixel(_) => unreachable!(),
                };
            }
            let shaped = MPSGraphShapedType::initWithShape_dataType(
                MPSGraphShapedType::alloc(),
                Some(&shape(&in_shape)),
                MPSDataType::Float16,
            );
            let feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphShapedType>> =
                NSDictionary::from_slices(&[&*x], &[&*shaped]);
            let gdev = MPSGraphDevice::deviceWithMTLDevice(device);
            let cd = MPSGraphCompilationDescriptor::new();
            cd.setOptimizationLevel(MPSGraphOptimization::Level1);
            let exe = g.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                Some(&gdev),
                &feeds,
                &NSArray::from_slice(&[&*t]),
                None,
                Some(&cd),
            );
            exe.setOptions(MPSGraphOptions::None);
            Stack { exe, in_shape, out_shape: cur }
        }
    }

    /// Encodes the stack: `input` (in_shape) -> `output` (out_shape), both fp16 NHWC buffers.
    pub fn encode(&self, cb: &MPSCommandBuffer, input: &TensorBuf, output: &TensorBuf) {
        unsafe {
            self.exe.encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                cb,
                &NSArray::from_slice(&[&*input.data]),
                Some(&NSArray::from_slice(&[&*output.data])),
                None,
            );
        }
    }
}

fn conv_desc(stride: usize, pad: usize) -> Retained<MPSGraphConvolution2DOpDescriptor> {
    conv_desc_pad(stride, [pad; 4])
}

/// padding: left, right, top, bottom
fn conv_desc_pad(stride: usize, [l, r, t, b]: [usize; 4]) -> Retained<MPSGraphConvolution2DOpDescriptor> {
    unsafe {
        MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
            stride, stride, 1, 1, 1, l, r, t, b,
            MPSGraphPaddingStyle::Explicit,
            MPSGraphTensorNamedDataLayout::NHWC,
            MPSGraphTensorNamedDataLayout::OIHW,
        )
        .expect("convolution descriptor")
    }
}

/// An fp16 NHWC tensor in a Metal buffer, wrapped once for MPSGraph.
pub struct TensorBuf {
    pub buf: Retained<Buffer>,
    pub data: Retained<MPSGraphTensorData>,
    pub shape: [usize; 4],
}

unsafe impl Send for TensorBuf {}
unsafe impl Sync for TensorBuf {}

impl TensorBuf {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, shape4: [usize; 4]) -> TensorBuf {
        let bytes = shape4.iter().product::<usize>() * 2;
        let buf = crate::kernels::buffer(device, bytes);
        let data = unsafe {
            MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                &buf,
                &shape(&shape4),
                MPSDataType::Float16,
            )
        };
        TensorBuf { buf, data, shape: shape4 }
    }
}
