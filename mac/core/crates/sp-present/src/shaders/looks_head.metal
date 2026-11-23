// The looks' node functions (framegen/looks_gl.py _HEAD), in Metal Shading Language. crate::looks_gpu adds a
// fragment function per pass that calls these in its nodes' order, with their settings from a float buffer.
// tests/shaders.rs holds them to framegen/looks.py's render(), the NumPy reference.

#include <metal_stdlib>
using namespace metal;

struct VOut { float4 pos [[position]]; };

// Every pass gets where this pixel is in the output image. Q is the pixel centre with row 0 at the top, and UV runs
// from 0 to 1.
struct LookK { float2 out_size; float2 origin; uint frame_no; uint pad0; uint pad1; uint pad2; };
struct Ctx { float2 Q; float2 UV; float2 out_size; uint frame_no; };

constant float3 W = float3(0.2126, 0.7152, 0.0722);
constexpr sampler LIN(filter::linear, address::clamp_to_edge);

static uint pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
static float hash3(uint x, uint y, uint z) { return float(pcg(x + pcg(y + pcg(z)))) * (1.0 / 4294967296.0); }
static float vnoise(uint x, uint y, float2 f, uint z) {
    float a = hash3(x, y, z), b = hash3(x + 1u, y, z), c = hash3(x, y + 1u, z), d = hash3(x + 1u, y + 1u, z);
    float top = a * (1.0 - f.x) + b * f.x;
    float bot = c * (1.0 - f.x) + d * f.x;
    return (top * (1.0 - f.y) + bot * f.y - 0.5) * 2.0;
}
static float3 v3(constant float* p) { return float3(p[0], p[1], p[2]); }

static float3 n_color(thread const Ctx& X, float3 c, float3 wb, float gain, float k, float sh, float hi, float fade,
                      float sat, float vib) {
    c = clamp(c * wb, 0.0, 1.0);
    c = clamp(c * gain, 0.0, 1.0);
    float3 s = c * c * (3.0 - 2.0 * c);
    c = c + k * (s - c);
    float3 x1 = 1.0 - c;
    c = c + sh * c * x1 * x1 + hi * c * c * x1;
    x1 = 1.0 - c;
    c = c + fade * x1 * x1;
    float y = dot(c, W);
    float spread = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    float f = sat * max(0.0, 1.0 + vib * (1.0 - spread));
    return y + (c - y) * f;
}
static float3 n_wheels(thread const Ctx& X, float3 c, float3 lift, float3 gain, float3 power) {
    return pow(max(gain * (c + lift * (1.0 - c)), 0.0), power);
}
static float3 n_split(thread const Ctx& X, float3 c, float3 sh, float3 hi, float bal) {
    float y = dot(c, W);
    float t = clamp(y + 0.5 * bal, 0.0, 1.0);
    float wh = t * t * (3.0 - 2.0 * t);
    return c + (1.0 - wh) * smoothstep(0.0, 0.25, y) * sh + wh * (1.0 - smoothstep(0.9, 1.0, y)) * hi;
}
static float3 n_mono(thread const Ctx& X, float3 c, float3 w, float3 tone) {
    float y = clamp(dot(c, w), 0.0, 1.0);
    return float3(y) + tone * (4.0 * y * (1.0 - y));
}
static float3 n_lut(thread const Ctx& X, float3 c, texture3d<float> t, float3 dmin, float3 dscale, float n) {
    float3 x = clamp((c - dmin) * dscale, 0.0, 1.0);
    return t.sample(LIN, (x * (n - 1.0) + 0.5) / n).rgb;
}
static float3 n_grain(thread const Ctx& X, float3 c, float a, float size, float col) {
    float px = max(0.5, size * X.out_size.y / 1080.0);
    float2 g = X.Q / px;
    float2 fl = floor(g);
    float2 f = g - fl;
    f = f * f * (3.0 - 2.0 * f);
    uint x = uint(int(fl.x)), y = uint(int(fl.y)), z = X.frame_no * 4u;
    float3 n3 = float3(vnoise(x, y, f, z), vnoise(x, y, f, z + 1u), vnoise(x, y, f, z + 2u));
    float nm = vnoise(x, y, f, z + 3u);
    float3 nn = nm * (1.0 - col) + n3 * col;
    float yl = clamp(dot(c, W), 0.0, 1.0);
    return c + a * nn * (0.25 + 3.0 * yl * (1.0 - yl));
}
static float3 n_vignette(thread const Ctx& X, float3 c, float a, float r0, float soft, float rnd) {
    float ax = 1.0 + (X.out_size.x / X.out_size.y - 1.0) * rnd;
    float2 d2 = (X.UV - 0.5) * 2.0 * float2(ax, 1.0);
    float d = length(d2) / length(float2(ax, 1.0));
    return c * (1.0 - a * smoothstep(r0 - soft, r0 + soft, d));
}
static float3 n_scanlines(thread const Ctx& X, float3 c, float a, float size) {
    float period = max(2.0, size * X.out_size.y / 1080.0);  // lines need 2 pixels or more
    return c * (1.0 - a * (0.5 + 0.5 * cos(6.2831853 * fract(X.Q.y / period))));
}
static float3 n_letterbox(thread const Ctx& X, float3 c, float aspect) {
    float2 o = X.out_size;
    float e = o.x / o.y < aspect ? abs(X.Q.y - 0.5 * o.y) - 0.5 * o.x / aspect : abs(X.Q.x - 0.5 * o.x) - 0.5 * o.y * aspect;
    return c * (1.0 - clamp(e + 0.5, 0.0, 1.0));
}
static float3 n_assist(thread const Ctx& X, float3 c, constant float* m) {
    float3 l = pow(c, float3(2.2));
    float3 o = float3(dot(float3(m[0], m[1], m[2]), l), dot(float3(m[3], m[4], m[5]), l), dot(float3(m[6], m[7], m[8]), l));
    return pow(clamp(o, 0.0, 1.0), float3(1.0 / 2.2));
}

static float3 s_bloom(thread const Ctx& X, float3 c, texture2d<float> src, texture2d<float> glow, float amount, float3 tint) {
    return c + amount * glow.sample(LIN, X.UV).rgb * tint;
}
static float3 s_halation(thread const Ctx& X, float3 c, texture2d<float> src, texture2d<float> glow, float amount, float3 tint) {
    return c + amount * glow.sample(LIN, X.UV).rgb * tint;
}
static float3 s_blur(thread const Ctx& X, float3 c, texture2d<float> src, texture2d<float> glow) {
    return glow.sample(LIN, X.UV).rgb;
}
static float3 s_clarity(thread const Ctx& X, float3 c, texture2d<float> src, texture2d<float> glow, float amount) {
    return c + amount * (c - glow.sample(LIN, X.UV).rgb);
}
static float3 s_ca(thread const Ctx& X, float3 c, texture2d<float> src, texture2d<float> glow, float k) {
    float2 off = (X.UV - 0.5) * k;
    return float3(src.sample(LIN, X.UV + off).r, c.g, src.sample(LIN, X.UV - off).b);
}
static float3 s_sharpen(thread const Ctx& X, float3 c, texture2d<float> src, texture2d<float> glow, float amount) {
    int2 p = int2(X.Q), m = int2(src.get_width(), src.get_height()) - 1;
    float3 nb = src.read(uint2(clamp(p + int2(1, 0), int2(0), m))).rgb
              + src.read(uint2(clamp(p - int2(1, 0), int2(0), m))).rgb
              + src.read(uint2(clamp(p + int2(0, 1), int2(0), m))).rgb
              + src.read(uint2(clamp(p - int2(0, 1), int2(0), m))).rgb;
    return c + amount * (c - 0.25 * nb);
}
