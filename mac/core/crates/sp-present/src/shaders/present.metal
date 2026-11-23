// The video window's passes (framegen/io.py, fsr.py, fill.py, looks_gl.py's pyramid), in Metal Shading Language.
//
// Every pass draws one triangle over the viewport. Fragment shaders work in render-target pixels ([[position]]),
// with pixel centres at +0.5 and row 0 at the top, as in Metal's textures. So unlike the GL versions nothing is
// flipped, and a texture's row 0 is the picture's top everywhere. A pass that places a picture gets its rectangle's
// top-left corner (origin) and size. The viewport is only the part of it on the target, since Zoom's picture goes
// past the edges.

#include <metal_stdlib>
using namespace metal;

struct VOut { float4 pos [[position]]; };

vertex VOut fullscreen(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

constexpr sampler LIN(filter::linear, address::clamp_to_edge);

// ---- bilinear resizes the picture into its rectangle (GL's glBlitFramebuffer with GL_LINEAR)
struct Place { float2 origin; float2 size; };

fragment float4 bilinear(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant Place& P [[buffer(0)]]) {
    float2 uv = (in.pos.xy - P.origin) / P.size;
    return float4(src.sample(LIN, uv).rgb, 1.0);
}

// ---- Panorama (fill.py) bends the picture across the screen, once it's drawn fit to the screen's height into its
// own texture
struct Pano { float2 screen; float t; float pad; };

fragment float4 panorama(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant Pano& P [[buffer(0)]]) {
    float2 p = in.pos.xy / P.screen;
    float x = p.x * 2.0 - 1.0;
    float u = P.t > 0.0 ? atan(P.t * x) / atan(P.t) : x;
    return float4(src.sample(LIN, float2(u * 0.5 + 0.5, p.y)).rgb, 1.0);
}

// ---- panels. The menu, messages and the stats overlay are premultiplied BGRA images, drawn 1:1 and blended. Over an
// HDR10 picture they're converted to PQ BT.2020 with white at `white` nits (hdr.py sdr_to_pq16_image), so they look
// the same as on an SDR picture.
struct PanelP { float2 origin; uint hdr; float white; };

constant float PQ_M1 = 2610.0 / 16384.0;
constant float PQ_M2 = 2523.0 / 4096.0 * 128.0;
constant float PQ_C1 = 3424.0 / 4096.0;
constant float PQ_C2 = 2413.0 / 4096.0 * 32.0;
constant float PQ_C3 = 2392.0 / 4096.0 * 32.0;

static float3 pq_oetf(float3 y) {
    float3 p = pow(clamp(y, 0.0, 1.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2);
}

static float3 bt709_to_2020(float3 c) {
    return float3(dot(float3(0.627403896, 0.329283038, 0.043313066), c),
                  dot(float3(0.069097289, 0.919540395, 0.011362316), c),
                  dot(float3(0.016391439, 0.088013308, 0.895595253), c));
}

fragment float4 panel(VOut in [[stage_in]], texture2d<float> img [[texture(0)]], constant PanelP& P [[buffer(0)]]) {
    float4 c = img.read(uint2(in.pos.xy - P.origin));
    if (P.hdr != 0 && c.a > 0.0) {
        float3 rgb = c.rgb / c.a;
        float3 lin = pow(rgb, 2.2) * (P.white / 10000.0);
        c.rgb = pq_oetf(bt709_to_2020(lin)) * c.a;
    }
    return c;
}

// ---- the looks' blurred copy (looks_gl.py _DOWN and _UP). It halves the picture six times with a 4x4 box, the first
// time with bloom's bright pass, then goes back up, mixing each level with the one below.
static float3 bright(float3 c, float thr) {
    float br = max(max(c.r, c.g), c.b);
    float knee = 0.1;
    float soft = clamp(br - thr + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    return c * (max(soft, br - thr) / max(br, 1e-4));
}

fragment float4 pyr_down(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant float& thr [[buffer(0)]]) {
    int2 p = int2(in.pos.xy) * 2 - 1, m = int2(src.get_width(), src.get_height()) - 1;
    float3 s = float3(0.0);
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            float3 c = src.read(uint2(clamp(p + int2(x, y), int2(0), m))).rgb;
            s += thr >= 0.0 ? bright(c, thr) : c;
        }
    return float4(s / 16.0, 1.0);
}

fragment float4 pyr_up(VOut in [[stage_in]], texture2d<float> level [[texture(0)]], texture2d<float> below [[texture(1)]],
                       constant float& r [[buffer(0)]]) {
    uint2 p = uint2(in.pos.xy);
    float2 uv = (float2(p) + 0.5) / float2(level.get_width(), level.get_height());
    return float4(mix(level.read(p).rgb, below.sample(LIN, uv).rgb, r), 1.0);
}

// ---- AMD FSR 1 (framegen/shaders/fsr1/ffx_fsr1.h, MIT licence, Copyright (c) 2021 Advanced Micro Devices, Inc.).
// These are the 32-bit float paths of EASU and RCAS written out in MSL, with FSR_RCAS_DENOISE as framegen/fsr.py
// builds it.
struct FsrP { float2 in_size; float2 out_size; float2 origin; float sharpness; float pad; };

static float APrxLoRcpF1(float a) { return as_type<float>(0x7ef07ebbu - as_type<uint>(a)); }
static float APrxMedRcpF1(float a) { float b = as_type<float>(0x7ef19fffu - as_type<uint>(a)); return b * (-b * a + 2.0); }
static float APrxLoRsqF1(float a) { return as_type<float>(0x5f347d74u - (as_type<uint>(a) >> 1u)); }
static float AMin3F1(float x, float y, float z) { return min(x, min(y, z)); }
static float AMax3F1(float x, float y, float z) { return max(x, max(y, z)); }
static float3 AMin3F3(float3 x, float3 y, float3 z) { return min(x, min(y, z)); }
static float3 AMax3F3(float3 x, float3 y, float3 z) { return max(x, max(y, z)); }

static void FsrEasuTapF(thread float3& aC, thread float& aW, float2 off, float2 dir, float2 len, float lob, float clp,
                        float3 c) {
    float2 v;
    v.x = (off.x * (dir.x)) + (off.y * dir.y);
    v.y = (off.x * (-dir.y)) + (off.y * dir.x);
    v *= len;
    float d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);
    float wB = float(2.0 / 5.0) * d2 + float(-1.0);
    float wA = lob * d2 + float(-1.0);
    wB *= wB;
    wA *= wA;
    wB = float(25.0 / 16.0) * wB + float(-(25.0 / 16.0 - 1.0));
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

static void FsrEasuSetF(thread float2& dir, thread float& len, float2 pp, bool biS, bool biT, bool biU, bool biV,
                        float lA, float lB, float lC, float lD, float lE) {
    float w = 0.0;
    if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);
    if (biT) w = pp.x * (1.0 - pp.y);
    if (biU) w = (1.0 - pp.x) * pp.y;
    if (biV) w = pp.x * pp.y;
    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    lenX = APrxLoRcpF1(lenX);
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = saturate(abs(dirX) * lenX);
    lenX *= lenX;
    len += lenX * w;
    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    lenY = APrxLoRcpF1(lenY);
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = saturate(abs(dirY) * lenY);
    lenY *= lenY;
    len += lenY * w;
}

fragment float4 fsr_easu(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant FsrP& P [[buffer(0)]]) {
    // FsrEasuCon(in_size, in_size, out_size)
    float2 rcpOut = 1.0 / P.out_size, rcpIn = 1.0 / P.in_size;
    float4 con0 = float4(P.in_size * rcpOut, 0.5 * P.in_size * rcpOut - 0.5);
    float4 con1 = float4(rcpIn, float2(1.0, -1.0) * rcpIn);
    float4 con2 = float4(float2(-1.0, 2.0) * rcpIn, float2(1.0, 2.0) * rcpIn);
    float2 con3 = float2(0.0, 4.0) * rcpIn;

    uint2 ip = uint2(in.pos.xy);
    float2 pp = float2(ip) * con0.xy + con0.zw;
    float2 fp = floor(pp);
    pp -= fp;
    float2 p0 = fp * con1.xy + con1.zw;
    float2 p1 = p0 + con2.xy;
    float2 p2 = p0 + con2.zw;
    float2 p3 = p0 + con3;
    float4 bczzR = src.gather(LIN, p0, int2(0), component::x);
    float4 bczzG = src.gather(LIN, p0, int2(0), component::y);
    float4 bczzB = src.gather(LIN, p0, int2(0), component::z);
    float4 ijfeR = src.gather(LIN, p1, int2(0), component::x);
    float4 ijfeG = src.gather(LIN, p1, int2(0), component::y);
    float4 ijfeB = src.gather(LIN, p1, int2(0), component::z);
    float4 klhgR = src.gather(LIN, p2, int2(0), component::x);
    float4 klhgG = src.gather(LIN, p2, int2(0), component::y);
    float4 klhgB = src.gather(LIN, p2, int2(0), component::z);
    float4 zzonR = src.gather(LIN, p3, int2(0), component::x);
    float4 zzonG = src.gather(LIN, p3, int2(0), component::y);
    float4 zzonB = src.gather(LIN, p3, int2(0), component::z);
    float4 bczzL = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
    float4 ijfeL = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
    float4 klhgL = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
    float4 zzonL = zzonB * 0.5 + (zzonR * 0.5 + zzonG);
    float bL = bczzL.x, cL = bczzL.y, iL = ijfeL.x, jL = ijfeL.y, fL = ijfeL.z, eL = ijfeL.w;
    float kL = klhgL.x, lL = klhgL.y, hL = klhgL.z, gL = klhgL.w, oL = zzonL.z, nL = zzonL.w;
    float2 dir = float2(0.0);
    float len = 0.0;
    FsrEasuSetF(dir, len, pp, true, false, false, false, bL, eL, fL, gL, jL);
    FsrEasuSetF(dir, len, pp, false, true, false, false, cL, fL, gL, hL, kL);
    FsrEasuSetF(dir, len, pp, false, false, true, false, fL, iL, jL, kL, nL);
    FsrEasuSetF(dir, len, pp, false, false, false, true, gL, jL, kL, lL, oL);
    float2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < float(1.0 / 32768.0);
    dirR = APrxLoRsqF1(dirR);
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir *= float2(dirR);
    len = len * 0.5;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) * APrxLoRcpF1(max(abs(dir.x), abs(dir.y)));
    float2 len2 = float2(1.0 + (stretch - 1.0) * len, 1.0 + -0.5 * len);
    float lob = 0.5 + float((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = APrxLoRcpF1(lob);
    float3 min4 = min(AMin3F3(float3(ijfeR.z, ijfeG.z, ijfeB.z), float3(klhgR.w, klhgG.w, klhgB.w),
                              float3(ijfeR.y, ijfeG.y, ijfeB.y)), float3(klhgR.x, klhgG.x, klhgB.x));
    float3 max4 = max(AMax3F3(float3(ijfeR.z, ijfeG.z, ijfeB.z), float3(klhgR.w, klhgG.w, klhgB.w),
                              float3(ijfeR.y, ijfeG.y, ijfeB.y)), float3(klhgR.x, klhgG.x, klhgB.x));
    float3 aC = float3(0.0);
    float aW = 0.0;
    FsrEasuTapF(aC, aW, float2(0.0, -1.0) - pp, dir, len2, lob, clp, float3(bczzR.x, bczzG.x, bczzB.x));
    FsrEasuTapF(aC, aW, float2(1.0, -1.0) - pp, dir, len2, lob, clp, float3(bczzR.y, bczzG.y, bczzB.y));
    FsrEasuTapF(aC, aW, float2(-1.0, 1.0) - pp, dir, len2, lob, clp, float3(ijfeR.x, ijfeG.x, ijfeB.x));
    FsrEasuTapF(aC, aW, float2(0.0, 1.0) - pp, dir, len2, lob, clp, float3(ijfeR.y, ijfeG.y, ijfeB.y));
    FsrEasuTapF(aC, aW, float2(0.0, 0.0) - pp, dir, len2, lob, clp, float3(ijfeR.z, ijfeG.z, ijfeB.z));
    FsrEasuTapF(aC, aW, float2(-1.0, 0.0) - pp, dir, len2, lob, clp, float3(ijfeR.w, ijfeG.w, ijfeB.w));
    FsrEasuTapF(aC, aW, float2(1.0, 1.0) - pp, dir, len2, lob, clp, float3(klhgR.x, klhgG.x, klhgB.x));
    FsrEasuTapF(aC, aW, float2(2.0, 1.0) - pp, dir, len2, lob, clp, float3(klhgR.y, klhgG.y, klhgB.y));
    FsrEasuTapF(aC, aW, float2(2.0, 0.0) - pp, dir, len2, lob, clp, float3(klhgR.z, klhgG.z, klhgB.z));
    FsrEasuTapF(aC, aW, float2(1.0, 0.0) - pp, dir, len2, lob, clp, float3(klhgR.w, klhgG.w, klhgB.w));
    FsrEasuTapF(aC, aW, float2(1.0, 2.0) - pp, dir, len2, lob, clp, float3(zzonR.z, zzonG.z, zzonB.z));
    FsrEasuTapF(aC, aW, float2(0.0, 2.0) - pp, dir, len2, lob, clp, float3(zzonR.w, zzonG.w, zzonB.w));
    float3 pix = min(max4, max(min4, aC * float3(1.0 / aW)));
    return float4(pix, 1.0);
}

#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))

fragment float4 fsr_rcas(VOut in [[stage_in]], texture2d<float> mid [[texture(0)]], constant FsrP& P [[buffer(0)]]) {
    int2 sp = int2(in.pos.xy) - int2(P.origin);
    int2 m = int2(mid.get_width(), mid.get_height()) - 1;
    float con = exp2(-P.sharpness);
    float3 b = mid.read(uint2(clamp(sp + int2(0, -1), int2(0), m))).rgb;
    float3 d = mid.read(uint2(clamp(sp + int2(-1, 0), int2(0), m))).rgb;
    float3 e = mid.read(uint2(clamp(sp, int2(0), m))).rgb;
    float3 f = mid.read(uint2(clamp(sp + int2(1, 0), int2(0), m))).rgb;
    float3 h = mid.read(uint2(clamp(sp + int2(0, 1), int2(0), m))).rgb;
    float bR = b.r, bG = b.g, bB = b.b, dR = d.r, dG = d.g, dB = d.b, eR = e.r, eG = e.g, eB = e.b;
    float fR = f.r, fG = f.g, fB = f.b, hR = h.r, hG = h.g, hB = h.b;
    float bL = bB * 0.5 + (bR * 0.5 + bG);
    float dL = dB * 0.5 + (dR * 0.5 + dG);
    float eL = eB * 0.5 + (eR * 0.5 + eG);
    float fL = fB * 0.5 + (fR * 0.5 + fG);
    float hL = hB * 0.5 + (hR * 0.5 + hG);
    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    nz = saturate(abs(nz) * APrxMedRcpF1(AMax3F1(AMax3F1(bL, dL, eL), fL, hL) - AMin3F1(AMin3F1(bL, dL, eL), fL, hL)));
    nz = -0.5 * nz + 1.0;
    float mn4R = min(AMin3F1(bR, dR, fR), hR);
    float mn4G = min(AMin3F1(bG, dG, fG), hG);
    float mn4B = min(AMin3F1(bB, dB, fB), hB);
    float mx4R = max(AMax3F1(bR, dR, fR), hR);
    float mx4G = max(AMax3F1(bG, dG, fG), hG);
    float mx4B = max(AMax3F1(bB, dB, fB), hB);
    float2 peakC = float2(1.0, -1.0 * 4.0);
    float hitMinR = min(mn4R, eR) * (1.0 / (4.0 * mx4R));
    float hitMinG = min(mn4G, eG) * (1.0 / (4.0 * mx4G));
    float hitMinB = min(mn4B, eB) * (1.0 / (4.0 * mx4B));
    float hitMaxR = (peakC.x - max(mx4R, eR)) * (1.0 / (4.0 * mn4R + peakC.y));
    float hitMaxG = (peakC.x - max(mx4G, eG)) * (1.0 / (4.0 * mn4G + peakC.y));
    float hitMaxB = (peakC.x - max(mx4B, eB)) * (1.0 / (4.0 * mn4B + peakC.y));
    float lobeR = max(-hitMinR, hitMaxR);
    float lobeG = max(-hitMinG, hitMaxG);
    float lobeB = max(-hitMinB, hitMaxB);
    float lobe = max(float(-FSR_RCAS_LIMIT), min(AMax3F1(lobeR, lobeG, lobeB), 0.0)) * con;
    lobe *= nz;  // FSR_RCAS_DENOISE
    float rcpL = APrxMedRcpF1(4.0 * lobe + 1.0);
    float pixR = (lobe * bR + lobe * dR + lobe * hR + lobe * fR + eR) * rcpL;
    float pixG = (lobe * bG + lobe * dG + lobe * hG + lobe * fG + eG) * rcpL;
    float pixB = (lobe * bB + lobe * dB + lobe * hB + lobe * fB + eB) * rcpL;
    return float4(pixR, pixG, pixB, 1.0);
}
