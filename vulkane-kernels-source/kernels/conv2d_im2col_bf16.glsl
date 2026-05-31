#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Conv2D im2col — bf16. Rearranges NCHW bf16 input into the patches
// matrix that matmul_coop_bf16_bf16_bf16 consumes for conv2d output.
//
// Input layout:
//   x: [batch, c_in, h, w]  (NCHW row-major), bf16 stored as uint16_t.
//
// Output layout (matches the f32 im2col + fuel_conv::im2col CPU oracle):
//   patches: [batch * groups, c_in_per_group * k_h * k_w, h_out * w_out]
//
// Each thread writes one bf16 element of `patches`. Out-of-bounds
// (zero-padded) positions write 0x0000 (= +0.0 in bf16).
//
// All math is on the bf16 BIT PATTERN (uint16_t) — no f32 conversion
// is needed since im2col only rearranges values, never computes.
// That keeps the bf16 round-trip lossless even for denormals / NaN.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly  buffer XBuf  { uint16_t x[]; };
layout(set = 0, binding = 1, std430)            buffer PBuf { uint16_t patches[]; };

layout(set = 0, binding = 2, std140) uniform Params {
    uint batch;
    uint c_in;
    uint h;
    uint w;
    uint h_out;
    uint w_out;
    uint k_h;
    uint k_w;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint groups;
    uint cin_per_g;
    uint total_elements;
    uint _pad;
} p;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= p.total_elements) return;

    uint spatial = p.h_out * p.w_out;
    uint patch_dim = p.cin_per_g * p.k_h * p.k_w;

    uint spatial_idx = tid % spatial;
    uint t1 = tid / spatial;
    uint patch_row = t1 % patch_dim;
    uint bg = t1 / patch_dim;

    uint ni = bg / p.groups;
    uint g  = bg % p.groups;

    uint kx = patch_row % p.k_w;
    uint t2 = patch_row / p.k_w;
    uint ky = t2 % p.k_h;
    uint ci_in_g = t2 / p.k_h;
    uint ci = g * p.cin_per_g + ci_in_g;

    uint ow = spatial_idx % p.w_out;
    uint oh = spatial_idx / p.w_out;

    int ih_signed = int(oh * p.stride_h + ky) - int(p.pad_h);
    int iw_signed = int(ow * p.stride_w + kx) - int(p.pad_w);
    bool in_bounds =
        ih_signed >= 0 && ih_signed < int(p.h) &&
        iw_signed >= 0 && iw_signed < int(p.w);

    uint16_t val;
    if (in_bounds) {
        uint ih = uint(ih_signed);
        uint iw = uint(iw_signed);
        uint x_off = ((ni * p.c_in + ci) * p.h + ih) * p.w + iw;
        val = x[x_off];
    } else {
        val = uint16_t(0u);   // bf16 +0.0
    }

    uint out_off = (bg * patch_dim + patch_row) * spatial + spatial_idx;
    patches[out_off] = val;
}
