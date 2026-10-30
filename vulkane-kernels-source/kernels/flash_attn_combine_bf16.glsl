#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// bf16 variant of flash_attn_combine_f32.glsl: partials stay f32, the output is
// bf16 (uint16_t), narrowed once at the store. Combine pass for flash_attn_split_f32.glsl (ROADMAP item 17, H12): merges the
// per-split partials of one (b, h_q, q_i) row in fixed split order, so the
// result is deterministic for a given `splits`:
//   M = max_s m_s;  L = sum_s l_s exp(m_s - M);  out = sum_s o_s exp(m_s - M) / L
// A split with l_s == 0 (empty or fully masked) contributes nothing. A row with
// L == 0 (every key masked) writes zeros, as flash_attn_f32.glsl does.
//
// Dispatch: (1, 1, B * Hq * Sq). `splits` is small, so each thread re-derives
// M and L itself rather than sharing them.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer PBuf { float P[]; };
layout(set = 0, binding = 1, std430) buffer OBuf { uint16_t O[]; };
layout(set = 0, binding = 2, std140) uniform Params {
    uint D;
    uint splits;
    uint rows;
    uint _pad0;
} p;

uint f32_to_bf16_bits(float x) {
    uint bits = floatBitsToUint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

void main() {
    uint row = gl_WorkGroupID.z;
    if (row >= p.rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint stride = p.D + 2u;
    uint base = row * p.splits * stride;

    float M = -3.402823e+38;
    for (uint s = 0u; s < p.splits; ++s) {
        float l = P[base + s * stride + p.D + 1u];
        if (l != 0.0) M = max(M, P[base + s * stride + p.D]);
    }
    float L = 0.0;
    for (uint s = 0u; s < p.splits; ++s) {
        float l = P[base + s * stride + p.D + 1u];
        if (l != 0.0) L += l * exp(P[base + s * stride + p.D] - M);
    }
    for (uint dx = tid; dx < p.D; dx += 256u) {
        if (L == 0.0) {
            O[row * p.D + dx] = uint16_t(0u);
            continue;
        }
        float acc = 0.0;
        for (uint s = 0u; s < p.splits; ++s) {
            float l = P[base + s * stride + p.D + 1u];
            if (l != 0.0) {
                acc += P[base + s * stride + dx] * exp(P[base + s * stride + p.D] - M);
            }
        }
        O[row * p.D + dx] = uint16_t(f32_to_bf16_bits(acc / L));
    }
}
