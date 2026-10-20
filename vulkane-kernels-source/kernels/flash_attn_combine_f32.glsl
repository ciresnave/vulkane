#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// Combine pass for flash_attn_split_f32.glsl (ROADMAP item 17, H12): merges the
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
layout(set = 0, binding = 1, std430) buffer OBuf { float O[]; };
layout(set = 0, binding = 2, std140) uniform Params {
    uint D;
    uint splits;
    uint rows;
    uint _pad0;
} p;

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
            O[row * p.D + dx] = 0.0;
            continue;
        }
        float acc = 0.0;
        for (uint s = 0u; s < p.splits; ++s) {
            float l = P[base + s * stride + p.D + 1u];
            if (l != 0.0) {
                acc += P[base + s * stride + dx] * exp(P[base + s * stride + p.D] - M);
            }
        }
        O[row * p.D + dx] = acc / L;
    }
}
