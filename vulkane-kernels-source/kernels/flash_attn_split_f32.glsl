#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// Split-KV partial pass for the single-pass attention forward (ROADMAP item 17,
// H12). Same math and masking as flash_attn_f32.glsl, but the key range is cut
// into `splits` chunks of `chunk` keys and each (b, h_q, q_i, split) gets its own
// workgroup, so a decode step with few heads still fills the GPU. Each workgroup
// writes an UNNORMALISED partial: o[d] = sum_j exp(s_j - m) V[j][d], with the
// chunk max m and sum l = sum_j exp(s_j - m). flash_attn_combine_f32.glsl merges
// the partials in a fixed split order (deterministic).
//
// Partial layout, per (row, split): [D floats o][m][l], stride D + 2.
//   row = (b * Hq + h) * Sq + q_i
//
// Dispatch: (splits, 1, B * Hq * Sq). chunk <= MAX_CHUNK.

const uint TPB = 256u;
const uint MAX_CHUNK = 1024u;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer QBuf { float Q[]; };
layout(set = 0, binding = 1, std430) readonly buffer KBuf { float K[]; };
layout(set = 0, binding = 2, std430) readonly buffer VBuf { float V[]; };
layout(set = 0, binding = 3, std430) buffer PBuf { float P[]; };
layout(set = 0, binding = 4, std430) readonly buffer ABuf { float alibi_slopes[]; };

layout(set = 0, binding = 5, std140) uniform Params {
    uint B;
    uint Hq;
    uint Hkv;
    uint Sq;
    uint Sk;
    uint D;
    float softmax_scale;
    uint causal;
    uint use_alibi;
    uint splits;
    uint chunk;
    uint _pad0;
} p;

shared float scores[MAX_CHUNK];
shared float partial_max[TPB];
shared float partial_sum[TPB];
shared float chunk_max_shared;
shared float chunk_sum_shared;

void main() {
    uint split = gl_WorkGroupID.x;
    uint row = gl_WorkGroupID.z;
    uint qi = row % p.Sq;
    uint t1 = row / p.Sq;
    uint hi = t1 % p.Hq;
    uint bi = t1 / p.Hq;
    if (bi >= p.B) return;

    uint tid = gl_LocalInvocationID.x;
    uint groups = p.Hq / p.Hkv;
    uint kv_h = hi / groups;

    uint q_row_off = row * p.D;
    uint k_off = (bi * p.Hkv + kv_h) * p.Sk * p.D;
    uint part_off = (row * p.splits + split) * (p.D + 2u);

    uint lo = split * p.chunk;
    uint hi_k = min(p.Sk, lo + p.chunk);
    uint n = (hi_k > lo) ? (hi_k - lo) : 0u;

    float alibi_h = (p.use_alibi != 0u) ? alibi_slopes[hi] : 0.0;

    for (uint j = tid; j < n; j += TPB) {
        uint kj = lo + j;
        if (p.causal != 0u && kj > qi) {
            scores[j] = -3.402823e+38;
            continue;
        }
        float acc = 0.0;
        for (uint dx = 0u; dx < p.D; ++dx) {
            acc += Q[q_row_off + dx] * K[k_off + kj * p.D + dx];
        }
        float s = acc * p.softmax_scale;
        if (p.use_alibi != 0u) {
            s += alibi_h * (float(kj) - float(qi));
        }
        scores[j] = s;
    }
    barrier();

    float local_max = -3.402823e+38;
    for (uint j = tid; j < n; j += TPB) {
        if (scores[j] > local_max) local_max = scores[j];
    }
    partial_max[tid] = local_max;
    barrier();
    for (uint stride = TPB >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other = partial_max[tid + stride];
            if (other > partial_max[tid]) partial_max[tid] = other;
        }
        barrier();
    }
    if (tid == 0u) chunk_max_shared = partial_max[0];
    barrier();
    float m = chunk_max_shared;

    // Empty or fully masked chunk: weight zero in the combine.
    if (m < -1e30) {
        for (uint dx = tid; dx < p.D; dx += TPB) P[part_off + dx] = 0.0;
        if (tid == 0u) {
            P[part_off + p.D] = -3.402823e+38;
            P[part_off + p.D + 1u] = 0.0;
        }
        return;
    }

    float local_sum = 0.0;
    for (uint j = tid; j < n; j += TPB) {
        float e = (scores[j] < -1e30) ? 0.0 : exp(scores[j] - m);
        scores[j] = e;
        local_sum += e;
    }
    partial_sum[tid] = local_sum;
    barrier();
    for (uint stride = TPB >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partial_sum[tid] += partial_sum[tid + stride];
        barrier();
    }
    if (tid == 0u) chunk_sum_shared = partial_sum[0];
    barrier();

    for (uint dx = tid; dx < p.D; dx += TPB) {
        float acc = 0.0;
        for (uint j = 0u; j < n; ++j) {
            acc += scores[j] * V[k_off + (lo + j) * p.D + dx];
        }
        P[part_off + dx] = acc;
    }
    if (tid == 0u) {
        P[part_off + p.D] = m;
        P[part_off + p.D + 1u] = chunk_sum_shared;
    }
}
