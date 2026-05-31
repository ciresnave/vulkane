#version 450
// FlashAttention backward — dQ, f32. One workgroup per (b, h_q, q_i)
// row, mirroring the forward kernel's parallelism. Each workgroup:
//   1. Recomputes the [Sk] score vector for this row in shared mem.
//   2. Recomputes the softmax P[qi, :].
//   3. Computes dP[kj] = dO[qi, :] · V[kj, :] for each kj.
//   4. Computes the row-correction row_dot = Σ_j' P[j'] · dP[j'].
//   5. Computes dS[kj] = (dP[kj] - row_dot) · P[kj].
//   6. Accumulates dQ[qi, dx] = scale · Σ_kj dS[kj] · K[kj, dx] for
//      each dx in [0, D).
//
// Suitable for Sk ≤ 4096; tiled online-softmax for long contexts
// is a follow-up. Same f32-only / GQA / causal / scale / alibi
// support set as flash_attn_f32; bails on window/softcap upstream.

const uint TPB = 256u;
const uint MAX_SK = 4096u;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer QBuf  { float Q[]; };
layout(set = 0, binding = 1, std430) readonly buffer KBuf  { float K[]; };
layout(set = 0, binding = 2, std430) readonly buffer VBuf  { float V[]; };
layout(set = 0, binding = 3, std430) readonly buffer dOBuf { float dO[]; };
layout(set = 0, binding = 4, std430) readonly buffer ABuf  { float alibi_slopes[]; };
layout(set = 0, binding = 5, std430) buffer dQBuf { float dQ[]; };

layout(set = 0, binding = 6, std140) uniform Params {
    uint B;
    uint Hq;
    uint Hkv;
    uint Sq;
    uint Sk;
    uint D;
    float softmax_scale;
    uint causal;
    uint use_alibi;
    uint _pad0, _pad1, _pad2;
} p;

shared float scores[MAX_SK];   // overloaded: stage 1 = pre-softmax,
                               // stage 3 = softmax P, then dS
shared float dp[MAX_SK];
shared float partial_max[TPB];
shared float partial_sum[TPB];
shared float partial_dot[TPB];
shared float row_max_shared;
shared float row_sum_shared;
shared float row_dot_shared;

void main() {
    uint linear = gl_WorkGroupID.z;
    uint qi = linear % p.Sq;
    uint t1 = linear / p.Sq;
    uint hi = t1 % p.Hq;
    uint bi = t1 / p.Hq;
    if (bi >= p.B) return;

    uint tid = gl_LocalInvocationID.x;

    uint groups = p.Hq / p.Hkv;
    uint kv_h = hi / groups;

    uint q_row_off = ((bi * p.Hq + hi) * p.Sq + qi) * p.D;
    uint k_off     = (bi * p.Hkv + kv_h) * p.Sk * p.D;
    uint v_off     = k_off;
    uint do_row_off = q_row_off;
    uint dq_row_off = q_row_off;

    float alibi_h = (p.use_alibi != 0u) ? alibi_slopes[hi] : 0.0;

    // Stage 1: scores[kj] = Q_row · K_row * scale + mask + alibi.
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        bool admissible = true;
        if (p.causal != 0u && kj > qi) admissible = false;
        if (!admissible) {
            scores[kj] = -3.402823e+38;
            continue;
        }
        float acc = 0.0;
        for (uint dx = 0u; dx < p.D; ++dx) {
            acc += Q[q_row_off + dx] * K[k_off + kj * p.D + dx];
        }
        float s = acc * p.softmax_scale;
        if (p.use_alibi != 0u) {
            float delta = float(kj) - float(qi);
            s += alibi_h * delta;
        }
        scores[kj] = s;
    }
    barrier();

    // Stage 2: row_max reduction.
    float local_max = -3.402823e+38;
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        if (scores[kj] > local_max) local_max = scores[kj];
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
    if (tid == 0u) row_max_shared = partial_max[0];
    barrier();
    float row_max = row_max_shared;

    if (row_max < -1e30) {
        // All-masked: dQ row stays zero (we don't touch it; caller
        // should zero-fill or accept that the kernel only writes
        // computed rows; we write zeros to be explicit).
        for (uint dx = tid; dx < p.D; dx += TPB) {
            dQ[dq_row_off + dx] = 0.0;
        }
        return;
    }

    // Stage 3: exp + row_sum, then divide → scores hold P[qi, :].
    float local_sum = 0.0;
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        float e = (scores[kj] < -1e30) ? 0.0 : exp(scores[kj] - row_max);
        scores[kj] = e;
        local_sum += e;
    }
    partial_sum[tid] = local_sum;
    barrier();
    for (uint stride = TPB >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        barrier();
    }
    if (tid == 0u) row_sum_shared = partial_sum[0];
    barrier();
    float row_sum = row_sum_shared;

    if (row_sum == 0.0) {
        for (uint dx = tid; dx < p.D; dx += TPB) {
            dQ[dq_row_off + dx] = 0.0;
        }
        return;
    }
    float inv_sum = 1.0 / row_sum;
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        scores[kj] *= inv_sum;     // P[qi, kj]
    }
    barrier();

    // Stage 4: dP[kj] = dO[qi, :] · V[kj, :].
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        float acc = 0.0;
        for (uint dx = 0u; dx < p.D; ++dx) {
            acc += dO[do_row_off + dx] * V[v_off + kj * p.D + dx];
        }
        dp[kj] = acc;
    }
    barrier();

    // Stage 5: row_dot = Σ_kj P[kj] · dP[kj]; reuse partial_dot.
    float local_dot = 0.0;
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        local_dot += scores[kj] * dp[kj];
    }
    partial_dot[tid] = local_dot;
    barrier();
    for (uint stride = TPB >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_dot[tid] += partial_dot[tid + stride];
        }
        barrier();
    }
    if (tid == 0u) row_dot_shared = partial_dot[0];
    barrier();
    float row_dot = row_dot_shared;

    // Stage 6: dS[kj] = (dP[kj] - row_dot) · P[kj]; overwrite
    // scores[] in place to save shared memory.
    for (uint kj = tid; kj < p.Sk; kj += TPB) {
        scores[kj] = (dp[kj] - row_dot) * scores[kj];
    }
    barrier();

    // Stage 7: dQ[qi, dx] = scale · Σ_kj dS[kj] · K[kj, dx].
    for (uint dx = tid; dx < p.D; dx += TPB) {
        float acc = 0.0;
        for (uint kj = 0u; kj < p.Sk; ++kj) {
            acc += scores[kj] * K[k_off + kj * p.D + dx];
        }
        dQ[dq_row_off + dx] = acc * p.softmax_scale;
    }
}
