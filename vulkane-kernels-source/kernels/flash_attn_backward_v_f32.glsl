#version 450
// FlashAttention backward — dV, f32. One workgroup per (b, h_kv, k_j)
// output column. Each workgroup loops over (h_q in group, q_i),
// cooperatively recomputing the softmax row and accumulating
// dV[k_j, dx] += P[q_i, k_j] · dO[q_i, dx]. Each thread owns exactly
// one d_x (tid < D); threads with tid >= D idle in the accumulation
// step but still participate in the cooperative softmax recompute.
//
// Constraint: D ≤ TPB = 256. Wrapper bails if larger.

const uint TPB = 256u;
const uint MAX_SK = 4096u;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer QBuf  { float Q[]; };
layout(set = 0, binding = 1, std430) readonly buffer KBuf  { float K[]; };
layout(set = 0, binding = 2, std430) readonly buffer VBuf  { float V[]; };
layout(set = 0, binding = 3, std430) readonly buffer dOBuf { float dO[]; };
layout(set = 0, binding = 4, std430) readonly buffer ABuf  { float alibi_slopes[]; };
layout(set = 0, binding = 5, std430) buffer dVBuf { float dV[]; };

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

shared float scores[MAX_SK];
shared float partial_max[TPB];
shared float partial_sum[TPB];
shared float row_max_shared;
shared float row_sum_shared;

void main() {
    uint linear = gl_WorkGroupID.z;
    uint kj = linear % p.Sk;
    uint t1 = linear / p.Sk;
    uint h_kv = t1 % p.Hkv;
    uint bi = t1 / p.Hkv;
    if (bi >= p.B) return;

    uint tid = gl_LocalInvocationID.x;
    uint groups = p.Hq / p.Hkv;
    bool active_thread = tid < p.D;
    float dv_accum = 0.0;

    uint k_off = (bi * p.Hkv + h_kv) * p.Sk * p.D;
    uint dv_off = k_off;

    for (uint g = 0u; g < groups; ++g) {
        uint h_q = h_kv * groups + g;
        uint q_h_off = (bi * p.Hq + h_q) * p.Sq * p.D;
        uint do_h_off = q_h_off;
        float alibi_h = (p.use_alibi != 0u) ? alibi_slopes[h_q] : 0.0;

        for (uint qi = 0u; qi < p.Sq; ++qi) {
            uint q_row_off = q_h_off + qi * p.D;
            uint do_row_off = do_h_off + qi * p.D;

            for (uint kk = tid; kk < p.Sk; kk += TPB) {
                bool admissible = true;
                if (p.causal != 0u && kk > qi) admissible = false;
                if (!admissible) {
                    scores[kk] = -3.402823e+38;
                    continue;
                }
                float acc = 0.0;
                for (uint dx = 0u; dx < p.D; ++dx) {
                    acc += Q[q_row_off + dx] * K[k_off + kk * p.D + dx];
                }
                float s = acc * p.softmax_scale;
                if (p.use_alibi != 0u) {
                    float delta = float(kk) - float(qi);
                    s += alibi_h * delta;
                }
                scores[kk] = s;
            }
            barrier();

            float local_max = -3.402823e+38;
            for (uint kk = tid; kk < p.Sk; kk += TPB) {
                if (scores[kk] > local_max) local_max = scores[kk];
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
                barrier();
                continue;
            }

            float local_sum = 0.0;
            for (uint kk = tid; kk < p.Sk; kk += TPB) {
                float e = (scores[kk] < -1e30) ? 0.0 : exp(scores[kk] - row_max);
                scores[kk] = e;
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
                barrier();
                continue;
            }
            float inv_sum = 1.0 / row_sum;
            for (uint kk = tid; kk < p.Sk; kk += TPB) {
                scores[kk] *= inv_sum;     // scores[k] = P[qi, k]
            }
            barrier();

            float p_kj = scores[kj];
            if (active_thread && p_kj != 0.0) {
                dv_accum += p_kj * dO[do_row_off + tid];
            }
            barrier();
        }
    }

    if (active_thread) {
        dV[dv_off + kj * p.D + tid] = dv_accum;
    }
}
