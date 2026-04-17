#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Stride-aware gemv: C = A @ B with M == 1.
// A and B may be non-contiguous (permuted/transposed) — the kernel
// reads via per-dim strides rather than assuming row-major layout.

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { float B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint sa_batch;  uint sa_row;  uint sa_col;
    uint sb_batch;  uint sb_row;  uint sb_col;
    uint sc_batch;
    uint n_rep;
    uint _pad;
} p;

shared float subgroup_partials[16];

void main() {
    uint col = gl_WorkGroupID.x;
    uint batch = gl_WorkGroupID.z;
    if (col >= p.N) return;

    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;

    uint tid = gl_LocalInvocationID.x;

    float partial = 0.0;
    for (uint k = tid; k < p.K; k += 128u) {
        partial += A[a_off + k * p.sa_col] * B[b_off + k * p.sb_row + col * p.sb_col];
    }

    float sg_sum = subgroupAdd(partial);
    if (subgroupElect()) {
        subgroup_partials[gl_SubgroupID] = sg_sum;
    }
    barrier();

    if (gl_SubgroupID == 0u) {
        float v = 0.0;
        if (gl_SubgroupInvocationID < gl_NumSubgroups) {
            v = subgroup_partials[gl_SubgroupInvocationID];
        }
        float total = subgroupAdd(v);
        if (subgroupElect()) {
            C[c_off + col] = total;
        }
    }
}
