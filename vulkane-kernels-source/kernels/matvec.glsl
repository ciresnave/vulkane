#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Specialized gemv: C = A @ B with M == 1.
// A: [1, K], B: [K, N], C: [1, N]  (row-major f32, batched optional)
//
// Dispatch: (N, 1, batch_count). One workgroup per output element.
// Workgroup size: 128 threads. Each thread strides over K; partial
// dot products are reduced with subgroupAdd + shared-memory across
// subgroups. Thread 0 writes the final scalar.
//
// This is the decode-phase matmul for LLM inference (single-token
// forward). At M=1 the tiled gemm kernels waste 63/64 of their
// threads on zero rows; this kernel keeps every thread reducing.
//
// Matches matmul.wgsl's Params struct and binding layout exactly so
// the backend can route dispatches to this pipeline when M == 1
// without any graph-level changes.

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { float B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;       // must be 1 when this pipeline is used
    uint N;
    uint K;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_c;
} params;

// One shared-memory slot per subgroup. Max reasonable subgroup count
// at 128 threads: 128 (if subgroupSize == 1, impossible in practice)
// but Vulkan guarantees subgroupSize is a power of two in [1, 128].
// Size 16 covers subgroupSize >= 8; all desktop GPUs have >= 32.
shared float subgroup_partials[16];

void main() {
    uint col = gl_WorkGroupID.x;
    uint batch = gl_WorkGroupID.z;
    if (col >= params.N) return;

    uint a_off = batch * params.batch_stride_a;
    uint b_off = batch * params.batch_stride_b;
    uint c_off = batch * params.batch_stride_c;

    uint tid = gl_LocalInvocationID.x;

    // Thread-local partial dot: strided over K.
    float partial = 0.0;
    for (uint k = tid; k < params.K; k += 128u) {
        partial += A[a_off + k] * B[b_off + k * params.N + col];
    }

    // Reduce within subgroup.
    float sg_sum = subgroupAdd(partial);

    // Subgroup leader writes to shared memory.
    if (subgroupElect()) {
        subgroup_partials[gl_SubgroupID] = sg_sum;
    }
    barrier();

    // Final reduce across subgroups by subgroup 0.
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
