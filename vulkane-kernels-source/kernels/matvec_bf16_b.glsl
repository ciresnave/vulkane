#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Mixed-precision gemv: C = A @ B with M == 1, B stored as bf16.
// A: [1, K]     row-major f32
// B: [K, N]     row-major bf16 (packed 2-per-u32 in memory)
// C: [1, N]     row-major f32
//
// This is the decode-phase matmul for LLM inference when weights
// live on device as bf16 — the memory win that makes larger models
// fit on constrained GPUs. Matches matvec.glsl's dispatch shape and
// Params layout exactly; the backend routes to this pipeline when
// B.dtype == BF16 (and M == 1).
//
// Storage strategy: B is declared as `uint B[]` so we don't depend
// on the `VK_KHR_16bit_storage` device feature. Each u32 packs two
// bf16 values. On little-endian systems (which Vulkan targets), the
// low half of u32[i] is the bf16 at logical index 2i, the high
// half is the bf16 at 2i+1.
//
// bf16 -> f32 conversion: bf16 is exactly the top 16 bits of a
// f32 (same exponent, truncated mantissa). Extending bf16 bits to
// f32 bits is a single left-shift.

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { uint  B[]; };
layout(set = 0, binding = 2, std430) buffer          CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;       // must be 1 when this pipeline is used
    uint N;
    uint K;
    uint batch_stride_a;   // in elements of A's dtype (f32)
    uint batch_stride_b;   // in elements of B's dtype (bf16)
    uint batch_stride_c;   // in elements of C's dtype (f32)
} params;

shared float subgroup_partials[16];

// Unpack the bf16 element at linear index `i` from the u32-packed
// B buffer. Bit-shifting the 16 bf16 bits into the top half of a
// u32 and reinterpreting as float gives exact f32 extension — no
// rounding, no conversion lookup tables.
float load_bf16(uint base_u32, uint i) {
    uint word = B[base_u32 + (i >> 1u)];
    uint bits = ((i & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
    return uintBitsToFloat(bits << 16);
}

void main() {
    uint col = gl_WorkGroupID.x;
    uint batch = gl_WorkGroupID.z;
    if (col >= params.N) return;

    uint a_off = batch * params.batch_stride_a;
    // batch_stride_b is in bf16 elements; the u32 base is half that.
    uint b_off_u32 = (batch * params.batch_stride_b) >> 1u;
    uint c_off = batch * params.batch_stride_c;

    uint tid = gl_LocalInvocationID.x;

    float partial = 0.0;
    for (uint k = tid; k < params.K; k += 128u) {
        float b_val = load_bf16(b_off_u32, k * params.N + col);
        partial += A[a_off + k] * b_val;
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
