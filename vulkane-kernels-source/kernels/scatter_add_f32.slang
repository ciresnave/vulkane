// ScatterAdd along `dim` — f32. Output starts initialized to `base`
// (copied by the wrapper); then for each src position p, the dst
// position is the same as p except dim is replaced by indices[p].
// out[dst] += src[p], implemented via a uint CAS loop because
// VK_EXT_shader_atomic_float isn't required (works on stock Vulkan).
//
// Bindings:
//   0: indices (StructuredBuffer<uint>, src_shape)
//   1: src     (StructuredBuffer<float>, src_shape)
//   2: out     (RWStructuredBuffer<uint>, base_shape — reinterpreted)
//   3: shape_buf (src_shape[0..rank] + base_shape[0..rank])
//   4: params (uniform)

struct Params {
    uint n_src;     // total src element count = iteration bound
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   indices;
[[vk::binding(1, 0)]] StructuredBuffer<float>  src;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>   p;

// Atomic float add via uint CAS. Works on stock Vulkan (no
// VK_EXT_shader_atomic_float required).
void atomic_add_f32(uint idx, float add_val) {
    uint old_bits = out[idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        float new_val = asfloat(old_bits) + add_val;
        uint new_bits = asuint(new_val);
        uint orig;
        InterlockedCompareExchange(out[idx], old_bits, new_bits, orig);
        if (orig == old_bits) {
            return;
        }
        old_bits = orig;
    }
    // Bounded iterations: extreme contention; the final write loses
    // a value but the kernel doesn't hang. (1000 iters at workgroup
    // size 256 is far past the realistic contention envelope.)
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_src) return;

    uint rank = p.rank;
    uint dim = p.dim;
    uint dst_off = 0u;

    // Decompose tid into src coords. For each axis d, dst_coord =
    // (d == dim) ? indices[tid] : src_coord.
    uint remainder = tid;
    for (uint d = 0u; d < rank; d++) {
        uint src_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_stride *= shape_buf[e];   // src_shape[e]
        }
        uint src_coord = remainder / src_stride;
        remainder -= src_coord * src_stride;

        uint dst_coord = (d == dim) ? indices[tid] : src_coord;

        uint dst_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            dst_stride *= shape_buf[rank + e];   // base_shape[e]
        }
        dst_off += dst_coord * dst_stride;
    }

    atomic_add_f32(dst_off, src[tid]);
}
