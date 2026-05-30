// ScatterAdd along `dim` — f64. Same shape as scatter_add_f32 but
// uses u64 CAS for atomic double add. Requires shaderFloat64 +
// shaderInt64 + (Vulkan core 1.2 or VK_KHR_shader_atomic_int64).
//
// Wrapper copies base → out before dispatch.

struct Params {
    uint n_src;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>       indices;
[[vk::binding(1, 0)]] StructuredBuffer<double>     src;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t> out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>       shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>       p;

// Reinterpret double ↔ uint64_t. Slang has builtin asdouble(uint, uint)
// and asuint(double, out uint, out uint) but here we want full-64-bit
// reinterpret. The cleanest portable form is via the low/high u32
// split.
double bits_to_double(uint64_t bits) {
    uint lo = uint(bits & 0xFFFFFFFFu);
    uint hi = uint(bits >> 32);
    return asdouble(lo, hi);
}

uint64_t double_to_bits(double v) {
    uint lo, hi;
    asuint(v, lo, hi);
    return (uint64_t(hi) << 32) | uint64_t(lo);
}

void atomic_add_f64(uint idx, double add_val) {
    uint64_t old_bits = out[idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        double new_val = bits_to_double(old_bits) + add_val;
        uint64_t new_bits = double_to_bits(new_val);
        uint64_t orig;
        InterlockedCompareExchange(out[idx], old_bits, new_bits, orig);
        if (orig == old_bits) return;
        old_bits = orig;
    }
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

    uint remainder = tid;
    for (uint d = 0u; d < rank; d++) {
        uint src_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_stride *= shape_buf[e];
        }
        uint src_coord = remainder / src_stride;
        remainder -= src_coord * src_stride;

        uint dst_coord = (d == dim) ? indices[tid] : src_coord;

        uint dst_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            dst_stride *= shape_buf[rank + e];
        }
        dst_off += dst_coord * dst_stride;
    }

    atomic_add_f64(dst_off, src[tid]);
}
