// IndexAdd — f64 via u64 CAS atomic double-add. Same shape as
// index_add_f32; wrapper copies base → out before dispatch.

struct Params {
    uint outer_count;
    uint base_dim_size;
    uint n_indices;
    uint inner_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>       indices;
[[vk::binding(1, 0)]] StructuredBuffer<double>     src;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t> out;
[[vk::binding(3, 0)]] ConstantBuffer<Params>       p;

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
    uint total = p.outer_count * p.n_indices * p.inner_count;
    if (tid >= total) return;

    uint inner_per_outer = p.n_indices * p.inner_count;
    uint outer = tid / inner_per_outer;
    uint rem = tid - outer * inner_per_outer;
    uint k = rem / p.inner_count;
    uint inner = rem - k * p.inner_count;

    uint target = indices[k];
    uint src_off = (outer * p.n_indices + k) * p.inner_count + inner;
    uint dst_off = (outer * p.base_dim_size + target) * p.inner_count + inner;

    atomic_add_f64(dst_off, src[src_off]);
}
