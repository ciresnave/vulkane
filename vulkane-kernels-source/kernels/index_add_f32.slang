// IndexAdd — f32. For each k in [0, n_indices), src[..., k, ...] is
// added to base[..., indices[k], ...] along the indexed axis. The
// wrapper copies base → out before dispatch; the kernel atomically
// accumulates `src` via uint CAS. Same pattern as scatter_add_f32
// (no VK_EXT_shader_atomic_float required).
//
// One thread per SRC element. Total threads = outer * n_indices * inner.

struct Params {
    uint outer_count;
    uint base_dim_size;
    uint n_indices;
    uint inner_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   indices;
[[vk::binding(1, 0)]] StructuredBuffer<float>  src;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out;   // reinterpret f32 as uint for CAS
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

void atomic_add_f32(uint idx, float add_val) {
    uint old_bits = out[idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        float new_val = asfloat(old_bits) + add_val;
        uint new_bits = asuint(new_val);
        uint orig;
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

    atomic_add_f32(dst_off, src[src_off]);
}
