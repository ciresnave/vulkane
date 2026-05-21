// In-place rectangular slab assignment for 8-byte elements
// (covers f64, i64). Same structure as write_slice_b4.slang but
// accesses each element as a uvec2 (two u32s). One thread per
// source element; reads 2 u32s, writes them at the dst slot.
//
// Shape buffer layout: [src_shape, dst_shape, range_start] —
// `rank` uints each.

struct Params {
    uint n_src;
    uint rank;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;        // 2 u32s per element
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_src) return;

    uint rank = p.rank;
    uint remainder = tid;
    uint dst_off_elems = 0u;

    for (uint d = 0u; d < rank; d++) {
        uint src_dim_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_dim_stride *= shape_buf[e];
        }
        uint coord = remainder / src_dim_stride;
        remainder -= coord * src_dim_stride;

        uint dst_dim_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            dst_dim_stride *= shape_buf[rank + e];
        }
        uint dst_coord = coord + shape_buf[2u * rank + d];
        dst_off_elems += dst_coord * dst_dim_stride;
    }

    uint src_u32_base = tid * 2u;
    uint dst_u32_base = dst_off_elems * 2u;
    dst[dst_u32_base + 0u] = src[src_u32_base + 0u];
    dst[dst_u32_base + 1u] = src[src_u32_base + 1u];
}
