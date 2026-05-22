// In-place rectangular slab assignment for 1-byte elements (u8 / i8 —
// int8 quant cache and ggml-style activation caches). Mirrors
// write_slice_b{2,4,8}.slang but with 4 elements packed per u32 word.
//
// IMPORTANT: each thread writes ONE u32 word into dst, which holds 4
// adjacent dst elements. For that write to be safe (no straddle / no
// race with neighboring threads), both range_start[last_dim] AND
// src_shape[last_dim] must be multiples of 4. The wrapper validates
// these and falls back to CPU otherwise — same shape as the b2
// kernel's even-cols requirement.
//
// Shape buffer layout: [src_shape, dst_shape, range_start] — `rank`
// uints each.

struct Params {
    uint n_src_quads;   // total quads (n_src / 4)
    uint rank;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;        // 4 elems / u32
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;        // 4 elems / u32
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;  // src + dst + range_start
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;       // quad index (= flat src index / 4)
    if (tid >= p.n_src_quads) return;

    uint rank = p.rank;
    uint src_flat_quad_start = tid * 4u;

    uint remainder = src_flat_quad_start;
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

    // Wrapper guarantees both range_start[last] % 4 == 0 and
    // src_shape[last] % 4 == 0, so dst_off_elems is a multiple of 4
    // and the four packed elements land in a single dst u32.
    uint src_quad = src[tid];
    uint dst_word_idx = dst_off_elems >> 2u;
    dst[dst_word_idx] = src_quad;
}
