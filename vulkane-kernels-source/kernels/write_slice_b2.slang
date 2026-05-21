// In-place rectangular slab assignment for 2-byte elements
// (covers f16, bf16). Same structure as write_slice_b4.slang but
// the source/dest buffers are accessed via `uint` and we pack 2
// elements per u32. Each thread reads 2 source elements (one pair)
// and writes them to the matching slab in dst.
//
// IMPORTANT: when `range_start[last_dim]` is odd OR `src_shape[last_dim]`
// is odd, the dst write would straddle u32 boundaries (race / overwrite
// risk). Wrapper validates and falls back to CPU for those cases.
//
// Shape buffer layout: [src_shape, dst_shape, range_start] — `rank`
// uints each, total 3 * rank uints.

struct Params {
    uint n_src_pairs;   // total pairs (n_src / 2)
    uint rank;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;        // packed half elements (2/u32)
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;        // packed half elements (2/u32)
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;  // src + dst + range_start
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;       // pair index (= flat src index / 2)
    if (tid >= p.n_src_pairs) return;

    uint rank = p.rank;
    uint src_flat_pair_start = tid * 2u;

    // Decompose flat element index → per-axis source coord.
    uint remainder = src_flat_pair_start;
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

    // src and dst contiguity guarantee — paired adjacent elements live
    // in the same u32 (the wrapper validates that range_start[last] is
    // even AND src_shape[last] is even, so the dst pair also lies on a
    // u32 boundary).
    uint src_pair = src[tid];
    uint dst_word_idx = dst_off_elems >> 1u;
    dst[dst_word_idx] = src_pair;
}
