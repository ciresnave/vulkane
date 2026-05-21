// In-place rectangular slab assignment for 4-byte elements
// (covers f32, i32, u32). Reads `n_src` elements from `src` (which
// is contiguous in its own rank-N shape) and writes them into the
// matching slab inside `dst` (which is rank-N contiguous, larger
// shape). Used by `Op::WriteSlice` — backs persistent KV-cache
// writes in the new lazy-graph KV story.
//
// Shape buffer layout (`shape_buf`):
//   [src_shape[0..rank],
//    dst_shape[0..rank],
//    range_start[0..rank]]
//
// Supports rank up to 8 (covers every shape Fuel uses in practice).
// Source slab geometry: each axis i contributes `src_shape[i]`
// elements; dest insertion point per axis is `range_start[i]`. The
// caller has already validated that `range_start[i] + src_shape[i]
// <= dst_shape[i]`.

struct Params {
    uint n_src;   // src total element count (also iteration bound)
    uint rank;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   src;        // source (contiguous, src_shape)
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;        // destination (contiguous, dst_shape)
[[vk::binding(2, 0)]] StructuredBuffer<uint>    shape_buf;  // src_shape + dst_shape + range_start
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_src) return;

    uint rank = p.rank;

    // Decompose tid into per-axis source coords (row-major).
    // Build src_stride[d] = product(src_shape[d+1..rank]) on the fly,
    // matching strided_copy.slang's two-loop structure.
    uint remainder = tid;
    uint dst_off = 0u;

    for (uint d = 0u; d < rank; d++) {
        uint src_dim_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_dim_stride *= shape_buf[e];   // src_shape[e]
        }
        uint coord = remainder / src_dim_stride;
        remainder -= coord * src_dim_stride;

        // dst_stride[d] = product(dst_shape[d+1..rank]).
        uint dst_dim_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            dst_dim_stride *= shape_buf[rank + e];   // dst_shape[e]
        }
        uint dst_coord = coord + shape_buf[2u * rank + d];   // + range_start[d]
        dst_off += dst_coord * dst_dim_stride;
    }

    dst[dst_off] = src[tid];
}
