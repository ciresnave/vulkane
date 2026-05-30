// Gather along `dim`, 4-byte elements (f32, i32, u32). One thread per
// output element.
//
// Semantics: source and output shapes agree on every dim except
// `dim`. For each output position with coords (c0, c1, ..., c_dim, ...,
// c_{rank-1}), the source coord for `dim` is read from
// `indices[out_idx]`; all other coords are the same.
//
// `shape_buf` layout: [source_shape[0..rank], output_shape[0..rank]].
// Rank up to 8.

struct Params {
    uint n_out;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;        // u32 per element
[[vk::binding(1, 0)]] StructuredBuffer<uint>   indices;    // U32, output_shape
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
// shape_buf packed as a separate storage buffer alongside Params;
// reuse layout_4s1u? Actually layout_3s1u has 3 storage + 1 uniform.
// We need 4 storage buffers (src, indices, output, shape_buf), so
// pass shape_buf as a 4th storage. Wrapper picks layout_4s1u.
[[vk::binding(3, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_out) return;

    uint rank = p.rank;
    uint dim = p.dim;
    uint src_off = 0u;

    // Decompose tid into output coords. For dim != p.dim, use the
    // coord directly as the source coord. For dim == p.dim, replace
    // with `indices[tid]`.
    uint remainder = tid;
    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];   // output_shape[e]
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint src_coord = (d == dim) ? indices[tid] : out_coord;

        uint src_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_stride *= shape_buf[e];   // source_shape[e]
        }
        src_off += src_coord * src_stride;
    }

    output[tid] = src[src_off];
}
