// Pad with replicate (edge) mode, 4-byte elements (f32, i32, u32).
// Per-axis mapping: clamp the input coord to [0, in_dim - 1].
//
// `shape_buf` layout: [in_shape[0..rank], out_shape[0..rank],
//                     left_pad[0..rank]]. Same as pad_const_b4.

struct Params {
    uint n_out;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

uint replicate_coord(uint out_coord, uint left_pad, uint in_dim) {
    if (out_coord < left_pad) {
        return 0u;
    } else if (out_coord >= left_pad + in_dim) {
        return in_dim - 1u;
    } else {
        return out_coord - left_pad;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_out) return;

    uint rank = p.rank;
    uint remainder = tid;
    uint in_off = 0u;

    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint in_dim   = shape_buf[d];
        uint in_coord = replicate_coord(out_coord, left_pad, in_dim);

        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];
        }
        in_off += in_coord * in_stride;
    }

    dst[tid] = src[in_off];
}
