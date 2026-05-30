// Pad with reflect mode, 2-byte elements (f16, bf16, i16, u16).
// Pair-thread layout: each thread handles 2 output elements (one u32
// word). PRECONDITION: n_out must be even.

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

uint reflect_coord(uint out_coord, uint left_pad, uint in_dim) {
    if (out_coord < left_pad) {
        return left_pad - out_coord;
    } else if (out_coord >= left_pad + in_dim) {
        uint excess = out_coord - left_pad - (in_dim - 1u);
        return in_dim - 1u - excess;
    } else {
        return out_coord - left_pad;
    }
}

// Map one output element index to its corresponding input element offset.
uint decode_in_off(uint out_idx, uint rank) {
    uint remainder = out_idx;
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
        uint in_coord = reflect_coord(out_coord, left_pad, in_dim);

        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];
        }
        in_off += in_coord * in_stride;
    }
    return in_off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint n_pairs = p.n_out >> 1u;
    if (pair_idx >= n_pairs) return;

    uint out_lo = pair_idx * 2u;
    uint out_hi = out_lo + 1u;
    uint o_lo = decode_in_off(out_lo, p.rank);
    uint o_hi = decode_in_off(out_hi, p.rank);

    uint w_lo = src[o_lo >> 1u];
    uint lo_bits = (o_lo & 1u) == 0u ? (w_lo & 0xFFFFu) : (w_lo >> 16u);
    uint w_hi = src[o_hi >> 1u];
    uint hi_bits = (o_hi & 1u) == 0u ? (w_hi & 0xFFFFu) : (w_hi >> 16u);

    dst[pair_idx] = lo_bits | (hi_bits << 16u);
}
