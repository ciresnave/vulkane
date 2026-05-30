// PadBackward (constant mode), 2-byte elements (f16, bf16, i16, u16).
// Pair-thread: each thread handles 2 input elements (one u32 word).
// PRECONDITION: n_in must be even.

struct Params {
    uint n_in;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   grad_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> grad_in;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

// Map one input element index to its corresponding output element offset.
uint decode_out_off(uint in_idx, uint rank) {
    uint remainder = in_idx;
    uint out_off = 0u;
    for (uint d = 0u; d < rank; d++) {
        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];
        }
        uint in_coord = remainder / in_stride;
        remainder -= in_coord * in_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint out_coord = in_coord + left_pad;

        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        out_off += out_coord * out_stride;
    }
    return out_off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint n_pairs = p.n_in >> 1u;
    if (pair_idx >= n_pairs) return;

    uint o_lo = decode_out_off(pair_idx * 2u,       p.rank);
    uint o_hi = decode_out_off(pair_idx * 2u + 1u,  p.rank);

    uint w_lo = grad_out[o_lo >> 1u];
    uint lo_bits = (o_lo & 1u) == 0u ? (w_lo & 0xFFFFu) : (w_lo >> 16u);
    uint w_hi = grad_out[o_hi >> 1u];
    uint hi_bits = (o_hi & 1u) == 0u ? (w_hi & 0xFFFFu) : (w_hi >> 16u);

    grad_in[pair_idx] = lo_bits | (hi_bits << 16u);
}
