// PadBackward (constant mode), 1-byte elements (u8, i8). Quad-thread:
// each thread handles 4 input elements. PRECONDITION: n_in multiple
// of 4.

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

uint read_byte(uint elem_idx) {
    uint word = grad_out[elem_idx >> 2u];
    uint shift = (elem_idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint quad_idx = gid.x;
    uint n_quads = p.n_in >> 2u;
    if (quad_idx >= n_quads) return;

    uint base = quad_idx * 4u;
    uint b0 = read_byte(decode_out_off(base + 0u, p.rank));
    uint b1 = read_byte(decode_out_off(base + 1u, p.rank));
    uint b2 = read_byte(decode_out_off(base + 2u, p.rank));
    uint b3 = read_byte(decode_out_off(base + 3u, p.rank));

    grad_in[quad_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
