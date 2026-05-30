// Pad with reflect mode, 1-byte elements (u8, i8). Quad-thread layout:
// each thread handles 4 output elements (one u32 word).
// PRECONDITION: n_out must be a multiple of 4.

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

uint read_byte(uint elem_idx) {
    uint word = src[elem_idx >> 2u];
    uint shift = (elem_idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint quad_idx = gid.x;
    uint n_quads = p.n_out >> 2u;
    if (quad_idx >= n_quads) return;

    uint base = quad_idx * 4u;
    uint b0 = read_byte(decode_in_off(base + 0u, p.rank));
    uint b1 = read_byte(decode_in_off(base + 1u, p.rank));
    uint b2 = read_byte(decode_in_off(base + 2u, p.rank));
    uint b3 = read_byte(decode_in_off(base + 3u, p.rank));

    dst[quad_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
