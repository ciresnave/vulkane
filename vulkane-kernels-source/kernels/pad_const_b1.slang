// Pad with constant fill, 1-byte elements (u8, i8). Storage as
// packed-u32 (4 bytes per u32 word). Quad-thread layout: each thread
// handles 4 output bytes (one u32 word).
//
// PRECONDITION: n_out (total output element count) must be a multiple
// of 4 — and the *innermost* dim of the output must be a multiple of
// 4 (so each per-row write lands on u32-aligned boundaries). Fuel's
// u8/i8 workloads (image preprocessing, padded ImageNet batches,
// padded byte-tensor activations) typically meet this.

struct Params {
    uint n_out;
    uint rank;
    uint fill_value;   // 4 fill bytes replicated to u32 (caller encodes)
    uint _pad0;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

// Decode one output element index → (in_bounds, in_off in element units).
void decode_pos(uint out_idx, uint rank, out bool in_bounds, out uint in_off) {
    uint remainder = out_idx;
    in_off = 0u;
    in_bounds = true;
    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint in_dim = shape_buf[d];
        if (out_coord < left_pad || out_coord >= left_pad + in_dim) {
            in_bounds = false;
        } else {
            uint in_coord = out_coord - left_pad;
            uint in_stride = 1u;
            for (uint e = d + 1u; e < rank; e++) {
                in_stride *= shape_buf[e];
            }
            in_off += in_coord * in_stride;
        }
    }
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
    uint fill_byte = p.fill_value & 0xFFu;

    uint b0, b1, b2, b3;
    bool ib0, ib1, ib2, ib3;
    uint o0, o1, o2, o3;
    decode_pos(base + 0u, p.rank, ib0, o0);
    decode_pos(base + 1u, p.rank, ib1, o1);
    decode_pos(base + 2u, p.rank, ib2, o2);
    decode_pos(base + 3u, p.rank, ib3, o3);

    b0 = ib0 ? read_byte(o0) : fill_byte;
    b1 = ib1 ? read_byte(o1) : fill_byte;
    b2 = ib2 ? read_byte(o2) : fill_byte;
    b3 = ib3 ? read_byte(o3) : fill_byte;

    dst[quad_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
