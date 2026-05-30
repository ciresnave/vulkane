// Gather along `dim`, 1-byte elements (u8, i8). Quad-thread: each
// thread handles 4 output elements (one u32 word).
//
// PRECONDITION: `n_out` must be a multiple of 4. As with gather_b2,
// the 4 elements may have different `indices[]` values — fine.

struct Params {
    uint n_out;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   indices;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>   p;

uint compute_src_off(uint out_idx, uint rank, uint dim) {
    uint remainder = out_idx;
    uint src_off = 0u;
    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint src_coord = (d == dim) ? indices[out_idx] : out_coord;

        uint src_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_stride *= shape_buf[e];
        }
        src_off += src_coord * src_stride;
    }
    return src_off;
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
    uint quad = gid.x;
    uint n_quads = p.n_out >> 2u;
    if (quad >= n_quads) return;

    uint base = quad * 4u;
    uint o0 = compute_src_off(base + 0u, p.rank, p.dim);
    uint o1 = compute_src_off(base + 1u, p.rank, p.dim);
    uint o2 = compute_src_off(base + 2u, p.rank, p.dim);
    uint o3 = compute_src_off(base + 3u, p.rank, p.dim);

    uint b0 = read_byte(o0);
    uint b1 = read_byte(o1);
    uint b2 = read_byte(o2);
    uint b3 = read_byte(o3);

    output[quad] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
