// Pad with constant fill, 2-byte elements (f16, bf16, i16, u16).
// Storage is packed two-per-u32 (so n_out and the per-dim strides
// here are still in *element* units, but src/dst are u32 buffers).
//
// Pair-thread layout: each thread handles 2 output elements (one u32
// word) per iter. PRECONDITION: the *innermost* (last) dim of the
// output must be even. For Fuel's pad workloads — Conv2D padding,
// LLM hidden_dim padding, attention-mask padding — this is always
// the case. The wrapper validates.
//
// Logic: for each pair (idx_lo, idx_hi) at output element indices
// (2k, 2k+1) where k = thread tid, decode each independently against
// the per-dim bounds.

struct Params {
    uint n_out;       // output element count (MUST be even)
    uint rank;
    uint fill_value;  // bit pattern: 2-byte fill in low 16, duplicated to high (caller encodes)
    uint _pad0;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

// Decode one output element index → (in_bounds, in_offset in elements).
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

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint n_pairs = p.n_out >> 1u;
    if (pair_idx >= n_pairs) return;

    uint out_idx_lo = pair_idx * 2u;
    uint out_idx_hi = out_idx_lo + 1u;

    bool lo_in_bounds, hi_in_bounds;
    uint lo_off, hi_off;
    decode_pos(out_idx_lo, p.rank, lo_in_bounds, lo_off);
    decode_pos(out_idx_hi, p.rank, hi_in_bounds, hi_off);

    // Read each 2-byte lane via the src u32 buffer.
    uint lo_bits;
    uint hi_bits;
    if (lo_in_bounds) {
        uint word = src[lo_off >> 1u];
        lo_bits = (lo_off & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
    } else {
        lo_bits = p.fill_value & 0xFFFFu;
    }
    if (hi_in_bounds) {
        uint word = src[hi_off >> 1u];
        hi_bits = (hi_off & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
    } else {
        hi_bits = p.fill_value & 0xFFFFu;
    }

    dst[pair_idx] = lo_bits | (hi_bits << 16u);
}
