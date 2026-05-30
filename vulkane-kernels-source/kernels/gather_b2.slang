// Gather along `dim`, 2-byte elements (f16, bf16, i16, u16). Pair-
// thread: each thread handles 2 output elements (one u32 word).
//
// PRECONDITION: the innermost (last) output dim must be even, so
// each pair (out_idx_lo, out_idx_hi) = (2k, 2k+1) lies within the
// same `dim` slot when dim != rank-1 (i.e. coord at dim is identical
// for the two halves). When dim == rank-1 (gather along the innermost
// dim), the two halves can have DIFFERENT indices — fine, the kernel
// reads them independently.

struct Params {
    uint n_out;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;        // packed bf16/f16
[[vk::binding(1, 0)]] StructuredBuffer<uint>   indices;    // U32 (one per element)
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

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair = gid.x;
    uint n_pairs = p.n_out >> 1u;
    if (pair >= n_pairs) return;

    uint elem_lo = pair * 2u;
    uint elem_hi = elem_lo + 1u;

    uint src_off_lo = compute_src_off(elem_lo, p.rank, p.dim);
    uint src_off_hi = compute_src_off(elem_hi, p.rank, p.dim);

    // Read each source half-word.
    uint w_lo = src[src_off_lo >> 1u];
    uint bits_lo = (src_off_lo & 1u) == 0u ? (w_lo & 0xFFFFu) : (w_lo >> 16u);
    uint w_hi = src[src_off_hi >> 1u];
    uint bits_hi = (src_off_hi & 1u) == 0u ? (w_hi & 0xFFFFu) : (w_hi >> 16u);

    output[pair] = bits_lo | (bits_hi << 16u);
}
