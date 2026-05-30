// MaskedFill, 2-byte elements (f16, bf16, i16, u16). Pair-thread: each
// thread handles 2 elements (one u32 word). PRECONDITION: element
// count `n` must be even. Reads 2 mask bytes from `mask` (which is
// 4 u8 bytes per u32).

struct Params {
    uint n;             // element count (MUST be even)
    uint fill_value;    // low 16 bits = 2-byte fill bit pattern (caller encodes)
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   mask;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair = gid.x;
    uint n_pairs = p.n >> 1u;
    if (pair >= n_pairs) return;

    uint elem_lo = pair * 2u;
    uint elem_hi = elem_lo + 1u;

    // Read 2 mask bytes (may straddle u32 boundaries).
    uint mw_lo = mask[elem_lo >> 2u];
    uint m_lo = (mw_lo >> ((elem_lo & 3u) * 8u)) & 0xFFu;
    uint mw_hi = mask[elem_hi >> 2u];
    uint m_hi = (mw_hi >> ((elem_hi & 3u) * 8u)) & 0xFFu;

    // Read input pair (packed u32).
    uint in_word = input[pair];
    uint in_lo = in_word & 0xFFFFu;
    uint in_hi = in_word >> 16u;

    uint out_lo = (m_lo != 0u) ? (p.fill_value & 0xFFFFu) : in_lo;
    uint out_hi = (m_hi != 0u) ? (p.fill_value & 0xFFFFu) : in_hi;

    output[pair] = out_lo | (out_hi << 16u);
}
