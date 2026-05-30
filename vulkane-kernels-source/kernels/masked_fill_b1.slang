// MaskedFill, 1-byte elements (u8, i8). Quad-thread: each thread
// handles 4 elements (one u32 word). PRECONDITION: element count `n`
// must be a multiple of 4. Reads 4 mask bytes (one u32 word) and
// 4 input bytes (one u32 word).

struct Params {
    uint n;             // element count (MUST be multiple of 4)
    uint fill_value;    // low 8 bits = 1-byte fill bit pattern (caller encodes)
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   mask;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint quad = gid.x;
    uint n_quads = p.n >> 2u;
    if (quad >= n_quads) return;

    // mask aligns with quad — each quad reads one u32 from the mask buffer
    // and one u32 from input/output.
    uint in_word = input[quad];
    uint m_word  = mask[quad];
    uint fill_b  = p.fill_value & 0xFFu;

    uint b0 = (m_word        & 0xFFu) != 0u ? fill_b : (in_word        & 0xFFu);
    uint b1 = ((m_word >>  8u) & 0xFFu) != 0u ? fill_b : ((in_word >>  8u) & 0xFFu);
    uint b2 = ((m_word >> 16u) & 0xFFu) != 0u ? fill_b : ((in_word >> 16u) & 0xFFu);
    uint b3 = ((m_word >> 24u) & 0xFFu) != 0u ? fill_b : ((in_word >> 24u) & 0xFFu);

    output[quad] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
