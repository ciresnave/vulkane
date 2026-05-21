// Cast f32 → bf16. Pack 2 bf16 per u32 in the output buffer (matches
// the storage layout that matvec_bf16_b.glsl reads from).
//
// Requires `n` to be EVEN — the wrapper validates and falls back to
// CPU for odd-length tensors. One thread per output u32 (= one pair
// of bf16 elements). bf16 truncation is bit-shift `bits >> 16` per
// IEEE-754 layout (round-toward-zero; bf16 has the same exponent as
// f32, just a 7-bit truncated mantissa).

struct Params {
    uint n;       // number of f32 elements (also number of bf16 elements)
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>  input;   // f32, n elements
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;  // packed bf16, n/2 u32s
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint base = pair_idx * 2u;
    if (base + 1u >= p.n) return;

    uint lo_bits = asuint(input[base])     >> 16;
    uint hi_bits = asuint(input[base + 1]) >> 16;
    output[pair_idx] = lo_bits | (hi_bits << 16);
}
