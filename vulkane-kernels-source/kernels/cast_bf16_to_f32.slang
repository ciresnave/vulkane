// Cast bf16 → f32. Unpack 2 bf16 per u32 from the input buffer
// (same layout matvec_bf16_b.glsl reads from). bf16-to-f32 is an
// exact bit-shift since bf16 is the upper 16 bits of f32.
//
// Requires `n` to be EVEN. One thread per input u32 (= one pair of
// bf16 elements).

struct Params {
    uint n;       // number of bf16 elements (also number of f32 elements)
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>    input;   // packed bf16, n/2 u32s
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;  // f32, n elements
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint base = pair_idx * 2u;
    if (base + 1u >= p.n) return;

    uint word = input[pair_idx];
    output[base]     = asfloat((word & 0xFFFFu) << 16);
    output[base + 1] = asfloat((word >> 16) << 16);
}
