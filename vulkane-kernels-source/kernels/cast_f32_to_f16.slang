// Cast f32 → f16. Pack 2 f16 per u32 in the output buffer. One thread
// per output u32 (pair of f16 elements). f32-to-f16 uses Slang's
// `f32tof16` builtin (round-to-nearest-even, IEEE-754 half-float).
//
// Requires `n` even — the wrapper validates.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>  input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint base = pair_idx * 2u;
    if (base + 1u >= p.n) return;

    uint lo = f32tof16(input[base]);
    uint hi = f32tof16(input[base + 1]);
    output[pair_idx] = lo | (hi << 16);
}
