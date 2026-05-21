// Cast f16 → f32. Unpack 2 f16 per u32 input. One thread per input
// u32 (pair of output f32s). Uses Slang's `f16tof32` builtin.
//
// Requires `n` even.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>    input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint base = pair_idx * 2u;
    if (base + 1u >= p.n) return;

    uint word = input[pair_idx];
    output[base]     = f16tof32(word & 0xFFFFu);
    output[base + 1] = f16tof32(word >> 16);
}
