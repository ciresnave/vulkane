// MaskedFill, 4-byte elements (f32, i32, u32). One thread per element.
//
// For element i: if mask[i] != 0 → output = fill_value; else copy input.
// mask is U8 storage packed 4-per-u32 in a uint StructuredBuffer.

struct Params {
    uint n;             // element count
    uint fill_value;    // 4-byte fill bit pattern
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // u32 per element
[[vk::binding(1, 0)]] StructuredBuffer<uint>   mask;    // 4 u8 bytes per u32
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint i = gid.x;
    if (i >= p.n) return;

    uint mask_word = mask[i >> 2u];
    uint mask_byte = (mask_word >> ((i & 3u) * 8u)) & 0xFFu;

    output[i] = (mask_byte != 0u) ? p.fill_value : input[i];
}
