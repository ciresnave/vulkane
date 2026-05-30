// MaskedFill, 8-byte elements (f64, i64). One thread per element.

struct Params {
    uint     n;
    uint     _pad;
    uint64_t fill_value;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t>   input;
[[vk::binding(1, 0)]] StructuredBuffer<uint>       mask;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>       p;

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
