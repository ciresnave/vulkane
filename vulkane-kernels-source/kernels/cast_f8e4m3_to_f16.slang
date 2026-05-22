// Cast F8E4M3 → f16. Reads 4 F8E4M3 per input u32, writes 2 f16 per
// output u32 (i.e. 1 input word → 2 output words). `n` must be a
// multiple of 4.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

float f8e4m3_to_f32(uint b) {
    uint sign = (b >> 7u) & 1u;
    uint e    = (b >> 3u) & 0xFu;
    uint m    = b & 0x7u;
    uint sign_bit = sign << 31u;
    if (e == 0xFu && m == 0x7u) return asfloat(sign_bit | 0x7FC00000u);
    if (e == 0u && m == 0u)     return asfloat(sign_bit);
    if (e == 0u) {
        uint lz = 0u;
        uint mm = m;
        if (mm < 4u) { mm <<= 1u; lz += 1u; }
        if (mm < 4u) { mm <<= 1u; lz += 1u; }
        int  e_f32_biased = -6 - int(lz) + 127;
        uint mant_f32 = (mm & 0x3u) << 21u;
        return asfloat(sign_bit | (uint(e_f32_biased) << 23u) | mant_f32);
    }
    int  e_unbiased = int(e) - 7;
    uint e_f32_biased = uint(e_unbiased + 127);
    uint mant_f32 = m << 20u;
    return asfloat(sign_bit | (e_f32_biased << 23u) | mant_f32);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint quad_idx = gid.x;
    uint base = quad_idx * 4u;
    if (base + 3u >= p.n) return;

    uint w = input[quad_idx];
    float f0 = f8e4m3_to_f32(w        & 0xFFu);
    float f1 = f8e4m3_to_f32((w >> 8u)  & 0xFFu);
    float f2 = f8e4m3_to_f32((w >> 16u) & 0xFFu);
    float f3 = f8e4m3_to_f32((w >> 24u) & 0xFFu);

    uint h0 = f32tof16(f0);
    uint h1 = f32tof16(f1);
    uint h2 = f32tof16(f2);
    uint h3 = f32tof16(f3);
    output[quad_idx * 2u]      = h0 | (h1 << 16u);
    output[quad_idx * 2u + 1u] = h2 | (h3 << 16u);
}
