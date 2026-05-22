// Cast F8E4M3 → bf16. Reads 4 F8E4M3 per input u32, writes 2 bf16 per
// output u32. `n` must be a multiple of 4.

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

// f32 -> bf16 (round-to-nearest-even, NaN preserved as canonical qNaN).
uint f32_to_bf16_bits(float x) {
    uint bits = asuint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return ((bits & 0x80000000u) >> 16u) | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
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

    uint h0 = f32_to_bf16_bits(f0);
    uint h1 = f32_to_bf16_bits(f1);
    uint h2 = f32_to_bf16_bits(f2);
    uint h3 = f32_to_bf16_bits(f3);
    output[quad_idx * 2u]      = h0 | (h1 << 16u);
    output[quad_idx * 2u + 1u] = h2 | (h3 << 16u);
}
