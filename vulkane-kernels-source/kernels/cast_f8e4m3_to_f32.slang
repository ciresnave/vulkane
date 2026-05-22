// Cast F8E4M3 → f32. Reads 4 packed F8E4M3 elements per u32 input
// word, writes 4 f32 outputs. `n` (output element count) must be a
// multiple of 4.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // 4 F8E4M3 / u32
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

// Decode one 8-bit F8E4M3 (low byte of `b`) into f32.
float f8e4m3_to_f32(uint b) {
    uint sign = (b >> 7u) & 1u;
    uint e    = (b >> 3u) & 0xFu;
    uint m    = b & 0x7u;
    uint sign_bit = sign << 31u;

    // NaN: S.1111.111
    if (e == 0xFu && m == 0x7u) return asfloat(sign_bit | 0x7FC00000u);
    // ±0
    if (e == 0u && m == 0u)     return asfloat(sign_bit);

    if (e == 0u) {
        // Subnormal F8E4M3: value = m × 2^-9, m ∈ [1, 7].
        // Normalize m to a leading-1 form, then assemble f32.
        uint lz = 0u;
        uint mm = m;
        if (mm < 4u) { mm <<= 1u; lz += 1u; }
        if (mm < 4u) { mm <<= 1u; lz += 1u; }
        // mm ∈ [4, 7]; leading bit at position 2 (value 4).
        int  e_f32_biased = -6 - int(lz) + 127;
        uint mant_f32 = (mm & 0x3u) << 21u;
        return asfloat(sign_bit | (uint(e_f32_biased) << 23u) | mant_f32);
    }

    // Normal F8E4M3.
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
    output[base]      = f8e4m3_to_f32(w        & 0xFFu);
    output[base + 1u] = f8e4m3_to_f32((w >> 8u)  & 0xFFu);
    output[base + 2u] = f8e4m3_to_f32((w >> 16u) & 0xFFu);
    output[base + 3u] = f8e4m3_to_f32((w >> 24u) & 0xFFu);
}
