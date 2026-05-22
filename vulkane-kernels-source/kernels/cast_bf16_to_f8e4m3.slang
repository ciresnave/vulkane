// Cast bf16 → F8E4M3. Reads 2 bf16 per input u32, writes 4 F8E4M3 per
// output u32. `n` must be a multiple of 4.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // 2 bf16 / u32
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;  // 4 F8E4M3 / u32
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

float bf16_bits_to_f32(uint b) { return asfloat(b << 16u); }

uint f32_to_f8e4m3(float x) {
    uint bits = asuint(x);
    uint sign = (bits >> 31u) & 1u;
    uint exp_f32 = (bits >> 23u) & 0xFFu;
    uint mant_f32 = bits & 0x7FFFFFu;
    if (exp_f32 == 0xFFu && mant_f32 != 0u) return 0x7Fu;
    if (exp_f32 == 0xFFu)                  return (sign << 7u) | 0x7Eu;
    if (exp_f32 == 0u)                     return sign << 7u;
    int e_unbiased = int(exp_f32) - 127;
    if (e_unbiased > 8 || (e_unbiased == 8 && mant_f32 > 0xD00000u)) {
        return (sign << 7u) | 0x7Eu;
    }
    if (e_unbiased < -10) return sign << 7u;
    uint mant_with_lead = mant_f32 | 0x800000u;
    if (e_unbiased >= -6) {
        uint mant_top3  = (mant_f32 >> 20u) & 0x7u;
        uint round_bit  = (mant_f32 >> 19u) & 1u;
        uint sticky     = (mant_f32 & 0x7FFFFu) != 0u ? 1u : 0u;
        uint round_up   = (round_bit != 0u) && ((sticky != 0u) || ((mant_top3 & 1u) != 0u)) ? 1u : 0u;
        uint mant       = mant_top3 + round_up;
        int  e_out      = e_unbiased + 7;
        if (mant == 8u) { mant = 0u; e_out += 1; }
        if (e_out > 15 || (e_out == 15 && mant == 7u)) {
            return (sign << 7u) | 0x7Eu;
        }
        return (sign << 7u) | (uint(e_out) << 3u) | mant;
    } else {
        uint shift = uint(-7 - e_unbiased) + 21u;
        uint m_int     = mant_with_lead >> shift;
        uint round_bit = (mant_with_lead >> (shift - 1u)) & 1u;
        uint sticky_mask = (1u << (shift - 1u)) - 1u;
        uint sticky    = (mant_with_lead & sticky_mask) != 0u ? 1u : 0u;
        uint round_up  = (round_bit != 0u) && ((sticky != 0u) || ((m_int & 1u) != 0u)) ? 1u : 0u;
        uint m_rounded = m_int + round_up;
        if (m_rounded == 8u) return (sign << 7u) | (1u << 3u);
        return (sign << 7u) | m_rounded;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint quad_idx = gid.x;
    uint base = quad_idx * 4u;
    if (base + 3u >= p.n) return;

    uint w0 = input[quad_idx * 2u];
    uint w1 = input[quad_idx * 2u + 1u];
    float f0 = bf16_bits_to_f32(w0 & 0xFFFFu);
    float f1 = bf16_bits_to_f32(w0 >> 16u);
    float f2 = bf16_bits_to_f32(w1 & 0xFFFFu);
    float f3 = bf16_bits_to_f32(w1 >> 16u);

    uint b0 = f32_to_f8e4m3(f0);
    uint b1 = f32_to_f8e4m3(f1);
    uint b2 = f32_to_f8e4m3(f2);
    uint b3 = f32_to_f8e4m3(f3);
    output[quad_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
