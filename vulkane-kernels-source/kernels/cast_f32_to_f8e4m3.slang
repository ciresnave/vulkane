// Cast f32 → F8E4M3 (OCP "FN" variant: 1 sign + 4 exp + 3 mantissa,
// bias 7, no infinity, single NaN at S.1111.111, max normal ±448).
// Round-to-nearest-even with saturation matches NVIDIA's __nv_cvt_*
// default behaviour and the float8::F8E4M3 reference.
//
// 4 f8e4m3 elements pack per u32 in the output buffer. One thread per
// output u32 (4 f32 inputs read, 4 f8e4m3 packed and written). `n`
// must be a multiple of 4 — wrapper validates and falls back to CPU
// otherwise.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>  input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

// Encode one f32 into 8 bits of F8E4M3 (returned in the low byte of a u32).
uint f32_to_f8e4m3(float x) {
    uint bits = asuint(x);
    uint sign = (bits >> 31u) & 1u;
    uint exp_f32 = (bits >> 23u) & 0xFFu;
    uint mant_f32 = bits & 0x7FFFFFu;

    // NaN -> canonical positive NaN (CUDA-compatible: sign not preserved).
    if (exp_f32 == 0xFFu && mant_f32 != 0u) return 0x7Fu;
    // Inf -> saturate to max normal ±448.
    if (exp_f32 == 0xFFu)                  return (sign << 7u) | 0x7Eu;
    // f32 zero or subnormal -> F8E4M3 zero (subnormal is < 2^-126, way
    // below F8 min subnormal 2^-9).
    if (exp_f32 == 0u)                     return sign << 7u;

    int e_unbiased = int(exp_f32) - 127;

    // Saturate above 464 (= 448 + half_ulp_at_448). 464 in f32:
    // exp=8, mantissa bits >= 0xD00000.
    if (e_unbiased > 8 || (e_unbiased == 8 && mant_f32 > 0xD00000u)) {
        return (sign << 7u) | 0x7Eu;
    }
    // Below smallest representable subnormal (2^-10 is half of 2^-9 —
    // rounds to 0; below 2^-10 always zero).
    if (e_unbiased < -10) return sign << 7u;

    uint mant_with_lead = mant_f32 | 0x800000u;  // 24-bit 1.mmm...

    if (e_unbiased >= -6) {
        // Normal F8E4M3 (biased exp 1..15)
        uint mant_top3  = (mant_f32 >> 20u) & 0x7u;
        uint round_bit  = (mant_f32 >> 19u) & 1u;
        uint sticky     = (mant_f32 & 0x7FFFFu) != 0u ? 1u : 0u;
        uint round_up   = (round_bit != 0u) && ((sticky != 0u) || ((mant_top3 & 1u) != 0u)) ? 1u : 0u;
        uint mant       = mant_top3 + round_up;
        int  e_out      = e_unbiased + 7;
        if (mant == 8u) { mant = 0u; e_out += 1; }
        // The (e=15, m=7) slot is NaN — saturate to (e=15, m=6) = 448.
        if (e_out > 15 || (e_out == 15 && mant == 7u)) {
            return (sign << 7u) | 0x7Eu;
        }
        return (sign << 7u) | (uint(e_out) << 3u) | mant;
    } else {
        // Subnormal F8E4M3 (biased exp 0, mantissa 0..7).
        // Right-shift mant_with_lead by (21 - 7 - e_unbiased) bits to
        // align the 3-bit subnormal mantissa at the LSB.
        uint shift = uint(-7 - e_unbiased) + 21u;
        uint m_int     = mant_with_lead >> shift;
        uint round_bit = (mant_with_lead >> (shift - 1u)) & 1u;
        uint sticky_mask = (1u << (shift - 1u)) - 1u;
        uint sticky    = (mant_with_lead & sticky_mask) != 0u ? 1u : 0u;
        uint round_up  = (round_bit != 0u) && ((sticky != 0u) || ((m_int & 1u) != 0u)) ? 1u : 0u;
        uint m_rounded = m_int + round_up;
        if (m_rounded == 8u) {
            // Subnormal rounded up to smallest normal (e=1, m=0).
            return (sign << 7u) | (1u << 3u);
        }
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

    uint b0 = f32_to_f8e4m3(input[base]);
    uint b1 = f32_to_f8e4m3(input[base + 1u]);
    uint b2 = f32_to_f8e4m3(input[base + 2u]);
    uint b3 = f32_to_f8e4m3(input[base + 3u]);
    output[quad_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
