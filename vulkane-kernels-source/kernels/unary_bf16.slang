// Element-wise unary ops for bf16. Mirrors unary.slang's 13-op
// surface (op_id 0..12) but Slang has no native `bfloat16_t` type, so
// the kernel stores bf16 as packed u16 lanes (two per u32) and does
// the math at f32 with manual round-trip conversion at the edges.
//
// f32 -> bf16: round-to-nearest-even of the upper 16 bits of the
// f32's bit pattern (matches the `bfloat::bf16::from_f32` reference
// implementation in fuel-core-types). NaN encoding is preserved as
// per-input bits (we don't quiet sNaN — same convention as CPU).
//
// All math is f32 (the polynomial / transcendental cost dominates
// the bandwidth saving anyway), so precision matches the f32
// kernels minus the ~7-bit truncation of the bf16 result.

struct Params {
    uint n_pairs;     // pair count (= n_elements / 2; wrapper guarantees n is even)
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // 2 bf16 / u32
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

// bf16 -> f32: zero-extend 16 bits into the upper half of a u32, then
// reinterpret as f32. Preserves NaN / Inf / denormal bit patterns.
float bf16_bits_to_f32(uint b) {
    return asfloat(b << 16u);
}

// f32 -> bf16: round-to-nearest-even of the upper 16 bits.
// NaN inputs are mapped to the canonical bf16 qNaN (0x7FC0).
uint f32_to_bf16_bits(float x) {
    uint bits = asuint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;  // bottom 19 bits of f32 mantissa

    // NaN: any f32 NaN becomes a single canonical bf16 NaN. (We can't
    // preserve payload across the 16-bit truncation in general.)
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }

    // Round-to-nearest-even: bias = 0x7FFF + (bit 16 of original).
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

float apply(uint op, float x) {
    if (op == 0u)  return -x;                                        // Neg
    if (op == 1u)  return x * x;                                     // Sqr
    if (op == 2u)  return sqrt(x);                                   // Sqrt
    if (op == 3u)  return exp(x);                                    // Exp
    if (op == 4u)  return log(x);                                    // Log
    if (op == 5u)  return sin(x);                                    // Sin
    if (op == 6u)  return cos(x);                                    // Cos
    if (op == 7u)  return tanh(x);                                   // Tanh
    if (op == 8u)  return 1.0 / (1.0 + exp(-x));                     // Sigmoid
    if (op == 9u)  return x / (1.0 + exp(-x));                       // Silu
    if (op == 10u) {                                                 // Gelu (tanh approx)
        float inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + tanh(inner));
    }
    if (op == 11u) return max(x, 0.0);                               // Relu
    return (x > 0.0) ? 1.0 : 0.0;                                    // Step
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    if (pair_idx >= p.n_pairs) return;

    uint packed = input[pair_idx];
    uint lo_bits = packed & 0xFFFFu;
    uint hi_bits = packed >> 16u;

    float x0 = bf16_bits_to_f32(lo_bits);
    float x1 = bf16_bits_to_f32(hi_bits);

    float y0 = apply(p.op_id, x0);
    float y1 = apply(p.op_id, x1);

    uint out_lo = f32_to_bf16_bits(y0);
    uint out_hi = f32_to_bf16_bits(y1);
    output[pair_idx] = out_lo | (out_hi << 16u);
}
