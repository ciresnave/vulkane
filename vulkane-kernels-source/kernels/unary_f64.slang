// Element-wise unary ops for f64. Mirrors unary.slang's 13-op
// surface (op_id 0..12) operating on `double` (shaderFloat64).
//
// Transcendentals (Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu) are
// implemented via polynomial approximations because the natural
// SPIR-V path is blocked:
//
//  - GLSL.std.450's transcendental opcodes (Exp/Log/Sin/Cos/Tan/Tanh)
//    are spec-restricted to 16/32-bit float inputs. They cannot
//    accept `double`.
//  - OpenCL.std's transcendentals do accept double, but the NVIDIA
//    Windows Vulkan driver rejects OpenCL.std at vkCreateComputePipeline
//    (verified empirically with a minimal kernel).
//
// So we implement the transcendentals in pure Slang using only
// arithmetic (OpFAdd/OpFMul/OpFDiv) — works on any Vulkan driver
// that exposes shaderFloat64. Target precision: ~1e-12 relative
// over the natural input range (sufficient for ML / scientific
// compute use; not full ULP-correct, which would require fdlibm-
// style implementation with Cody-Waite range reduction and
// double-double accumulation).

struct Params {
    uint n;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     p;

// ----- Constants ---------------------------------------------------------
//
// All double-precision constants are written as IEEE 754 hex floating-
// point literals. Slang's `L` suffix on decimal literals is silently
// ignored — `0.6931471805599453L` gets parsed as f32 first then
// promoted, emitting `0.69314718246459961` (only 8 significant digits
// preserved). Hex literals like `0x1.62e42fefa39efp-1L` are parsed
// directly as IEEE 754 bit patterns and round-trip cleanly.

static const double LN2         = 0x1.62e42fefa39efp-1L;  // ln(2)
static const double INV_LN2     = 0x1.71547652b82fep+0L;  // 1/ln(2)
static const double PI          = 0x1.921fb54442d18p+1L;
static const double PI_2        = 0x1.921fb54442d18p+0L;
static const double TWO_PI      = 0x1.921fb54442d18p+2L;
static const double SQRT2       = 0x1.6a09e667f3bcdp+0L;

// Cody-Waite splits — bottom 19 mantissa bits zeroed, so for any
// integer k with |k| < 2^20 the product k * X_HI is exact in f64.
// Then (x - k*X_HI) - k*X_LO preserves ~52 bits across the reduction.
// LN2 = LN2_HI + LN2_LO (exact at f64 precision).
static const double LN2_HI      = 0x1.62e42fef80000p-1L;
static const double LN2_LO      = 0x1.1cf7800000000p-36L;
// π/2 = PI_2_HI + PI_2_LO (exact at f64 precision).
static const double PI_2_HI     = 0x1.921fb54400000p+0L;
static const double PI_2_LO     = 0x1.0b46000000000p-34L;
// 2π for stage-1 sin/cos reduction (full 2π = TWO_PI_HI + TWO_PI_LO)
static const double TWO_PI_HI   = 0x1.921fb54400000p+2L;
static const double TWO_PI_LO   = 0x1.0b46000000000p-32L;

// ----- Helper: 2^k for integer k in [-1022, 1023] ------------------------
//
// IEEE 754 double: bit pattern with biased exponent (k + 1023),
// zero mantissa, sign bit zero. The high 32 bits are
// (k + 1023) << 20; the low 32 bits are zero.
//
// asdouble(lowbits, highbits) is the HLSL/Slang builtin that packs
// two u32 into a double (no extended-instruction-set dependency —
// lowers to OpBitcast in SPIR-V).
double pow2_int(int k) {
    uint hi = uint(k + 1023) << 20;
    return asdouble(0u, hi);
}

// ----- exp(x) ------------------------------------------------------------
//
// Range reduction: x = k*ln(2) + r where k = round(x / ln(2)) and
// |r| <= ln(2)/2 ≈ 0.347. Uses Cody-Waite split of ln(2) so that
// the reduction loses ≤1 ULP across the full operating range:
//   r = (x - k*LN2_HI) - k*LN2_LO
// Then exp(x) = 2^k * exp(r), with exp(r) approximated by a
// 14-term Taylor (through r^14, c_n = 1/n!). Worst-case truncation
// for |r| <= ln(2)/2:
//   (ln(2)/2)^15 / 15! ≈ 5.5e-15
// well below 1 ULP for any non-tiny result.
double poly_exp_d(double x) {
    // Overflow / underflow saturation. exp(709) ≈ 8e307 (max double),
    // exp(-746) ≈ 5e-324 (smallest subnormal).
    if (x > 709.0L)  return 1.0L / 0.0L;   // +inf
    if (x < -745.0L) return 0.0L;

    double k_d = floor(x * INV_LN2 + 0.5L);
    int k = int(k_d);
    // Cody-Waite: k_d * LN2_HI is exact for |k| < 2^20 (LN2_HI has
    // 19 trailing zero mantissa bits). Subtracting recovers all bits.
    double r = (x - k_d * LN2_HI) - k_d * LN2_LO;

    // 14-term Taylor of exp(r) via Horner; c_n = 1/n! as hex literals.
    double poly = 0x1.93974a8c07c9dp-37L;                // 1/14!
    poly = 0x1.6124613a86d09p-33L  + r * poly;           // 1/13!
    poly = 0x1.1eed8eff8d898p-29L  + r * poly;           // 1/12!
    poly = 0x1.ae64567f544e4p-26L  + r * poly;           // 1/11!
    poly = 0x1.27e4fb7789f5cp-22L  + r * poly;           // 1/10!
    poly = 0x1.71de3a556c734p-19L  + r * poly;           // 1/9!
    poly = 0x1.a01a01a01a01ap-16L  + r * poly;           // 1/8!
    poly = 0x1.a01a01a01a01ap-13L  + r * poly;           // 1/7!
    poly = 0x1.6c16c16c16c17p-10L  + r * poly;           // 1/6!
    poly = 0x1.1111111111111p-7L   + r * poly;           // 1/5!
    poly = 0x1.5555555555555p-5L   + r * poly;           // 1/4!
    poly = 0x1.5555555555555p-3L   + r * poly;           // 1/3!
    poly = 0x1.0000000000000p-1L   + r * poly;           // 1/2!
    poly = 0x1.0000000000000p+0L   + r * poly;           // 1/1!
    poly = 0x1.0000000000000p+0L   + r * poly;           // 1/0!

    return poly * pow2_int(k);
}

// ----- log(x) ------------------------------------------------------------
//
// Range reduction: x = m * 2^e where 1 <= m < 2 (extracted via bit
// manipulation). If m > sqrt(2), divide by 2 and increment e so that
// m is brought to [sqrt(2)/2, sqrt(2)) — the symmetric interval that
// keeps |f = (m-1)/(m+1)| small. Then
//   log(x) = e*ln(2) + log(m)
//   log(m) = 2 * (f + f³/3 + f⁵/5 + f⁷/7 + ...)
// 8 terms of the atanh series give worst-case error ≈ 1e-13 over
// |f| <= (sqrt(2)-1)/(sqrt(2)+1) ≈ 0.172.
double poly_log_d(double x) {
    if (x < 0.0)  return 0.0 / 0.0;   // NaN
    if (x == 0.0) return -1.0 / 0.0;  // -inf

    // Extract IEEE 754 biased exponent + raw mantissa.
    uint lo, hi;
    asuint(x, lo, hi);
    int e = int((hi >> 20) & 0x7FFu) - 1023;

    // Reset exponent to bias (1023) so m = mantissa * 2^0 in [1, 2).
    uint m_hi = (hi & 0x000FFFFFu) | (1023u << 20);
    double m = asdouble(lo, m_hi);

    // Bring m into [sqrt(2)/2, sqrt(2)) so |f| stays bounded.
    if (m > SQRT2) {
        m *= 0.5;
        e += 1;
    }

    double f = (m - 0x1.0000000000000p+0L) / (m + 0x1.0000000000000p+0L);
    double f2 = f * f;

    // Horner: log(m) = 2f * sum_{k=0}^{7} f^{2k} / (2k+1)
    double poly = 0x1.1111111111111p-4L;                 // 1/15
    poly = 0x1.3b13b13b13b14p-4L  + f2 * poly;           // 1/13
    poly = 0x1.745d1745d1746p-4L  + f2 * poly;           // 1/11
    poly = 0x1.c71c71c71c71cp-4L  + f2 * poly;           // 1/9
    poly = 0x1.2492492492492p-3L  + f2 * poly;           // 1/7
    poly = 0x1.999999999999ap-3L  + f2 * poly;           // 1/5
    poly = 0x1.5555555555555p-2L  + f2 * poly;           // 1/3
    poly = 0x1.0000000000000p+0L  + f2 * poly;

    return double(e) * LN2 + 0x1.0000000000000p+1L * f * poly;
}

// ----- sin(x), cos(x) ---------------------------------------------------
//
// Two-stage Cody-Waite range reduction:
//   Stage 1: y = x mod 2π, computed as
//     (x - k1*TWO_PI_HI) - k1*TWO_PI_LO   where k1 = round(x / 2π)
//   Stage 2: z = y mod (π/2), into [-π/4, π/4]:
//     (y - j*PI_2_HI) - j*PI_2_LO         where j = round(y / (π/2))
// Each stage's high constant has 19 trailing zero mantissa bits,
// so for |k1| < 2^20 (i.e. |x| ≲ 2^21 ≈ 2.1M) both `k_d * HI`
// products are exact in f64. Result: ~1 ULP precision over that
// range. For |x| beyond 2^20, accuracy degrades — full Payne-Hanek
// reduction would extend it but is V.3.E.5 follow-up work.
//
// Polynomials: 7-term Taylor each for sin(z), cos(z) on [-π/4, π/4].
// Quadrant select:
//   j=0: +sin(z)    j=1: +cos(z)    j=2: -sin(z)    j=3: -cos(z)
double poly_sin_d(double x) {
    // Stage 1: reduce to [-π, π].
    double k1_d = floor(x / TWO_PI + 0.5L);
    double y = (x - k1_d * TWO_PI_HI) - k1_d * TWO_PI_LO;

    // Stage 2: reduce y to [-π/4, π/4] via quadrant.
    double j_d = floor(y / PI_2 + 0.5L);
    double z = (y - j_d * PI_2_HI) - j_d * PI_2_LO;
    int j = int(j_d) & 3;

    double z2 = z * z;

    // sin(z) = z * P(z²) where P(u) = sum_{k=0}^{8} (-1)^k u^k / (2k+1)!
    // 9-term Taylor through z^17. Truncation at |z|=π/4:
    //   (π/4)^19 / 19! ≈ 5e-19 — well below 1 ULP.
    double sin_p =  0x1.952c77030ad4ap-49L;               // +1/17!
    sin_p = -0x1.ae7f3e733b81fp-41L  + z2 * sin_p;        // -1/15!
    sin_p =  0x1.6124613a86d09p-33L  + z2 * sin_p;        // +1/13!
    sin_p = -0x1.ae64567f544e4p-26L  + z2 * sin_p;        // -1/11!
    sin_p =  0x1.71de3a556c734p-19L  + z2 * sin_p;        // +1/9!
    sin_p = -0x1.a01a01a01a01ap-13L  + z2 * sin_p;        // -1/7!
    sin_p =  0x1.1111111111111p-7L   + z2 * sin_p;        // +1/5!
    sin_p = -0x1.5555555555555p-3L   + z2 * sin_p;        // -1/3!
    sin_p =  0x1.0000000000000p+0L   + z2 * sin_p;        // +1
    double sz = z * sin_p;

    // cos(z) = Q(z²) where Q(u) = sum_{k=0}^{8} (-1)^k u^k / (2k)!
    // 9-term Taylor through z^16. Truncation at |z|=π/4:
    //   (π/4)^18 / 18! ≈ 6e-18 — well below 1 ULP.
    double cos_p =  0x1.ae7f3e733b81fp-45L;               // +1/16!
    cos_p = -0x1.93974a8c07c9dp-37L  + z2 * cos_p;        // -1/14!
    cos_p =  0x1.1eed8eff8d898p-29L  + z2 * cos_p;        // +1/12!
    cos_p = -0x1.27e4fb7789f5cp-22L  + z2 * cos_p;        // -1/10!
    cos_p =  0x1.a01a01a01a01ap-16L  + z2 * cos_p;        // +1/8!
    cos_p = -0x1.6c16c16c16c17p-10L  + z2 * cos_p;        // -1/6!
    cos_p =  0x1.5555555555555p-5L   + z2 * cos_p;        // +1/4!
    cos_p = -0x1.0000000000000p-1L   + z2 * cos_p;        // -1/2!
    cos_p =  0x1.0000000000000p+0L   + z2 * cos_p;        // +1
    double cz = cos_p;

    if (j == 0) return  sz;
    if (j == 1) return  cz;
    if (j == 2) return -sz;
                return -cz;
}

double poly_cos_d(double x) {
    return poly_sin_d(x + PI_2);
}

// ----- sigmoid / silu / gelu composites — used directly from main(). ---

// ----- tanh(x) ----------------------------------------------------------
//
// tanh(x) = (1 - e^(-2|x|)) / (1 + e^(-2|x|)), copy sign of x.
// For |x| >= 18, e^(-2|x|) < 1e-15 so the result saturates to ±1
// within f64 precision.
// tanh: piecewise to avoid catastrophic cancellation in the
// exp-based formula when x is near zero.
//   |x| >= 18 → saturate to ±1 (within f64 precision)
//   |x|  < 0.5 → Taylor (no cancellation; bounded coefficients)
//   else → (1 - e^(-2|x|)) / (1 + e^(-2|x|)), |t| sign-corrected
double poly_tanh_d(double x) {
    double ax = abs(x);
    if (ax >= 0x1.2p+4L) {  // 18.0
        return (x < 0.0L) ? -0x1.0p+0L : 0x1.0p+0L;
    }
    if (ax < 0x1.0p-2L) {   // 0.25 — below this, exp(-2x) is close to 1
                            //   and cancellation in (1 - e2x) wrecks
                            //   precision. Above 0.25 we have ~5 digits
                            //   of cancellation slack; Taylor with 8
                            //   terms gives ~1e-13 over [0, 0.25].
        double x2 = x * x;
        // Horner: tanh(x) = x * sum_{k=0}^{7} a_k x^(2k)
        double p = -0x1.7da36452b75e3p-10L;              // -929569/638512875
        p =  0x1.d6d3d0e157de0p-9L   + x2 * p;           // +21844/6081075
        p = -0x1.226e355e6c23dp-7L   + x2 * p;           // -1382/155925
        p =  0x1.664f4882c10fap-6L   + x2 * p;           // +62/2835
        p = -0x1.ba1ba1ba1ba1cp-5L   + x2 * p;           // -17/315
        p =  0x1.1111111111111p-3L   + x2 * p;           // +2/15
        p = -0x1.5555555555555p-2L   + x2 * p;           // -1/3
        p =  0x1.0p+0L               + x2 * p;           // +1
        return x * p;
    }
    // Mid-range: (1 - e^(-2|x|)) / (1 + e^(-2|x|))
    double e2x = poly_exp_d(-0x1.0p+1L * ax);
    double t = (0x1.0p+0L - e2x) / (0x1.0p+0L + e2x);
    return (x < 0.0L) ? -t : t;
}

// ----- Entry point ------------------------------------------------------

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.n) return;
    double x = input[idx];
    double y;
    uint op = p.op_id;

    if (op == 0u) {
        y = -x;                                                  // Neg
    } else if (op == 1u) {
        y = x * x;                                               // Sqr
    } else if (op == 2u) {
        y = sqrt(x);                                             // Sqrt
    } else if (op == 3u) {
        y = poly_exp_d(x);                                       // Exp
    } else if (op == 4u) {
        y = poly_log_d(x);                                       // Log
    } else if (op == 5u) {
        y = poly_sin_d(x);                                       // Sin
    } else if (op == 6u) {
        y = poly_cos_d(x);                                       // Cos
    } else if (op == 7u) {
        y = poly_tanh_d(x);                                      // Tanh
    } else if (op == 8u) {
        y = 1.0L / (1.0L + poly_exp_d(-x));                      // Sigmoid
    } else if (op == 9u) {
        y = x / (1.0L + poly_exp_d(-x));                         // Silu
    } else if (op == 10u) {
        // Gelu (tanh approx): 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))
        double inner = 0.7978845608028654L * (x + 0.044715L * x * x * x);
        y = 0.5L * x * (1.0L + poly_tanh_d(inner));
    } else if (op == 11u) {
        y = max(x, 0.0L);                                        // Relu
    } else {
        y = (x > 0.0L) ? 1.0L : 0.0L;                            // Step
    }
    output[idx] = y;
}
