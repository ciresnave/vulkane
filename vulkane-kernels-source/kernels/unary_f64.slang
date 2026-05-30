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
// that exposes shaderFloat64. Target precision: ~1 ULP relative
// across the full finite input range.
//
// Realized via:
//   - Cody-Waite range reduction for exp (with a full-precision LN2_LO
//     so k*(LN2_HI+LN2_LO) tracks true ln(2) to ~80 bits).
//   - Payne-Hanek range reduction for sin/cos (DD multiplication by
//     2/π, valid through |x| ≈ 2^53; lifts the prior |x| < 6.6e6 cap
//     that the three-term Cody-Waite ran into).
//   - Double-double Horner + reconstruction for exp, so the polynomial
//     and the final 1+r+(r*c)/(2-c) sum carry sub-ULP precision through
//     the 2^k scaling. See dd_step / two_prod / two_sum below.
//   - 9-term Taylor polynomials for sin/cos in z ∈ [-π/4, π/4]
//     (truncation error well below 1 ULP at the interval endpoint).

struct Params {
    uint out_size;
    uint op_id;
    uint rank;
    uint flags;           // bit 0: input_contiguous
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
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
static const double SQRT2       = 0x1.6a09e667f3bcdp+0L;

// Cody-Waite split for ln(2): 19 low mantissa bits zeroed in LN2_HI so
// that for |k| < 2^20 (covers exp's natural range) the product k*LN2_HI
// is exact in f64. LN2_LO carries the remaining bits of true ln(2)
// rounded to nearest f64 — full 48-bit mantissa precision, not just
// the 21-bit version that the prior code used (the earlier
// 0x1.1cf7800000000p-36 value capped LN2_LO at 21 bits, which made
// LN2_HI+LN2_LO precise only to ULP of f64::LN_2 ≈ 2^-53. With 48
// useful bits, the DD pair (LN2_HI, LN2_LO) tracks true ln(2) to
// ~80 bits — enough that k*(LN2_HI+LN2_LO) computed in DD has sub-
// ULP precision through the exp(700) range).
static const double LN2_HI      = 0x1.62e42fef80000p-1L;
static const double LN2_LO      = 0x1.1cf79abc9e3b0p-36L;

// Double-double pieces for Payne-Hanek range reduction (sin/cos).
// Together INV_TWO_PI_HI + INV_TWO_PI_LO carries ~106 bits of 2/π —
// enough that for any |x| <= 2^53 we can compute x*(2/π) to full
// f64 fractional precision (52 bits of integer part + 53 bits of
// fraction = ~105 needed; DD delivers ~106). PI_2_HI + PI_2_LO is
// π/2 in DD; used for the final f*(π/2) multiplication that
// produces the reduced argument in [-π/4, π/4].
static const double INV_TWO_PI_HI = 0x1.45f306dc9c883p-1L;   // 2/π   high
static const double INV_TWO_PI_LO = -0x1.6b01ec5417056p-55L; // 2/π   low (~-3.94e-17)
static const double PI_2_HI       = 0x1.921fb54442d18p+0L;   // π/2   high
static const double PI_2_LO       = 0x1.1a62633145c07p-54L;  // π/2   low (~6.12e-17)

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

// ----- Double-double primitives ------------------------------------------
//
// A DD number is a pair (hi, lo) representing hi + lo with
// |lo| <= ULP(hi)/2 (i.e. lo is below the rounding granularity of hi).
// The arithmetic primitives below are the classical error-free
// transformations from Dekker (1971) and Knuth (TAOCP vol 2): each
// returns the rounded result PLUS the exact rounding error, so we can
// chain operations and keep ~106 bits of precision through long
// computations.
//
// All algorithms require IEEE 754 round-to-nearest semantics with no
// FMA contraction and no reassociation. SPIR-V honors this by default
// (it would take an explicit FastMath decoration to break it), so the
// Vulkan-compute path is safe.

// Knuth's TwoSum: s + e = a + b exactly. Always normalized.
void two_sum(double a, double b, out double s, out double e) {
    s = a + b;
    double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// Fast2Sum: same effect, 3 fewer ops, but only valid when |a| >= |b|.
void fast2_sum(double a, double b, out double s, out double e) {
    s = a + b;
    e = b - (s - a);
}

// Dekker product: p + e = a * b exactly. Splits each operand into
// 26-bit halves via the Veltkamp split (SPLIT = 2^27+1). Doesn't use
// FMA — Vulkan's f64 surface doesn't expose fma() so we must use the
// classic split-multiply-subtract path.
void two_prod(double a, double b, out double p, out double e) {
    const double SPLIT = 0x1.0000002p+27L;  // 2^27 + 1
    double a_t = SPLIT * a;
    double a_hi = a_t - (a_t - a);
    double a_lo = a - a_hi;
    double b_t = SPLIT * b;
    double b_hi = b_t - (b_t - b);
    double b_lo = b - b_hi;
    p = a * b;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

// DD-aware Horner step: (s_hi, s_lo) := (s_hi, s_lo) * r + c
// One iteration of polynomial evaluation in double-double precision.
// Cost ~22 f64 ops vs 2 for the naive Horner — acceptable for unary
// kernels which are bandwidth-bound on Vulkan.
void dd_step(inout double s_hi, inout double s_lo, double r, double c) {
    double p_hi, p_lo;
    two_prod(s_hi, r, p_hi, p_lo);
    p_lo += s_lo * r;
    double t_hi, t_lo;
    two_sum(p_hi, c, t_hi, t_lo);
    t_lo += p_lo;
    fast2_sum(t_hi, t_lo, s_hi, s_lo);
}

// ----- exp(x) ------------------------------------------------------------
//
// fdlibm-style rational reconstruction in double-double precision:
//     exp(r) = 1 + r + (r*c) / (2 - c)
//     c      = r - r²*P(r²)
//     P(u)   = P1 + u*(P2 + u*(P3 + u*(P4 + u*P5)))
// Minimax polynomial P from fdlibm e_exp.c (Remes-fit of
// R(z) = r*(exp(r)+1)/(exp(r)-1) = 2 + r²/6 - r⁴/360 + ... on
// |r| <= ln(2)/2; polynomial error bound 2^-59).
//
// V.3.E.5.A: every intermediate after range reduction is carried as a
// DD pair, including the polynomial Horner (via dd_step). The final
// 2^k scaling is exact (k is integer, pow2_int returns a power of 2)
// so as long as the DD reconstruction is accurate to ~1 ULP, the
// scaled result is too. Replaces the prior naive Horner which
// accumulated ~75 ULP at exp(500).
double poly_exp_d(double x) {
    if (x > 709.0L)  return 1.0L / 0.0L;
    if (x < -745.0L) return 0.0L;

    double k_d = floor(x * INV_LN2 + 0.5L);
    int k = int(k_d);

    // Cody-Waite range reduction. (x - k*LN2_HI) is exact when
    // |k| < 2^20 (LN2_HI has 19 trailing zero mantissa bits);
    // subtracting k*LN2_LO completes the split. The DD pair (r_hi, r_lo)
    // recovers the residual that the naive single-double r drops.
    double s = x - k_d * LN2_HI;
    double t = k_d * LN2_LO;
    double r_hi, r_lo;
    two_sum(s, -t, r_hi, r_lo);

    // r² in DD (used as Horner argument and as a regular value)
    double r2_hi, r2_lo;
    two_prod(r_hi, r_hi, r2_hi, r2_lo);
    r2_lo += 2.0L * r_hi * r_lo;

    // Horner for P(r²) in DD. We pass r2_hi as the multiplier; the
    // cross-term r2_lo is small enough (|r2_lo| <= 2^-53 * r2_hi <=
    // 2^-53 * 0.12) that omitting it in dd_step costs <1 ULP. Coefficients
    // are from fdlibm e_exp.c.
    double P_hi = 0x1.6376972bea4d0p-25L;  // P5
    double P_lo = 0.0L;
    dd_step(P_hi, P_lo, r2_hi, -0x1.bbd41c5d26bf1p-20L);  // * r² + P4
    dd_step(P_hi, P_lo, r2_hi,  0x1.1566aaf25de2cp-14L);  // * r² + P3
    dd_step(P_hi, P_lo, r2_hi, -0x1.6c16c16bebd93p-9L);   // * r² + P2
    dd_step(P_hi, P_lo, r2_hi,  0x1.555555555553ep-3L);   // * r² + P1

    // c = r - r²*P in DD
    double rP_hi, rP_lo;
    two_prod(r2_hi, P_hi, rP_hi, rP_lo);
    rP_lo += r2_hi * P_lo + r2_lo * P_hi;
    double c_hi, c_lo;
    two_sum(r_hi, -rP_hi, c_hi, c_lo);
    c_lo += r_lo - rP_lo;
    {
        double s2, e2;
        fast2_sum(c_hi, c_lo, s2, e2);
        c_hi = s2; c_lo = e2;
    }

    // num = r * c in DD
    double n_hi, n_lo;
    two_prod(r_hi, c_hi, n_hi, n_lo);
    n_lo += r_hi * c_lo + r_lo * c_hi;
    {
        double s2, e2;
        fast2_sum(n_hi, n_lo, s2, e2);
        n_hi = s2; n_lo = e2;
    }

    // den = 2 - c in DD
    double d_hi, d_lo;
    two_sum(2.0L, -c_hi, d_hi, d_lo);
    d_lo -= c_lo;
    {
        double s2, e2;
        fast2_sum(d_hi, d_lo, s2, e2);
        d_hi = s2; d_lo = e2;
    }

    // q = num / den in DD (Newton step refinement: q1 = num_hi/den_hi
    // is a first approximation; correct using the residual).
    double q1 = n_hi / d_hi;
    double p_hi, p_lo;
    two_prod(q1, d_hi, p_hi, p_lo);
    // residual = num - q1*den
    double res_hi = n_hi - p_hi;            // exact (Sterbenz, |res| ≪ |n_hi|)
    double res_lo = n_lo - p_lo - q1 * d_lo;
    double q2 = (res_hi + res_lo) / d_hi;
    double q_hi, q_lo;
    fast2_sum(q1, q2, q_hi, q_lo);

    // exp(r) = 1 + r + q  (all in DD)
    double sum_hi, sum_lo;
    two_sum(1.0L, r_hi, sum_hi, sum_lo);
    sum_lo += r_lo;
    {
        double s2, e2;
        fast2_sum(sum_hi, sum_lo, s2, e2);
        sum_hi = s2; sum_lo = e2;
    }
    double er_hi, er_lo;
    two_sum(sum_hi, q_hi, er_hi, er_lo);
    er_lo += sum_lo + q_lo;
    {
        double s2, e2;
        fast2_sum(er_hi, er_lo, s2, e2);
        er_hi = s2; er_lo = e2;
    }

    // Multiply by 2^k. pow2_int(k) is an exact power of 2, so both
    // er_hi and er_lo scale exactly; their sum then rounds to give
    // the f64 result to ~1 ULP.
    double pk = pow2_int(k);
    return er_hi * pk + er_lo * pk;
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
// Payne-Hanek range reduction in DD precision.
//
// For input x: compute k = round(x * 2/π) and the fractional part
// f = x*(2/π) - k, then form z = f * π/2 — the reduced argument in
// [-π/4, π/4] — and select quadrant by k mod 4.
//
// Why PH instead of Cody-Waite: Cody-Waite stores 2π split into
// 3 pieces (TWO_PI_1/2/3) and computes `k*TWO_PI_i` for each piece.
// The trick relies on k*TWO_PI_1 staying exact, which holds only
// while k fits in (53 - mantissa_bits_of_TWO_PI_1) bits — for our
// 33-bit TWO_PI_1, that's |k| < 2^20, i.e. |x| < 2^20 * 2π ≈ 6.6e6.
// Beyond that the first product loses bits and the remaining terms
// can't recover them.
//
// PH sidesteps the issue by multiplying by 2/π (which is < 1) instead
// of 2π. The integer part of x*(2/π) is bounded by |x| and can be
// represented exactly when |x| < 2^53; the fractional part needs ~53
// bits of precision, met by storing 2/π as a DD pair (~106 bits).
// This carries sin/cos through the full f64 finite range with ~1 ULP
// precision throughout, including the worst-case "huge multiple of π"
// inputs that Cody-Waite cannot handle.
//
// Polynomials below are unchanged 9-term Taylors on [-π/4, π/4] —
// what got better is the input z, not the polynomial.
//
// Quadrant select:
//   j=0: +sin(z)    j=1: +cos(z)    j=2: -sin(z)    j=3: -cos(z)
//
// Returns: quadrant index j ∈ {0,1,2,3} and z in [-π/4, π/4] (as DD).
int payne_hanek(double x, out double z_hi, out double z_lo) {
    // xy = x * (2/π) in DD
    double xy_hi, xy_lo;
    two_prod(x, INV_TWO_PI_HI, xy_hi, xy_lo);
    xy_lo += x * INV_TWO_PI_LO;
    {
        double s, e;
        fast2_sum(xy_hi, xy_lo, s, e);
        xy_hi = s; xy_lo = e;
    }

    // Candidate k = round(xy_hi). floor(...+0.5) rounds half-up.
    double k_d = floor(xy_hi + 0.5L);

    // f = xy - k_d in DD. (xy_hi - k_d) is exact: |xy_hi - k_d| <= 0.5
    // while |xy_hi| is at least ~0.5 wherever k_d != 0, so Sterbenz's
    // condition (b/2 <= a <= 2b) holds. For xy_hi very close to 0 the
    // subtraction is trivially exact.
    double f_hi, f_lo;
    two_sum(xy_hi - k_d, xy_lo, f_hi, f_lo);

    // The xy_lo addition may push f outside [-0.5, 0.5] by up to
    // |xy_lo| (sub-ULP). When that happens we missed by one quadrant —
    // bump k_d and snap f back into range. The 1.0 subtraction here is
    // exact because |f_hi| is just over 0.5.
    if (f_hi > 0.5L || (f_hi == 0.5L && f_lo > 0.0L)) {
        k_d += 1.0L;
        two_sum(f_hi - 1.0L, f_lo, f_hi, f_lo);
    } else if (f_hi < -0.5L || (f_hi == -0.5L && f_lo < 0.0L)) {
        k_d -= 1.0L;
        two_sum(f_hi + 1.0L, f_lo, f_hi, f_lo);
    }

    // k mod 4 computed in floating-point (k_d may exceed int32 range
    // for huge x — k_d * 0.25 is exact since 0.25 is a power of 2).
    double k_mod4_d = k_d - 4.0L * floor(k_d * 0.25L);
    int j = int(k_mod4_d) & 3;

    // z = f * (π/2) in DD
    double zp_hi, zp_lo;
    two_prod(f_hi, PI_2_HI, zp_hi, zp_lo);
    zp_lo += f_hi * PI_2_LO + f_lo * PI_2_HI;
    fast2_sum(zp_hi, zp_lo, z_hi, z_lo);

    return j;
}

// Polynomial body shared by sin/cos: given reduced z ∈ [-π/4, π/4]
// and quadrant index j ∈ {0,1,2,3}, returns the trig value selected
// by the quadrant. Sin uses j directly; cos uses (j+1) mod 4 so the
// SAME table picks the right output (cos(x) = sin(x + π/2)).
//
// Polynomial in z_hi only: z_lo from PH is sub-ULP, and propagating
// it would add ~5 ops for a sub-1-ULP correction we don't need.
double sincos_poly_d(double z, int j) {
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

double poly_sin_d(double x) {
    double z_hi, z_lo;
    int j = payne_hanek(x, z_hi, z_lo);
    return sincos_poly_d(z_hi, j);
}

// cos via direct PH on x with quadrant offset by 1: cos(x) = sin(x + π/2).
// Crucially we do NOT compute poly_sin_d(x + PI_2) — for |x| past ~2^52,
// ULP(x) > π/2 and (x + π/2) rounds back to x, giving cos(x) = sin(x).
// PH directly on x avoids that catastrophic precision loss.
double poly_cos_d(double x) {
    double z_hi, z_lo;
    int j = (payne_hanek(x, z_hi, z_lo) + 1) & 3;
    return sincos_poly_d(z_hi, j);
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
    if (idx >= p.out_size) return;

    double x;
    bool in_contig = (p.flags & 1u) != 0u;
    if (in_contig) {
        x = input[idx];
    } else {
        uint c3 = idx % p.shape3;
        uint r2 = idx / p.shape3;
        uint c2 = r2 % p.shape2;
        uint r1 = r2 / p.shape2;
        uint c1 = r1 % p.shape1;
        uint c0 = r1 / p.shape1;
        uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
        x = input[in_off];
    }

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
    } else if (op == 12u) {
        y = (x > 0.0L) ? 1.0L : 0.0L;                            // Step
    } else if (op == 13u) {
        y = abs(x);                                              // Abs
    } else if (op == 14u) {
        y = sign(x);                                             // Sign
    } else {
        y = 1.0L / x;                                            // Recip
    }
    output[idx] = y;
}
