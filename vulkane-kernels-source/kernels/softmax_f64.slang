// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused softmax along the last dimension — f64.
//
// Formula:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// Native f64 end-to-end (matches baracuda's CUDA pattern). Requires
// shaderFloat64 + GroupNonUniformArithmetic. exp is the pure-arithmetic
// poly_exp_d copied from unary_f64.slang (GLSL.std.450 Exp is 16/32-bit only and
// OpenCL.std is rejected by NVIDIA Windows Vulkan); Max stays GLSL.std.450.
//
// Same Params shape (8 bytes) and 3-phase structure as the f32 kernel.

struct Params {
    uint n_rows;
    uint n_cols;
};

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
    //
    // pow2_int is only valid for k in [-1022, 1023]. x in [-745, -708.4] gives
    // k down to -1075, where uint(k + 1023) wraps and the "power of 2" is
    // garbage (negative or huge), which poisoned a whole softmax row sum
    // (#106). Those results are subnormal, so scale in two exact-power-of-2
    // steps instead: the first lands in the normal range, the second rounds
    // once into the subnormal range.
    if (k < -1021) {
        double p1 = pow2_int(k + 100);
        double p2 = pow2_int(-100);
        return (er_hi * p1 + er_lo * p1) * p2;
    }
    double pk = pow2_int(k);
    return er_hi * pk + er_lo * pk;
}

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     params;

// 256 / subgroup size entries; 64 covers subgroup sizes down to 4.
groupshared double wg_max[64];
groupshared double wg_sum[64];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1 — per-row max in f64. -DBL_MAX as the per-lane sentinel.
    double local_max = -1.7976931348623157e+308;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = input[row_off + col];
        local_max = max(local_max, v);
    }

    double sg_max = WaveActiveMax(local_max);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_max[sg_id] = sg_max;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    // Combine the per-subgroup partials serially on one thread. Not
    // WaveActiveMax over subgroup 0: that only sees the lanes of ONE subgroup,
    // so it needs n_sg <= lane count, i.e. a subgroup size of at least 16.
    // Lavapipe (size 8) has 32 partials and silently dropped 24 of them (#106).
    if (tid == 0u) {
        double m = wg_max[0];
        for (uint i = 1u; i < n_sg; i++) {
            m = max(m, wg_max[i]);
        }
        wg_max[0] = m;
    }
    GroupMemoryBarrierWithGroupSync();
    double row_max = wg_max[0];

    // Phase 2 — exp + write + accumulate sum.
    double local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = poly_exp_d(input[row_off + col] - row_max);
        output[row_off + col] = v;
        local_sum += v;
    }

    double sg_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wg_sum[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        double s = 0.0;
        for (uint i = 0u; i < n_sg; i++) {
            s += wg_sum[i];
        }
        wg_sum[0] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    double row_sum = wg_sum[0];

    // Phase 3 — normalize.
    double inv_sum = 1.0 / row_sum;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_off + col] *= inv_sum;
    }
}
