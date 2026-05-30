#version 450
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Cooperative-matrix tiled matmul: C = A @ B  (pure-bf16 path).
// A: [M, K] bf16 (u16 storage)
// B: [K, N] bf16 (u16 storage)
// C: [M, N] bf16 (u32-packed storage; two bf16 lanes per word)
//
// Sibling of matmul_coop_bf16_bf16.glsl (bf16×bf16 → f32). Same
// coop[3] tile (A=f16, B=f16, C=f32, R=f32) with f32 accumulator;
// the only differences are (1) C buffer type and (2) a post-matmul
// downcast staging step:
//
//   1. coopMatStore the 16×16 subgroup accumulator to a shared f32
//      staging tile  ([16 × 64] = 4 KB).
//   2. Workgroup barrier.
//   3. 128 threads cooperatively read the staging tile, convert
//      each f32 → bf16 (round-to-nearest-even, NaN → 0x7FC0), and
//      pack pairs of adjacent column lanes into u32 writes to C.
//
// PRECONDITION: n % 16 == 0 (already required by the coop tile);
// because col_base is a multiple of WG_N = 64, every (col, col+1)
// pair the downcast step touches is u32-aligned in C.

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { uint16_t A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { uint16_t B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { uint C[]; };  // packed bf16

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint sa_batch;  uint sa_row;  uint sa_col;
    uint sb_batch;  uint sb_row;  uint sb_col;
    uint sc_batch;  // in bf16 ELEMENTS, not in u32 words
    uint n_rep;
    uint _pad;
} p;

const uint CM = 16u;
const uint CN = 16u;
const uint CK = 16u;
const uint N_SG = 4u;
const uint WG_N = CN * N_SG;     // 64

shared float16_t a_tile[CM * CK];
shared float16_t b_tile[CK * WG_N];
// Staging for the f32 accumulator before bf16 downcast.
shared float    out_tile[CM * WG_N];   // 16 × 64 = 1024 f32 = 4 KB

float16_t bf16_u16_to_f16(uint16_t bits) {
    uint bits32 = uint(bits) << 16;
    return float16_t(uintBitsToFloat(bits32));
}

// f32 → bf16 round-to-nearest-even; NaN → canonical 0x7FC0.
uint f32_to_bf16_bits(float x) {
    uint bits = floatBitsToUint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

void main() {
    uint batch = gl_WorkGroupID.z;
    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;   // bf16 element offset

    uint row_base = gl_WorkGroupID.y * CM;
    uint col_base = gl_WorkGroupID.x * WG_N;

    uint tid = gl_LocalInvocationID.x;
    uint sg_idx = tid / 32u;

    coopmat<float, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator>(0.0);

    uint k_tiles = (p.K + CK - 1u) / CK;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        uint k_base = kt * CK;

        for (uint i = 0u; i < 2u; ++i) {
            uint idx = tid + i * 128u;
            if (idx < CM * CK) {
                uint ar = idx / CK;
                uint ak = idx % CK;
                uint gr = row_base + ar;
                uint gk = k_base + ak;
                float16_t v = float16_t(0.0);
                if (gr < p.M && gk < p.K) {
                    v = bf16_u16_to_f16(A[a_off + gr * p.sa_row + gk * p.sa_col]);
                }
                a_tile[idx] = v;
            }
        }

        for (uint i = 0u; i < 8u; ++i) {
            uint idx = tid + i * 128u;
            if (idx < CK * WG_N) {
                uint bk = idx / WG_N;
                uint bn = idx % WG_N;
                uint gk = k_base + bk;
                uint gc = col_base + bn;
                float16_t v = float16_t(0.0);
                if (gk < p.K && gc < p.N) {
                    v = bf16_u16_to_f16(B[b_off + gk * p.sb_row + gc * p.sb_col]);
                }
                b_tile[bk * WG_N + bn] = v;
            }
        }

        barrier();

        coopmat<float16_t, gl_ScopeSubgroup, CM, CK, gl_MatrixUseA> matA;
        coopmat<float16_t, gl_ScopeSubgroup, CK, CN, gl_MatrixUseB> matB;

        coopMatLoad(matA, a_tile, 0, CK, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matB, b_tile, sg_idx * CN, WG_N, gl_CooperativeMatrixLayoutRowMajor);

        acc = coopMatMulAdd(matA, matB, acc);

        barrier();
    }

    // Stage each subgroup's 16×16 acc to shared f32, then barrier so
    // any thread can read any cell for the bf16 pack step.
    coopMatStore(acc, out_tile, sg_idx * CN, WG_N, gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    // 128 threads × 8 elements = 1024 = CM * WG_N. Each thread owns
    // 8 contiguous staging cells in one row (tid / 8), columns
    // [(tid % 8) * 8, (tid % 8) * 8 + 8). That's 4 (lo, hi) pairs,
    // each packed into one u32 word in C.
    uint local_row = tid / 8u;
    uint col_start = (tid & 7u) * 8u;
    uint gr = row_base + local_row;
    if (gr < p.M) {
        for (uint i = 0u; i < 4u; ++i) {
            uint lc = col_start + i * 2u;
            uint gc = col_base + lc;
            if (gc + 1u <= p.N) {
                float v0 = out_tile[local_row * WG_N + lc];
                float v1 = out_tile[local_row * WG_N + lc + 1u];
                uint bf0 = f32_to_bf16_bits(v0);
                uint bf1 = f32_to_bf16_bits(v1);
                uint packed = bf0 | (bf1 << 16u);
                uint word_idx = (c_off + gr * p.N + gc) >> 1u;
                C[word_idx] = packed;
            }
        }
    }
}
