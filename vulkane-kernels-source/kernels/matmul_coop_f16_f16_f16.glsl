#version 450
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Cooperative-matrix tiled matmul: C = A @ B  (pure-f16 path).
// A: [M, K] f16 (float16_t)
// B: [K, N] f16 (float16_t)
// C: [M, N] f16 (u32-packed storage; two f16 lanes per word)
//
// Sibling of matmul_coop_bf16_bf16_bf16.glsl. Native float16_t inputs
// (no downcast on load); f32 accumulator → shared-mem staging →
// 128-thread per-lane conversion and packed-u32 write to C.

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float16_t A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { float16_t B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { uint C[]; };  // packed f16

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint sa_batch;  uint sa_row;  uint sa_col;
    uint sb_batch;  uint sb_row;  uint sb_col;
    uint sc_batch;  // in f16 ELEMENTS
    uint n_rep;
    uint _pad;
} p;

const uint CM = 16u;
const uint CN = 16u;
const uint CK = 16u;
const uint N_SG = 4u;
const uint WG_N = CN * N_SG;

shared float16_t a_tile[CM * CK];
shared float16_t b_tile[CK * WG_N];
shared float    out_tile[CM * WG_N];

// Pack two float16_t into a u32 word. f32→f16 uses Vulkan's native
// float16_t cast (round-to-nearest-even by default on NVIDIA);
// then we extract the underlying 16 bits via uint16BitsToHalf's
// inverse — float16BitsToUint16.
uint pack_two_f16_to_u32(float v0, float v1) {
    uint16_t h0 = float16BitsToUint16(float16_t(v0));
    uint16_t h1 = float16BitsToUint16(float16_t(v1));
    return uint(h0) | (uint(h1) << 16u);
}

void main() {
    uint batch = gl_WorkGroupID.z;
    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;

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
                    v = A[a_off + gr * p.sa_row + gk * p.sa_col];
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
                    v = B[b_off + gk * p.sb_row + gc * p.sb_col];
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

    coopMatStore(acc, out_tile, sg_idx * CN, WG_N, gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    // 128 threads × 8 cells = 1024 = CM × WG_N. Thread layout matches
    // the bf16→bf16 variant.
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
                uint packed = pack_two_f16_to_u32(v0, v1);
                uint word_idx = (c_off + gr * p.N + gc) >> 1u;
                C[word_idx] = packed;
            }
        }
    }
}
