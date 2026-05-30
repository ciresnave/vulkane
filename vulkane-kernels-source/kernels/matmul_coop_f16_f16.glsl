#version 450
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Cooperative-matrix tiled matmul: C = A @ B
// A: [M, K] f16 (float16_t)
// B: [K, N] f16 (float16_t)
// C: [M, N] f32
//
// Native f16 inputs — no downcast needed (unlike the bf16 sibling).
// Uses coop[3] tile (A=f16, B=f16, C=f32, R=f32) with f32 accumulator.
//
// Tile: 16x64 per workgroup (4 subgroups × 16-col tiles).
// K-loop: chunks of 16. Dispatch: (ceil(N/64), ceil(M/16), batch).

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float16_t A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { float16_t B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint sa_batch;  uint sa_row;  uint sa_col;
    uint sb_batch;  uint sb_row;  uint sb_col;
    uint sc_batch;
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

        // Cooperative load of a_tile[CM*CK] — 256 elements, 2 per thread.
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

        // Cooperative load of b_tile[CK*WG_N] — 1024 elements, 8 per thread.
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

    uint out_col = col_base + sg_idx * CN;
    if (row_base < p.M && out_col < p.N) {
        coopMatStore(acc, C, c_off + row_base * p.N + out_col, p.N,
                     gl_CooperativeMatrixLayoutRowMajor);
    }
}
