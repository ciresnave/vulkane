#version 450
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Cooperative-matrix tiled matmul: C = A @ B
// A: [M, K] f32 activations
// B: [K, N] bf16 weights (packed as u16 in storage)
// C: [M, N] f32 output
//
// Uses VK_KHR_cooperative_matrix with f16 inputs + f32 accumulation
// (coop shape [3]: M=16 N=16 K=16, A=f16, B=f16, C=f32, R=f32).
// f32 activations are downcast to f16 on shared-mem load; bf16
// weights are converted to f16 on shared-mem load (lossless in
// mantissa for typical weight magnitudes). The coop-matmul
// accumulator stays f32, so the only precision loss is in the
// inputs, not the accumulation.
//
// Workgroup: 128 threads = 4 subgroups of 32 lanes.
// Each subgroup handles one 16x16 output tile.
// Workgroup output: 16 rows × 64 cols (4 subgroups across N).
// K-loop: chunks of 16.
//
// Dispatch: (ceil(N/64), ceil(M/16), batch)

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { uint16_t B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_c;
    uint n_rep;
    uint _pad;
} params;

const uint CM = 16u;    // cooperative matrix M
const uint CN = 16u;    // cooperative matrix N
const uint CK = 16u;    // cooperative matrix K
const uint N_SG = 4u;   // subgroups per workgroup
const uint WG_N = CN * N_SG;  // 64 cols per workgroup

// Shared tiles for one K-chunk: A as f16, B as f16.
// 1D for coopMatLoad compatibility (it takes T[] not T[][]).
shared float16_t a_tile[CM * CK];     // 16 × 16 = 256 elems = 512 B
shared float16_t b_tile[CK * WG_N];   // 16 × 64 = 1024 elems = 2 KB

void main() {
    uint batch = gl_WorkGroupID.z;
    uint a_off = batch * params.batch_stride_a;
    uint b_off = (batch / params.n_rep) * params.batch_stride_b;
    uint c_off = batch * params.batch_stride_c;

    uint row_base = gl_WorkGroupID.y * CM;
    uint col_base = gl_WorkGroupID.x * WG_N;

    uint tid = gl_LocalInvocationID.x;
    uint sg_idx = tid / 32u;  // which subgroup (0..3)

    // Initialize accumulator (f32, 16x16 per subgroup).
    coopmat<float, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator>(0.0);

    uint k_tiles = (params.K + CK - 1u) / CK;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        uint k_base = kt * CK;

        // Cooperative load of a_tile[CM*CK] — 256 f16 elements.
        // 128 threads → 2 elements each. Convert f32 → f16 on the fly.
        for (uint i = 0u; i < 2u; ++i) {
            uint idx = tid + i * 128u;
            if (idx < CM * CK) {
                uint ar = idx / CK;
                uint ak = idx % CK;
                uint gr = row_base + ar;
                uint gk = k_base + ak;
                float v = 0.0;
                if (gr < params.M && gk < params.K) {
                    v = A[a_off + gr * params.K + gk];
                }
                a_tile[idx] = float16_t(v);
            }
        }

        // Cooperative load of b_tile[CK][WG_N] — 16*64=1024 f16 elements.
        // 128 threads → 8 elements each. Convert bf16 → f16.
        for (uint i = 0u; i < 8u; ++i) {
            uint idx = tid + i * 128u;
            if (idx < CK * WG_N) {
                uint bk = idx / WG_N;
                uint bn = idx % WG_N;
                uint gk = k_base + bk;
                uint gc = col_base + bn;
                float16_t v = float16_t(0.0);
                if (gk < params.K && gc < params.N) {
                    // bf16 → f32 → f16. bf16 bits in u16; extend to f32
                    // via left-shift, then narrow to f16 for the coop
                    // matrix input.
                    uint bits32 = uint(B[b_off + gk * params.N + gc]) << 16;
                    v = float16_t(uintBitsToFloat(bits32));
                }
                b_tile[bk * WG_N + bn] = v;
            }
        }

        barrier();

        // Each subgroup loads its 16x16 slice of the shared tiles
        // and issues one cooperative MulAdd.
        coopmat<float16_t, gl_ScopeSubgroup, CM, CK, gl_MatrixUseA> matA;
        coopmat<float16_t, gl_ScopeSubgroup, CK, CN, gl_MatrixUseB> matB;

        // Load A tile — same for all subgroups (same 16 rows).
        // a_tile is row-major [CM, CK], stride = CK.
        coopMatLoad(matA, a_tile, 0, CK, gl_CooperativeMatrixLayoutRowMajor);

        // Load B tile — each subgroup reads its 16-col slice from
        // b_tile which is row-major [CK, WG_N], stride = WG_N.
        // Subgroup sg_idx's 16-col slice starts at column sg_idx*CN.
        coopMatLoad(matB, b_tile, sg_idx * CN, WG_N, gl_CooperativeMatrixLayoutRowMajor);

        acc = coopMatMulAdd(matA, matB, acc);

        barrier();
    }

    // Store accumulated 16x16 f32 result per subgroup.
    // Each subgroup writes to its column slice of the output.
    uint out_col = col_base + sg_idx * CN;
    if (row_base < params.M && out_col < params.N) {
        coopMatStore(acc, C, c_off + row_base * params.N + out_col, params.N,
                     gl_CooperativeMatrixLayoutRowMajor);
    }
}
