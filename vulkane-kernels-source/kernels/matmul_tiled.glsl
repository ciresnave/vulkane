#version 450
// Tiled matrix multiply: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]  (all row-major f32)
//
// 16x16 workgroup, each thread computes a 4x4 register tile.
// Output tile per workgroup: 64x64. K-chunk: BK=16.
// Shared-memory cache: a_tile[64][16] + b_tile[16][64] = 8 KB.
//
// GLSL source -> SPIR-V via shaderc (glslang). Written for
// comparison/replacement of matmul.wgsl — naga's WGSL -> SPIR-V
// does not cache in groupshared, so register tiling alone. Shared
// memory here cuts global loads from O(N*M*K) to O(N*M*K/BK).

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { float B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_c;
} params;

const uint TILE = 4u;   // per-thread output tile edge
const uint BM = 64u;    // workgroup M-tile = 16 * TILE
const uint BN = 64u;    // workgroup N-tile = 16 * TILE
const uint BK = 16u;    // inner k-chunk size

shared float a_tile[BM][BK]; // 64 * 16 * 4 = 4 KB
shared float b_tile[BK][BN]; // 16 * 64 * 4 = 4 KB

void main() {
    uint batch = gl_WorkGroupID.z;
    uint a_off = batch * params.batch_stride_a;
    uint b_off = batch * params.batch_stride_b;
    uint c_off = batch * params.batch_stride_c;

    uint lx = gl_LocalInvocationID.x; // 0..16 -> col direction
    uint ly = gl_LocalInvocationID.y; // 0..16 -> row direction
    uint lid = ly * 16u + lx;         // 0..256

    uint row_base = gl_WorkGroupID.y * BM; // first row of output tile
    uint col_base = gl_WorkGroupID.x * BN; // first col of output tile

    float acc[TILE][TILE];
    for (uint i = 0u; i < TILE; ++i) {
        for (uint j = 0u; j < TILE; ++j) {
            acc[i][j] = 0.0;
        }
    }

    uint k_tiles = (params.K + BK - 1u) / BK;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        uint k_base = kt * BK;

        // Cooperative load of a_tile[BM][BK]: 64*16 = 1024 floats.
        // 256 threads => 4 floats per thread.
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = lid + i * 256u;         // 0..1023
            uint ar = idx / BK;                // 0..63
            uint ak = idx - ar * BK;           // 0..15
            uint gr = row_base + ar;
            uint gk = k_base + ak;
            float v = 0.0;
            if (gr < params.M && gk < params.K) {
                v = A[a_off + gr * params.K + gk];
            }
            a_tile[ar][ak] = v;
        }

        // Cooperative load of b_tile[BK][BN]: 16*64 = 1024 floats.
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = lid + i * 256u;         // 0..1023
            uint bk_i = idx / BN;              // 0..15
            uint bn_j = idx - bk_i * BN;       // 0..63
            uint gk = k_base + bk_i;
            uint gc = col_base + bn_j;
            float v = 0.0;
            if (gk < params.K && gc < params.N) {
                v = B[b_off + gk * params.N + gc];
            }
            b_tile[bk_i][bn_j] = v;
        }

        barrier();

        // Inner k loop — each thread owns a 4x4 output tile starting
        // at (ly*TILE, lx*TILE) within the workgroup tile.
        uint ar_base = ly * TILE;
        uint bc_base = lx * TILE;
        for (uint kk = 0u; kk < BK; ++kk) {
            float a_col[TILE];
            float b_row[TILE];
            for (uint i = 0u; i < TILE; ++i) {
                a_col[i] = a_tile[ar_base + i][kk];
            }
            for (uint j = 0u; j < TILE; ++j) {
                b_row[j] = b_tile[kk][bc_base + j];
            }
            for (uint i = 0u; i < TILE; ++i) {
                for (uint j = 0u; j < TILE; ++j) {
                    acc[i][j] += a_col[i] * b_row[j];
                }
            }
        }

        barrier();
    }

    // Write results.
    uint ar_base = ly * TILE;
    uint bc_base = lx * TILE;
    for (uint i = 0u; i < TILE; ++i) {
        uint r = row_base + ar_base + i;
        if (r >= params.M) continue;
        for (uint j = 0u; j < TILE; ++j) {
            uint c = col_base + bc_base + j;
            if (c < params.N) {
                C[c_off + r * params.N + c] = acc[i][j];
            }
        }
    }
}
