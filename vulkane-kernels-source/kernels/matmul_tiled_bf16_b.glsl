#version 450
// Tiled matrix multiply with bf16 weights: C = A @ B
// A: [M, K]   row-major f32
// B: [K, N]   row-major bf16 (packed 2-per-u32 in storage)
// C: [M, N]   row-major f32
//
// Same 64×64 workgroup-tile + 4×4 register-tile strategy as
// matmul_tiled.glsl; the only difference is that global loads of
// B go through a bf16-unpack step before landing in shared memory.
// Shared tiles, inner k-loop, and accumulation all stay f32 — bf16
// only buys us the 2× memory saving on the weight matrix, not a
// narrower accumulator.
//
// The backend routes (A:f32, B:bf16) matmuls with M > 1 here.
// For M == 1 see matvec_bf16_b.glsl (subgroup-reduced gemv).

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { uint  B[]; };
layout(set = 0, binding = 2, std430) buffer          CBuf { float C[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint M;
    uint N;
    uint K;
    uint batch_stride_a;
    uint batch_stride_b;  // in bf16 elements
    uint batch_stride_c;
} params;

const uint TILE = 4u;
const uint BM = 64u;
const uint BN = 64u;
const uint BK = 16u;

shared float a_tile[BM][BK];
shared float b_tile[BK][BN];

float load_bf16(uint base_u32, uint i) {
    uint word = B[base_u32 + (i >> 1u)];
    uint bits = ((i & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
    return uintBitsToFloat(bits << 16);
}

void main() {
    uint batch = gl_WorkGroupID.z;
    uint a_off = batch * params.batch_stride_a;
    // batch_stride_b is in bf16 elements; the u32 base is half that.
    uint b_off_u32 = (batch * params.batch_stride_b) >> 1u;
    uint c_off = batch * params.batch_stride_c;

    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;
    uint lid = ly * 16u + lx;

    uint row_base = gl_WorkGroupID.y * BM;
    uint col_base = gl_WorkGroupID.x * BN;

    float acc[TILE][TILE];
    for (uint i = 0u; i < TILE; ++i) {
        for (uint j = 0u; j < TILE; ++j) {
            acc[i][j] = 0.0;
        }
    }

    uint k_tiles = (params.K + BK - 1u) / BK;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        uint k_base = kt * BK;

        // Cooperative load of a_tile[BM][BK] — identical to f32 path.
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = lid + i * 256u;
            uint ar = idx / BK;
            uint ak = idx - ar * BK;
            uint gr = row_base + ar;
            uint gk = k_base + ak;
            float v = 0.0;
            if (gr < params.M && gk < params.K) {
                v = A[a_off + gr * params.K + gk];
            }
            a_tile[ar][ak] = v;
        }

        // Cooperative load of b_tile[BK][BN] with bf16 unpack.
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = lid + i * 256u;
            uint bk_i = idx / BN;
            uint bn_j = idx - bk_i * BN;
            uint gk = k_base + bk_i;
            uint gc = col_base + bn_j;
            float v = 0.0;
            if (gk < params.K && gc < params.N) {
                v = load_bf16(b_off_u32, gk * params.N + gc);
            }
            b_tile[bk_i][bn_j] = v;
        }

        barrier();

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
