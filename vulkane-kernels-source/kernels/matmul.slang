// Tiled matrix multiply: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]  (all row-major f32)
//
// Uses 16x16 workgroups with each thread computing a 4x4 tile of C.
// Effective tile: 64x64 output per workgroup.
// No shared memory — relies on register tiling.
// Based on webgpu-blas (MIT) optimization pattern.
//
// Replaces matmul.wgsl. Written in Slang per the shader policy.

struct Params {
    uint M, N, K;
    uint sa_batch, sa_row, sa_col;
    uint sb_batch, sb_row, sb_col;
    uint sc_batch;
    uint n_rep, _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   A;
[[vk::binding(1, 0)]] StructuredBuffer<float>   B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

static const uint TILE = 4;  // each thread computes TILE x TILE output elements

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 gid : SV_DispatchThreadID,
    uint3 wid : SV_GroupID)
{
    uint batch = wid.z;
    uint row_base = gid.y * TILE;
    uint col_base = gid.x * TILE;

    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;

    // Accumulator: TILE x TILE register tile.
    float acc[4][4];
    for (uint i = 0u; i < TILE; i++) {
        for (uint j = 0u; j < TILE; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Walk the K dimension.
    for (uint k = 0u; k < p.K; k++) {
        // Load TILE elements from A column and B row.
        float a_col[4];
        float b_row[4];
        for (uint i = 0u; i < TILE; i++) {
            uint r = row_base + i;
            if (r < p.M) {
                a_col[i] = A[a_off + r * p.sa_row + k * p.sa_col];
            } else {
                a_col[i] = 0.0f;
            }
        }
        for (uint j = 0u; j < TILE; j++) {
            uint c = col_base + j;
            if (c < p.N) {
                b_row[j] = B[b_off + k * p.sb_row + c * p.sb_col];
            } else {
                b_row[j] = 0.0f;
            }
        }
        // Outer product accumulate.
        for (uint i = 0u; i < TILE; i++) {
            for (uint j = 0u; j < TILE; j++) {
                acc[i][j] += a_col[i] * b_row[j];
            }
        }
    }

    // Write results.
    for (uint i = 0u; i < TILE; i++) {
        for (uint j = 0u; j < TILE; j++) {
            uint r = row_base + i;
            uint c = col_base + j;
            if (r < p.M && c < p.N) {
                C[c_off + r * p.N + c] = acc[i][j];
            }
        }
    }
}
