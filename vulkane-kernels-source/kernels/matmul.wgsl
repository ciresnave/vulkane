// Tiled matrix multiply: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]  (all row-major f32)
//
// Uses 16x16 workgroups with each thread computing a 4x4 tile of C.
// Effective tile: 64x64 output per workgroup.
// No shared memory — relies on register tiling with vec4 loads.
// Based on webgpu-blas (MIT) optimization pattern.

struct Params {
    M: u32,
    N: u32,
    K: u32,
    batch_stride_a: u32,  // M*K for batched, 0 for unbatched
    batch_stride_b: u32,  // K*N
    batch_stride_c: u32,  // M*N
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 4u;   // each thread computes TILE x TILE output elements

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let batch = wid.z;
    let row_base = gid.y * TILE;
    let col_base = gid.x * TILE;

    let a_off = batch * params.batch_stride_a;
    let b_off = batch * params.batch_stride_b;
    let c_off = batch * params.batch_stride_c;

    // Accumulator: TILE x TILE register tile.
    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < TILE; i++) {
        for (var j = 0u; j < TILE; j++) {
            acc[i][j] = 0.0;
        }
    }

    // Walk the K dimension.
    for (var k = 0u; k < params.K; k++) {
        // Load TILE elements from A column and B row.
        var a_col: array<f32, 4>;
        var b_row: array<f32, 4>;
        for (var i = 0u; i < TILE; i++) {
            let r = row_base + i;
            if r < params.M {
                a_col[i] = A[a_off + r * params.K + k];
            }
        }
        for (var j = 0u; j < TILE; j++) {
            let c = col_base + j;
            if c < params.N {
                b_row[j] = B[b_off + k * params.N + c];
            }
        }
        // Outer product accumulate.
        for (var i = 0u; i < TILE; i++) {
            for (var j = 0u; j < TILE; j++) {
                acc[i][j] += a_col[i] * b_row[j];
            }
        }
    }

    // Write results.
    for (var i = 0u; i < TILE; i++) {
        for (var j = 0u; j < TILE; j++) {
            let r = row_base + i;
            let c = col_base + j;
            if r < params.M && c < params.N {
                C[c_off + r * params.N + c] = acc[i][j];
            }
        }
    }
}
