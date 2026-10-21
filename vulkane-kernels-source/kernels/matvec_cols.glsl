#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// GEMV for M == 1 with a row-major [K, N] right operand (unit stride along N):
//   C[1, N] = A[1, K] @ B[K, N].
//
// matvec.glsl gives each output column its own workgroup and lets the 128
// threads stride over K, so neighbouring threads read B[k * N + col] and
// B[(k + 1) * N + col]: addresses N floats apart, one useful float per cache
// line. This kernel maps threads to COLUMNS instead. A workgroup owns 32
// columns; 8 lanes each load a vec4 (4 adjacent columns), so a warp reads
// consecutive floats of a B row (coalesced), and the 32 K-slices of the
// workgroup walk down K together. The slices are summed in a fixed order, so
// the result is deterministic (matvec.glsl's subgroupAdd order is not pinned).
//
// Requirements (checked by the host wrapper, which otherwise uses matvec.glsl):
//   N % 4 == 0, sb_col == 1, and B's row stride (sb_row) and batch stride
//   (sb_batch) multiples of 4 floats, so every vec4 load is 16-byte aligned.
//
// Dispatch: (ceil(N / 32), 1, batch). A may be strided (sa_col).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint LANES = 8u;       // threads along N, each owning 4 columns
const uint KSLICES = 32u;    // 256 / LANES threads walking down K
const uint COLS = LANES * 4u;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { vec4 B4[]; };
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

shared vec4 part[KSLICES][LANES];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint lane = tid % LANES;
    uint ks = tid / LANES;
    uint batch = gl_WorkGroupID.z;
    uint col = gl_WorkGroupID.x * COLS + lane * 4u;   // first of this lane's 4 columns

    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;

    vec4 acc0 = vec4(0.0);
    vec4 acc1 = vec4(0.0);
    vec4 acc2 = vec4(0.0);
    vec4 acc3 = vec4(0.0);
    if (col < p.N) {
        uint k = ks;
        // Four independent loads in flight per thread.
        for (; k + 3u * KSLICES < p.K; k += 4u * KSLICES) {
            float a0 = A[a_off + (k) * p.sa_col];
            float a1 = A[a_off + (k + KSLICES) * p.sa_col];
            float a2 = A[a_off + (k + 2u * KSLICES) * p.sa_col];
            float a3 = A[a_off + (k + 3u * KSLICES) * p.sa_col];
            vec4 b0 = B4[(b_off + (k) * p.sb_row + col) >> 2];
            vec4 b1 = B4[(b_off + (k + KSLICES) * p.sb_row + col) >> 2];
            vec4 b2 = B4[(b_off + (k + 2u * KSLICES) * p.sb_row + col) >> 2];
            vec4 b3 = B4[(b_off + (k + 3u * KSLICES) * p.sb_row + col) >> 2];
            acc0 += a0 * b0;
            acc1 += a1 * b1;
            acc2 += a2 * b2;
            acc3 += a3 * b3;
        }
        for (; k < p.K; k += KSLICES) {
            acc0 += A[a_off + k * p.sa_col] * B4[(b_off + k * p.sb_row + col) >> 2];
        }
    }
    part[ks][lane] = (acc0 + acc1) + (acc2 + acc3);
    barrier();

    if (ks == 0u && col < p.N) {
        vec4 t = vec4(0.0);
        for (uint i = 0u; i < KSLICES; ++i) {
            t += part[i][lane];
        }
        C[c_off + col + 0u] = t.x;
        C[c_off + col + 1u] = t.y;
        C[c_off + col + 2u] = t.z;
        C[c_off + col + 3u] = t.w;
    }
}
