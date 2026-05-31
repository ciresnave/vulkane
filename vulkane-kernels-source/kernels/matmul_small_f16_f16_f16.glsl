#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
// Small-shape f16 × f16 → f16 matmul fallback. f32 accumulator;
// final downcast via `float16_t(acc)` (round-to-nearest-even).

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float16_t A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { float16_t B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { float16_t C[]; };

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

void main() {
    uint j = gl_GlobalInvocationID.x;
    uint i = gl_GlobalInvocationID.y;
    uint batch = gl_GlobalInvocationID.z;
    if (i >= p.M || j >= p.N) return;

    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;

    float acc = 0.0;
    for (uint kk = 0u; kk < p.K; ++kk) {
        float a  = float(A[a_off + i * p.sa_row + kk * p.sa_col]);
        float bv = float(B[b_off + kk * p.sb_row + j * p.sb_col]);
        acc += a * bv;
    }
    C[c_off + i * p.N + j] = float16_t(acc);
}
