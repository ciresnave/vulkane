#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Small-shape bf16 × bf16 → f32 matmul fallback. One thread per
// output element; f32 accumulator. Handles ANY shape (no 16-tile
// divisibility requirement). Slow but correct — picker routes here
// when the cooperative-matrix shape constraints fail (M < 16,
// M % 16 != 0, N % 16 != 0, or m == 1 matvec cases).
//
// 16×16 workgroup; dispatch (ceil(N/16), ceil(M/16), batch).

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { uint16_t A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { uint16_t B[]; };
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

float bf16_u16_to_f32(uint16_t bits) {
    return uintBitsToFloat(uint(bits) << 16);
}

void main() {
    uint j = gl_GlobalInvocationID.x;   // N
    uint i = gl_GlobalInvocationID.y;   // M
    uint batch = gl_GlobalInvocationID.z;
    if (i >= p.M || j >= p.N) return;

    uint a_off = batch * p.sa_batch;
    uint b_off = (batch / p.n_rep) * p.sb_batch;
    uint c_off = batch * p.sc_batch;

    float acc = 0.0;
    for (uint kk = 0u; kk < p.K; ++kk) {
        float a  = bf16_u16_to_f32(A[a_off + i * p.sa_row + kk * p.sa_col]);
        float bv = bf16_u16_to_f32(B[b_off + kk * p.sb_row + j * p.sb_col]);
        acc += a * bv;
    }
    C[c_off + i * p.N + j] = acc;
}
