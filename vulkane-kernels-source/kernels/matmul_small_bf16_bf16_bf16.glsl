#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
// Small-shape bf16 × bf16 → bf16 matmul fallback. Same as
// matmul_small_bf16_bf16_f32 but the f32 accumulator is downcast
// to bf16 on the final store (round-to-nearest-even, canonical NaN).
//
// One thread per output element; no shared-memory staging needed
// because each thread owns its own (i, j) slot in C.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { uint16_t A[]; };
layout(set = 0, binding = 1, std430) readonly buffer BBuf { uint16_t B[]; };
layout(set = 0, binding = 2, std430) buffer CBuf { uint16_t C[]; };

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

uint f32_to_bf16_bits(float x) {
    uint bits = floatBitsToUint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

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
        float a  = bf16_u16_to_f32(A[a_off + i * p.sa_row + kk * p.sa_col]);
        float bv = bf16_u16_to_f32(B[b_off + kk * p.sb_row + j * p.sb_col]);
        acc += a * bv;
    }
    C[c_off + i * p.N + j] = uint16_t(f32_to_bf16_bits(acc));
}
