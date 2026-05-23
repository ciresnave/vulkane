// Element-wise unary ops for f16 with per-input stride support.
// Mirrors unary.slang's 13-op surface (op_id 0..12) and stride-aware
// param layout — only the buffer element type and math precision
// differ. Operates on `float16_t` directly via `VK_KHR_16bit_storage`
// + `shaderFloat16`. The 16-bit storage buffer extension is required
// for `StructuredBuffer<float16_t>`.
//
// Computation is done in f16 throughout — no f32 widening. Math
// builtins (exp/log/sin/cos/tanh) accept float16_t via Slang's
// overload set in this configuration.

struct Params {
    uint out_size;
    uint op_id;
    uint rank;
    uint flags;           // bit 0: input_contiguous
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>        p;

float16_t apply(uint op, float16_t x) {
    if (op == 0u)  return -x;                                // Neg
    if (op == 1u)  return x * x;                             // Sqr
    if (op == 2u)  return sqrt(x);                           // Sqrt
    if (op == 3u)  return exp(x);                            // Exp
    if (op == 4u)  return log(x);                            // Log
    if (op == 5u)  return sin(x);                            // Sin
    if (op == 6u)  return cos(x);                            // Cos
    if (op == 7u)  return tanh(x);                           // Tanh
    if (op == 8u)  return float16_t(1.0) / (float16_t(1.0) + exp(-x));   // Sigmoid
    if (op == 9u)  return x / (float16_t(1.0) + exp(-x));    // Silu
    if (op == 10u) {                                         // Gelu (tanh approx)
        float16_t k0 = float16_t(0.7978845608);
        float16_t k1 = float16_t(0.044715);
        return float16_t(0.5) * x
            * (float16_t(1.0) + tanh(k0 * (x + k1 * x * x * x)));
    }
    if (op == 11u) return max(x, float16_t(0.0));            // Relu
    return (x > float16_t(0.0)) ? float16_t(1.0) : float16_t(0.0);  // Step
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    float16_t x;
    bool in_contig = (p.flags & 1u) != 0u;

    if (in_contig) {
        x = input[idx];
    } else {
        uint c3 = idx % p.shape3;
        uint r2 = idx / p.shape3;
        uint c2 = r2 % p.shape2;
        uint r1 = r2 / p.shape2;
        uint c1 = r1 % p.shape1;
        uint c0 = r1 / p.shape1;
        uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
        x = input[in_off];
    }

    output[idx] = apply(p.op_id, x);
}
