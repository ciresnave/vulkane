// Element-wise integer power: y = x^exp.
//
// Stride-aware: shares the affine.slang Params shape (1 input,
// 1 output, scalar params + per-dim shape/stride + flags) so the
// descriptor layout stays identical and the auto-Contiguize gate
// can be opt-in turned off (KernelCaps::strided_input).
// One thread per element; integer exponent supplied as i32.

struct Params {
    uint out_size;
    uint flags;           // bit 0: input_contiguous
    int  exp;
    uint _pad;
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

// Slang/HLSL has pow(float, float). For integer exponents the
// common fast path is repeated multiplication; we keep it simple
// with `pow(x, float(exp))` since exponents in real usage stay
// small (square, cube, etc.) and shader-call overhead dominates
// over the FP-vs-integer-mul difference.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    float x;
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

    int e = p.exp;
    float y;
    if (e == 0) {
        y = 1.0f;
    } else if (e == 1) {
        y = x;
    } else if (e == 2) {
        y = x * x;
    } else if (e == 3) {
        y = x * x * x;
    } else {
        // Generic case: pow(x, e). Handles negative exponents too
        // (pow(0, -k) → +inf which mirrors CPU behavior).
        y = pow(x, float(e));
    }
    output[idx] = y;
}
