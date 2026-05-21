// Element-wise integer power: y = x^exp.
//
// Mirrors affine.slang shape (1 input, 1 output, scalar params).
// One thread per element; integer exponent supplied as i32.

struct Params {
    uint n;
    int  exp;
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
    if (idx >= p.n) return;
    float x = input[idx];
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
