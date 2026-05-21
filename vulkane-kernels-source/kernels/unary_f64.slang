// Element-wise unary ops for f64. Mirrors unary.slang's 13-op
// surface but operates on `double` (shaderFloat64 required).
//
// Surface limited to Neg / Sqr / Sqrt / Relu / Step because
// GLSL.std.450's Exp / Log / Sin / Cos / Tanh / Sigmoid / Silu /
// Gelu instructions are spec-restricted to 16-bit and 32-bit float
// inputs — they reject `double`. Slang silently emits an invalid
// `OpExtInst Exp %double ...`, which spirv-val flags
// ("GLSL.std.450 Exp: expected Result Type to be a 16 or 32-bit
// scalar or vector float type"); on RTX 4070 + NVIDIA Windows the
// driver doesn't catch this at pipeline-create and crashes inside
// vkCmdDispatch instead.
//
// To add f64 transcendentals later: either (a) use OpenCL.std
// extended instruction set instead of GLSL.std.450 (its double
// variants are spec-allowed), or (b) compute in f32 and round-trip
// (acceptable precision for activations; loses range for log/exp
// of large magnitudes), or (c) hand-coded series expansion.

struct Params {
    uint n;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.n) return;
    double x = input[idx];
    double y;
    uint op = p.op_id;

    // Subset: Neg(0), Sqr(1), Sqrt(2), Relu(11), Step(12). Others
    // (3..10) bail at the dispatch wrapper layer for now.
    if (op == 0u) {
        y = -x;
    } else if (op == 1u) {
        y = x * x;
    } else if (op == 2u) {
        y = sqrt(x);
    } else if (op == 11u) {
        y = max(x, 0.0);
    } else {
        // op_id == 12 (Step), or fallback.
        y = (x > 0.0) ? 1.0 : 0.0;
    }
    output[idx] = y;
}
