// Element-wise unary ops for f64. Mirrors unary.slang's 13-op
// surface but operates on `double` (shaderFloat64 required).
//
// IMPORTANT NVIDIA driver quirk: if-chains across f64 math functions
// (`exp`, `log`, `sin`, `cos`, `tanh`) appear to JIT-compile poorly
// (host-side STATUS_ACCESS_VIOLATION at vkCmdDispatch on Windows + RTX
// 4070, even when the runtime op_id only takes the simple branches).
// Workaround: split into two pipelines — fast ops (Neg/Sqr/Sqrt/Max/
// Step) in this file, transcendentals (Exp/Log/Sin/Cos/Tanh/Sigmoid/
// Silu/Gelu) deferred to V.3.E.5 follow-up once the driver behavior
// is understood. Slang itself emits clean SPIR-V (verified via
// spirv-dis) — the issue is downstream in NVK's JIT.

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
