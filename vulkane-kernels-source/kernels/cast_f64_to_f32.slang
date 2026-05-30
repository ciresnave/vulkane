// Cast f64 → f32 (narrowing, rounds to nearest-even per IEEE 754).
// One thread per element. Requires shaderFloat64.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<double> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.n) return;
    output[idx] = float(input[idx]);
}
