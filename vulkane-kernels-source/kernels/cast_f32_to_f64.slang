// Cast f32 → f64 (widening, lossless). One thread per element.
//
// f32 is exactly representable in f64, so this is a pure widening
// conversion with no rounding. Requires shaderFloat64.

struct Params {
    uint n;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.n) return;
    output[idx] = double(input[idx]);
}
