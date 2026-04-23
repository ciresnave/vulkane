// Affine: y = x * mul + add. Handles AddScalar and MulScalar.
//
// Replaces affine.wgsl. Written in Slang per the shader policy.

struct Params {
    uint n;
    uint _pad;
    float mul;
    float add;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.n) return;
    output[idx] = input[idx] * p.mul + p.add;
}
