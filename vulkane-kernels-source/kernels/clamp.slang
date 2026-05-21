// Element-wise clamp: y = clamp(x, lo, hi). One thread per element.
//
// Matches the affine.slang shape (1 input, 1 output, scalar params)
// to keep the descriptor layout = layout_2s1u and the pipeline
// dispatch path identical.

struct Params {
    uint n;
    uint _pad;
    float lo;
    float hi;
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
    float x = input[idx];
    output[idx] = clamp(x, p.lo, p.hi);
}
