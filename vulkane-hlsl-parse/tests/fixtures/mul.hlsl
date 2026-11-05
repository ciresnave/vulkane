// Pure HLSL (no Slang syntax): elementwise multiply, explicit registers.
StructuredBuffer<float> input0 : register(t0);
StructuredBuffer<float> input1 : register(t1);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void mul(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    output[i] = input0[i] * input1[i];
}
