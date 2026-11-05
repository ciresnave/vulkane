// SPDX-License-Identifier: MIT OR Apache-2.0
// Pure HLSL: serial inclusive scan.
StructuredBuffer<float> input0 : register(t0);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void csum(uint3 tid : SV_DispatchThreadID)
{
    float acc = 0.0f;
    for (uint i = 0; i < 256; i++) { acc += input0[i]; output[i] = acc; }
}
