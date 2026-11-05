// SPDX-License-Identifier: MIT OR Apache-2.0
// Pure HLSL: cbuffer with register, groupshared, semantics, GroupIndex. Residue (groupshared) for the lifters.
cbuffer Params : register(b0)
{
    uint n;
};
StructuredBuffer<float> input0 : register(t0);
RWStructuredBuffer<float> output : register(u0);
groupshared float tile[128];

[numthreads(128, 1, 1)]
void tiled(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    tile[gi] = input0[tid.x];
    GroupMemoryBarrierWithGroupSync();
    if (tid.x < n) output[tid.x] = tile[gi];
}
