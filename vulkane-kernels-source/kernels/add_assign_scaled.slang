// In-place scaled accumulate: dst[i] += src[i] * scale.
// One thread per element. Reads dst, reads src, writes dst.
//
// Replaces add_assign_scaled.wgsl. Written in Slang per the shader policy.

struct Params {
    uint n;
    float scale;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(1, 0)]] StructuredBuffer<float>   src;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint i = gid.x;
    if (i >= p.n) return;
    dst[i] = dst[i] + src[i] * p.scale;
}
