// Element-wise clamp: y = clamp(x, lo, hi). One thread per element.
//
// Stride-aware: shares the affine.slang Params shape so the
// descriptor layout = layout_2s1u and the pipeline dispatch path
// stay identical, while supporting non-contiguous inputs via per-dim
// shape + stride decomposition when `flags & 1u == 0`.

struct Params {
    uint out_size;
    uint flags;           // bit 0: input_contiguous
    float lo;
    float hi;
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    float x;
    bool in_contig = (p.flags & 1u) != 0u;

    if (in_contig) {
        x = input[idx];
    } else {
        uint c3 = idx % p.shape3;
        uint r2 = idx / p.shape3;
        uint c2 = r2 % p.shape2;
        uint r1 = r2 / p.shape2;
        uint c1 = r1 % p.shape1;
        uint c0 = r1 / p.shape1;
        uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
        x = input[in_off];
    }

    output[idx] = clamp(x, p.lo, p.hi);
}
