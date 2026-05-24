// Flip along a single axis. Rank-N input (padded to rank 4 with
// leading 1s), per-input strides — output is always contig over the
// input's shape, so the dispatch derives output strides from the
// shape inline. 2-byte element variant (f16 / bf16); element-level
// walk via `uint16_t` binding (VK_KHR_16bit_storage +
// shaderFloat16 / storageBuffer16BitAccess; both already on per V.3.E).
//
// Output coord = input coord with the `axis` dim reversed.

struct Params {
    uint out_size;
    uint axis;              // which dim is flipped (0..3)
    uint _pad0, _pad1;
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>       p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.out_size) return;

    // Decompose tid into per-dim coords over the contig output shape.
    uint c3 = tid % p.shape3;
    uint r2 = tid / p.shape3;
    uint c2 = r2 % p.shape2;
    uint r1 = r2 / p.shape2;
    uint c1 = r1 % p.shape1;
    uint c0 = r1 / p.shape1;

    // Flip the coord at `axis`. The axis ladder is a 4-way conditional;
    // SPIR-V predication makes the wasted branches single-cycle.
    if      (p.axis == 0u) c0 = (p.shape0 - 1u) - c0;
    else if (p.axis == 1u) c1 = (p.shape1 - 1u) - c1;
    else if (p.axis == 2u) c2 = (p.shape2 - 1u) - c2;
    else                   c3 = (p.shape3 - 1u) - c3;

    uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
    output[tid] = input[in_off];
}
