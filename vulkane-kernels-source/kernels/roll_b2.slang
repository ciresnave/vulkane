// Cyclic shift along a single axis. Rank-N input (padded to rank 4
// with leading 1s), per-input strides — output is contig over the
// input's shape. 2-byte element variant (f16 / bf16) via uint16_t
// binding (VK_KHR_16bit_storage).
//
// Wrapper passes `offset` already normalized to [0, shape[axis]) such
// that `src_coord = (out_coord + offset) mod shape[axis]`. (offset =
// (shape[axis] - shift_normalized) mod shape[axis] encodes the
// original `(j - shift) mod dim_size` semantics — keeps all
// arithmetic inside the kernel unsigned.)

struct Params {
    uint out_size;
    uint axis;
    uint offset;            // pre-normalized to [0, shape[axis])
    uint _pad;
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

    uint c3 = tid % p.shape3;
    uint r2 = tid / p.shape3;
    uint c2 = r2 % p.shape2;
    uint r1 = r2 / p.shape2;
    uint c1 = r1 % p.shape1;
    uint c0 = r1 / p.shape1;

    if      (p.axis == 0u) c0 = (c0 + p.offset) % p.shape0;
    else if (p.axis == 1u) c1 = (c1 + p.offset) % p.shape1;
    else if (p.axis == 2u) c2 = (c2 + p.offset) % p.shape2;
    else                   c3 = (c3 + p.offset) % p.shape3;

    uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
    output[tid] = input[in_off];
}
