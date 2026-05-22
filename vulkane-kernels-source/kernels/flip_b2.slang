// Flip for 2-byte elements (f16 / bf16). Walk per-element to keep
// the dim-reversal honest for any `inner_count` (including odd
// inner * dim_size totals that won't pair-pack cleanly). Bind storage
// as `uint16_t` via VK_KHR_16bit_storage so writes are well-defined
// at the half-word level — slangc + the NVIDIA driver both accept
// this when shaderFloat16 + storageBuffer16BitAccess are enabled,
// which fuel already turns on (per V.3.E memory).

struct Params {
    uint outer_count;
    uint dim_size;
    uint inner_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>       p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    uint plane = p.dim_size * p.inner_count;
    uint total = p.outer_count * plane;
    if (tid >= total) return;

    uint o = tid / plane;
    uint rest = tid % plane;
    uint j = rest / p.inner_count;
    uint n = rest % p.inner_count;

    uint src_j = (p.dim_size - 1u) - j;
    uint src_idx = (o * p.dim_size + src_j) * p.inner_count + n;
    output[tid] = input[src_idx];
}
