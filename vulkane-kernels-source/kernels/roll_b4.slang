// Cyclic shift along a single axis. View input/output as a flat
// 3-tuple (outer, dim_size, inner).
//
// Wrapper passes `offset` already normalized to [0, dim_size) such
// that `src_j = (j + offset) mod dim_size`. (offset = (dim_size -
// shift_normalized) mod dim_size encodes the original `(j - shift)
// mod dim_size` semantics — keeps all arithmetic inside the kernel
// unsigned.)
//
// 4-byte element variant (f32 / i32 / u32).

struct Params {
    uint outer_count;
    uint dim_size;
    uint inner_count;
    uint offset;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

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

    uint src_j = (j + p.offset) % p.dim_size;
    uint src_idx = (o * p.dim_size + src_j) * p.inner_count + n;
    output[tid] = input[src_idx];
}
