// Cyclic shift along a single axis. View input/output as a flat
// 3-tuple (outer, dim_size, inner).
//
// Rule: out[o, j, n] = in[o, (j - shift) mod dim_size, n]
// where the modulo is Euclidean — negative shifts wrap forward.
//
// `shift` is passed as `i32` since dim_size fits in u32; the wrapper
// normalizes shift to (-dim_size, dim_size) before dispatch so we
// don't need int64 arithmetic inside the kernel.
//
// 4-byte element variant (f32 / i32 / u32).

struct Params {
    uint outer_count;
    uint dim_size;
    uint inner_count;
    int  shift;
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

    int  d = int(p.dim_size);
    int  src_j_signed = (int(j) - p.shift) % d;
    if (src_j_signed < 0) src_j_signed += d;
    uint src_j = uint(src_j_signed);

    uint src_idx = (o * p.dim_size + src_j) * p.inner_count + n;
    output[tid] = input[src_idx];
}
