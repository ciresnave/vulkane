// Signed-stride Contiguize for 2-byte elements (f16 / bf16).
// 16-bit-storage element access; see strided_copy_signed_b4.slang for
// the algorithm rationale.

struct Params {
    uint out_size;
    uint rank;
    int  src_offset;
    uint dst_offset;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint16_t> output;
[[vk::binding(2, 0)]] StructuredBuffer<uint>       shape_strides;
[[vk::binding(3, 0)]] ConstantBuffer<Params>       p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint out_idx = gid.x;
    if (out_idx >= p.out_size) return;

    uint remainder = out_idx;
    int  src_flat = p.src_offset;
    for (uint d = 0u; d < p.rank; d++) {
        uint dim_stride = 1u;
        for (uint e = d + 1u; e < p.rank; e++) {
            dim_stride *= shape_strides[e];
        }
        uint coord = remainder / dim_stride;
        remainder -= coord * dim_stride;
        int stride_signed = asint(shape_strides[p.rank + d]);
        src_flat += int(coord) * stride_signed;
    }

    output[p.dst_offset + out_idx] = input[uint(src_flat)];
}
