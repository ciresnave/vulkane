// Strided copy: read `out_size` elements from `input` using the
// given source strides + start offset, write contiguously into
// `output` starting at `dst_offset`.
//
// Handles permute, broadcast (via stride=0 on broadcast dims),
// concat (via dst_offset), and slice (via src_offset).
//
// `shape_strides` storage buffer layout:
//   [shape[0], shape[1], ..., shape[rank-1],
//    stride[0], stride[1], ..., stride[rank-1]]
//
// Replaces strided_copy.wgsl. Written in Slang per the shader policy.

struct Params {
    uint out_size;
    uint rank;
    uint src_offset;
    uint dst_offset;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] StructuredBuffer<uint>    shape_strides;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint out_idx = gid.x;
    if (out_idx >= p.out_size) return;

    // Unflatten out_idx in row-major order using the output shape.
    // Output is contiguous; compute coord per dim by successive division.
    uint remainder = out_idx;
    uint src_flat = p.src_offset;

    // Compute the contiguous stride for each dim of the output shape.
    // We build strides[d] = product of shape[d+1..rank].
    // Then coord = remainder / strides[d], remainder %= strides[d].
    for (uint d = 0u; d < p.rank; d++) {
        uint dim_stride = 1u;
        for (uint e = d + 1u; e < p.rank; e++) {
            dim_stride *= shape_strides[e];
        }
        uint coord = remainder / dim_stride;
        remainder -= coord * dim_stride;
        src_flat += coord * shape_strides[p.rank + d];
    }

    output[p.dst_offset + out_idx] = input[src_flat];
}
