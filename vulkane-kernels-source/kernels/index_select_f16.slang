// Row-wise lookup along a specified dim — f16.
//
// Same semantics as index_select.slang; storage is float16_t. Pure
// data movement (no math), so f16 is bit-exact end-to-end.
//
// Requires VK_KHR_16bit_storage + shaderFloat16.

struct Params {
    uint out_size;
    uint outer;
    uint axis_out;
    uint inner;
    uint axis_in;
    uint _pad0, _pad1, _pad2;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] StructuredBuffer<uint>        ids;
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>        p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    uint i = idx % p.inner;
    uint ko = idx / p.inner;
    uint k = ko % p.axis_out;
    uint o = ko / p.axis_out;

    uint src_axis = ids[k];
    uint safe_axis = min(src_axis, p.axis_in - 1u);
    uint src_idx = (o * p.axis_in + safe_axis) * p.inner + i;
    output[idx] = input[src_idx];
}
