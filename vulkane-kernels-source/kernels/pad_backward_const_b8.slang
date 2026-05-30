// PadBackward (constant mode), 8-byte elements (f64, i64). Same shape
// as pad_backward_const_b4 with uint64_t storage.

struct Params {
    uint n_in;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t>   grad_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint64_t> grad_in;
[[vk::binding(2, 0)]] StructuredBuffer<uint>       shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>       p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_in) return;

    uint rank = p.rank;
    uint remainder = tid;
    uint out_off = 0u;

    for (uint d = 0u; d < rank; d++) {
        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];
        }
        uint in_coord = remainder / in_stride;
        remainder -= in_coord * in_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint out_coord = in_coord + left_pad;

        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        out_off += out_coord * out_stride;
    }

    grad_in[tid] = grad_out[out_off];
}
