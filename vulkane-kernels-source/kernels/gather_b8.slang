// Gather along `dim`, 8-byte elements (f64, i64). Same shape as
// gather_b4 with uint64_t storage.

struct Params {
    uint n_out;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t>   src;
[[vk::binding(1, 0)]] StructuredBuffer<uint>       indices;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t> output;
[[vk::binding(3, 0)]] StructuredBuffer<uint>       shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>       p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_out) return;

    uint rank = p.rank;
    uint dim = p.dim;
    uint src_off = 0u;

    uint remainder = tid;
    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint src_coord = (d == dim) ? indices[tid] : out_coord;

        uint src_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_stride *= shape_buf[e];
        }
        src_off += src_coord * src_stride;
    }

    output[tid] = src[src_off];
}
