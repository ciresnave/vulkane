// Pad with constant fill, 8-byte elements (f64, i64, u64). One thread
// per output element. Storage as uint64_t — Slang's `uint64_t` maps
// to OpTypeInt 64. Requires shaderInt64 capability (which every
// driver that exposes shaderFloat64 also exposes).
//
// `shape_buf` layout same as pad_const_b4.

struct Params {
    uint   n_out;
    uint   rank;
    uint64_t fill_value;   // 8-byte bit-pattern (caller encodes)
};

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t>   src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint64_t> dst;
[[vk::binding(2, 0)]] StructuredBuffer<uint>       shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>       p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_out) return;

    uint rank = p.rank;
    uint remainder = tid;
    uint in_off = 0u;
    bool in_bounds = true;

    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint in_dim = shape_buf[d];
        if (out_coord < left_pad || out_coord >= left_pad + in_dim) {
            in_bounds = false;
        } else {
            uint in_coord = out_coord - left_pad;
            uint in_stride = 1u;
            for (uint e = d + 1u; e < rank; e++) {
                in_stride *= shape_buf[e];
            }
            in_off += in_coord * in_stride;
        }
    }

    dst[tid] = in_bounds ? src[in_off] : p.fill_value;
}
