// PadBackward (constant mode), 4-byte elements (f32, i32, u32).
//
// Forward Pad const copies in[idx] → out[idx + left_pad] (and writes
// fill where out is outside the unpadded region). Backward: for each
// INPUT position, the corresponding output position is just
// `in_coord[d] + left_pad[d]`. No accumulation (each input position
// is hit exactly once in the forward pass).
//
// One thread per input element. Reads grad_out at the unpadded
// position; writes grad_in directly.
//
// `shape_buf` layout: [in_shape[0..rank], out_shape[0..rank],
//                     left_pad[0..rank]]. Same as the forward Pad
// kernels (only in_shape is iterated here).

struct Params {
    uint n_in;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   grad_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> grad_in;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

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
        // Stride of in_shape[d] = product(in_shape[d+1..rank]).
        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];   // in_shape[e]
        }
        uint in_coord = remainder / in_stride;
        remainder -= in_coord * in_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint out_coord = in_coord + left_pad;

        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];   // out_shape[e]
        }
        out_off += out_coord * out_stride;
    }

    grad_in[tid] = grad_out[out_off];
}
