// Pad with constant fill, 4-byte elements (f32, i32, u32). One thread
// per output element. If the output position maps to a valid input
// position (after subtracting left-padding), copy from input;
// otherwise write `fill_value`.
//
// `shape_buf` layout: [in_shape[0..rank], out_shape[0..rank],
//                     left_pad[0..rank]]
// (right_pad is derived from out_shape - in_shape - left_pad; the
// kernel only needs the bounds check, so right_pad is implicit.)
//
// Rank up to 8 supported.

struct Params {
    uint n_out;     // total output elements (iteration bound)
    uint rank;
    uint fill_value;  // u32 bit-pattern of the fill (caller encodes per dtype)
    uint _pad0;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   src;        // input, contiguous in in_shape
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;        // output, contiguous in out_shape
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;  // in_shape + out_shape + left_pad
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_out) return;

    uint rank = p.rank;
    // Decompose tid into per-dim out-coords (row-major). For each
    // dim d, divide by stride(d) = product(out_shape[d+1..rank]).
    uint remainder = tid;
    uint in_off = 0u;
    bool in_bounds = true;

    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];   // out_shape[e]
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint in_dim = shape_buf[d];   // in_shape[d]
        // Check if out_coord falls inside the valid (non-pad) zone.
        if (out_coord < left_pad || out_coord >= left_pad + in_dim) {
            in_bounds = false;
        } else {
            uint in_coord = out_coord - left_pad;
            uint in_stride = 1u;
            for (uint e = d + 1u; e < rank; e++) {
                in_stride *= shape_buf[e];   // in_shape[e]
            }
            in_off += in_coord * in_stride;
        }
    }

    dst[tid] = in_bounds ? src[in_off] : p.fill_value;
}
