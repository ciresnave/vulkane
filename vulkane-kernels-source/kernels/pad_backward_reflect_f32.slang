// PadBackward reflect mode — f32. Forward reflect can map multiple
// output positions to the same input position; backward accumulates
// each grad_out contribution into the corresponding grad_in slot.
//
// One thread per OUTPUT element. Compute the input coord per axis
// via reflect_coord (same formula as forward pad_reflect_b4), then
// atomically add grad_out[tid] into grad_in[in_off] via uint CAS
// (no VK_EXT_shader_atomic_float required).
//
// PRECONDITION: the wrapper zero-fills grad_in before dispatch (so
// the atomic OR-style accumulation starts from 0).
//
// `shape_buf` layout: [in_shape[0..rank], out_shape[0..rank],
//                     left_pad[0..rank]]. Same as forward pad_b4.

struct Params {
    uint n_out;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>  grad_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> grad_in;   // f32 reinterpreted as uint
[[vk::binding(2, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

uint reflect_coord(uint out_coord, uint left_pad, uint in_dim) {
    if (out_coord < left_pad) {
        return left_pad - out_coord;
    } else if (out_coord >= left_pad + in_dim) {
        uint excess = out_coord - left_pad - (in_dim - 1u);
        return in_dim - 1u - excess;
    } else {
        return out_coord - left_pad;
    }
}

void atomic_add_f32(uint idx, float add_val) {
    uint old_bits = grad_in[idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        float new_val = asfloat(old_bits) + add_val;
        uint new_bits = asuint(new_val);
        uint orig;
        InterlockedCompareExchange(grad_in[idx], old_bits, new_bits, orig);
        if (orig == old_bits) return;
        old_bits = orig;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.n_out) return;

    uint rank = p.rank;
    uint remainder = tid;
    uint in_off = 0u;

    for (uint d = 0u; d < rank; d++) {
        uint out_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            out_stride *= shape_buf[rank + e];   // out_shape[e]
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint in_dim   = shape_buf[d];
        uint in_coord = reflect_coord(out_coord, left_pad, in_dim);

        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];   // in_shape[e]
        }
        in_off += in_coord * in_stride;
    }

    atomic_add_f32(in_off, grad_out[tid]);
}
