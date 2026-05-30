// PadBackward replicate mode — f64. Mirrors pad_backward_replicate_f32
// but uses u64 CAS for atomic double-add.

struct Params {
    uint n_out;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>     grad_out;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint64_t> grad_in;
[[vk::binding(2, 0)]] StructuredBuffer<uint>       shape_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>       p;

uint replicate_coord(uint out_coord, uint left_pad, uint in_dim) {
    if (out_coord < left_pad) {
        return 0u;
    } else if (out_coord >= left_pad + in_dim) {
        return in_dim - 1u;
    } else {
        return out_coord - left_pad;
    }
}

double bits_to_double(uint64_t bits) {
    uint lo = uint(bits & 0xFFFFFFFFu);
    uint hi = uint(bits >> 32);
    return asdouble(lo, hi);
}

uint64_t double_to_bits(double v) {
    uint lo, hi;
    asuint(v, lo, hi);
    return (uint64_t(hi) << 32) | uint64_t(lo);
}

void atomic_add_f64(uint idx, double add_val) {
    uint64_t old_bits = grad_in[idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        double new_val = bits_to_double(old_bits) + add_val;
        uint64_t new_bits = double_to_bits(new_val);
        uint64_t orig;
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
            out_stride *= shape_buf[rank + e];
        }
        uint out_coord = remainder / out_stride;
        remainder -= out_coord * out_stride;

        uint left_pad = shape_buf[2u * rank + d];
        uint in_dim   = shape_buf[d];
        uint in_coord = replicate_coord(out_coord, left_pad, in_dim);

        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];
        }
        in_off += in_coord * in_stride;
    }

    atomic_add_f64(in_off, grad_out[tid]);
}
