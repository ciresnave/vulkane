// PadBackward reflect mode — bf16. grad_out / grad_in are packed
// two-bf16-per-u32; atomic accumulation uses sub-word CAS (same
// primitive as scatter_add_bf16).

struct Params {
    uint n_out;
    uint rank;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   grad_out;   // packed bf16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> grad_in;    // packed bf16
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

float bf16_bits_to_f32(uint b) { return asfloat(b << 16u); }

uint f32_to_bf16_bits(float x) {
    uint bits = asuint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

void atomic_add_bf16(uint lane_idx, float add_val) {
    uint word_idx = lane_idx >> 1u;
    uint lane = lane_idx & 1u;
    uint old_word = grad_in[word_idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        uint old_bits = (lane == 0u) ? (old_word & 0xFFFFu) : (old_word >> 16u);
        float old_val = bf16_bits_to_f32(old_bits);
        float new_val = old_val + add_val;
        uint new_bits = f32_to_bf16_bits(new_val);
        uint new_word = (lane == 0u)
            ? ((old_word & 0xFFFF0000u) | new_bits)
            : ((old_word & 0x0000FFFFu) | (new_bits << 16u));
        uint orig;
        InterlockedCompareExchange(grad_in[word_idx], old_word, new_word, orig);
        if (orig == old_word) return;
        old_word = orig;
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
        uint in_coord = reflect_coord(out_coord, left_pad, in_dim);

        uint in_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            in_stride *= shape_buf[e];
        }
        in_off += in_coord * in_stride;
    }

    // Read one bf16 grad_out lane at `tid`.
    uint go_word = grad_out[tid >> 1u];
    uint go_lane = tid & 1u;
    uint go_bits = (go_lane == 0u) ? (go_word & 0xFFFFu) : (go_word >> 16u);
    float go_val = bf16_bits_to_f32(go_bits);

    atomic_add_bf16(in_off, go_val);
}
