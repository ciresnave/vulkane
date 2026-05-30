// IndexAdd — bf16 via sub-word CAS atomic add. Each u32 word holds
// two bf16 lanes; the kernel CAS-replaces only the target half.

struct Params {
    uint outer_count;
    uint base_dim_size;
    uint n_indices;
    uint inner_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   indices;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   src;     // packed bf16
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out;     // packed bf16
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

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
    uint old_word = out[word_idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        uint old_bits = (lane == 0u) ? (old_word & 0xFFFFu) : (old_word >> 16u);
        float old_val = bf16_bits_to_f32(old_bits);
        float new_val = old_val + add_val;
        uint new_bits = f32_to_bf16_bits(new_val);
        uint new_word = (lane == 0u)
            ? ((old_word & 0xFFFF0000u) | new_bits)
            : ((old_word & 0x0000FFFFu) | (new_bits << 16u));
        uint orig;
        InterlockedCompareExchange(out[word_idx], old_word, new_word, orig);
        if (orig == old_word) return;
        old_word = orig;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    uint total = p.outer_count * p.n_indices * p.inner_count;
    if (tid >= total) return;

    uint inner_per_outer = p.n_indices * p.inner_count;
    uint outer = tid / inner_per_outer;
    uint rem = tid - outer * inner_per_outer;
    uint k = rem / p.inner_count;
    uint inner = rem - k * p.inner_count;

    uint target = indices[k];
    uint src_off = (outer * p.n_indices + k) * p.inner_count + inner;
    uint dst_off = (outer * p.base_dim_size + target) * p.inner_count + inner;

    // Read one bf16 src lane.
    uint src_word = src[src_off >> 1u];
    uint src_lane = src_off & 1u;
    uint src_bits = (src_lane == 0u) ? (src_word & 0xFFFFu) : (src_word >> 16u);
    float src_val = bf16_bits_to_f32(src_bits);

    atomic_add_bf16(dst_off, src_val);
}
