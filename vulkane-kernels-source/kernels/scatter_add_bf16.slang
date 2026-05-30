// ScatterAdd along `dim` — bf16. Same shape as scatter_add_f32 but
// each u32 word holds TWO bf16 lanes, so atomic accumulation runs a
// sub-word CAS: read the full word, replace only the target lane's
// bottom/upper 16 bits, CAS-back. Math is done at f32 (the bandwidth
// saving from bf16 is irrelevant once we hit the CAS loop).
//
// Wrapper copies base → out before dispatch and binds `out` with a
// u32-rounded descriptor range so the final half-word survives
// robust-access discard if n_dst is odd.

struct Params {
    uint n_src;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   indices;  // U32 lanes
[[vk::binding(1, 0)]] StructuredBuffer<uint>   src;      // 2 bf16 / u32
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out;      // 2 bf16 / u32
[[vk::binding(3, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>   p;

// bf16 -> f32: zero-extend 16 bits into the upper half of a u32, then
// reinterpret as f32. Preserves NaN / Inf / denormal bit patterns.
float bf16_bits_to_f32(uint b) {
    return asfloat(b << 16u);
}

// f32 -> bf16: round-to-nearest-even of the upper 16 bits.
// NaN inputs are mapped to the canonical bf16 qNaN (0x7FC0).
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
    if (tid >= p.n_src) return;

    uint rank = p.rank;
    uint dim = p.dim;
    uint dst_off = 0u;

    uint remainder = tid;
    for (uint d = 0u; d < rank; d++) {
        uint src_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            src_stride *= shape_buf[e];
        }
        uint src_coord = remainder / src_stride;
        remainder -= src_coord * src_stride;

        uint dst_coord = (d == dim) ? indices[tid] : src_coord;

        uint dst_stride = 1u;
        for (uint e = d + 1u; e < rank; e++) {
            dst_stride *= shape_buf[rank + e];
        }
        dst_off += dst_coord * dst_stride;
    }

    // Read one bf16 src lane at `tid`.
    uint src_word = src[tid >> 1u];
    uint src_lane = tid & 1u;
    uint src_bits = (src_lane == 0u) ? (src_word & 0xFFFFu) : (src_word >> 16u);
    float src_val = bf16_bits_to_f32(src_bits);

    atomic_add_bf16(dst_off, src_val);
}
