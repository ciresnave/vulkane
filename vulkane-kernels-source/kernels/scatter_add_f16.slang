// ScatterAdd along `dim` — f16. Same sub-word CAS pattern as
// scatter_add_bf16: each u32 word holds TWO f16 lanes, so atomic
// accumulation reads the full word, replaces only the target lane's
// 16-bit half, and CAS-back. Math is done at f32 via Slang's
// `f16tof32` / `f32tof16` half-word builtins.

struct Params {
    uint n_src;
    uint rank;
    uint dim;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   indices;  // U32 lanes
[[vk::binding(1, 0)]] StructuredBuffer<uint>   src;      // 2 f16 / u32
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out;      // 2 f16 / u32
[[vk::binding(3, 0)]] StructuredBuffer<uint>   shape_buf;
[[vk::binding(4, 0)]] ConstantBuffer<Params>   p;

void atomic_add_f16(uint lane_idx, float add_val) {
    uint word_idx = lane_idx >> 1u;
    uint lane = lane_idx & 1u;
    uint old_word = out[word_idx];
    for (uint iter = 0u; iter < 1000u; iter++) {
        uint old_bits = (lane == 0u) ? (old_word & 0xFFFFu) : (old_word >> 16u);
        float old_val = f16tof32(old_bits);
        float new_val = old_val + add_val;
        uint new_bits = f32tof16(new_val) & 0xFFFFu;
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

    // Read one f16 src lane at `tid`.
    uint src_word = src[tid >> 1u];
    uint src_lane = tid & 1u;
    uint src_bits = (src_lane == 0u) ? (src_word & 0xFFFFu) : (src_word >> 16u);
    float src_val = f16tof32(src_bits);

    atomic_add_f16(dst_off, src_val);
}
