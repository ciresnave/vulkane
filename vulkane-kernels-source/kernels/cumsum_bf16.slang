// Inclusive prefix sum along a single axis. Rank-N input (padded to
// rank 4) + per-input strides; output is contig over the input's
// shape. Sequential per-slice walk — see cumsum_f32.slang for the
// algorithm comment.
//
// bf16 variant — Slang has no native `bfloat16_t`, so storage is
// `uint16_t` (one bf16 per u16 lane) via VK_KHR_16bit_storage +
// storageBuffer16BitAccess. Accumulator is f32 with manual bit-level
// bf16↔f32 conversion at the edges. (Long sums in bf16 storage with
// f32 accumulation give better numerical stability than in-storage
// accumulation; matches the f16 kernel's choice of "accumulator =
// storage dtype" only in spirit since bf16 isn't a native math type.)

struct Params {
    uint slice_count;
    uint axis;
    uint dim_size;
    uint _pad;
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>       p;

// bf16 -> f32: zero-extend 16 bits into the upper half of a u32, then
// reinterpret as f32. Preserves NaN / Inf / denormal bit patterns.
float bf16_bits_to_f32(uint16_t b) {
    return asfloat(uint(b) << 16u);
}

// f32 -> bf16: round-to-nearest-even of the upper 16 bits.
// NaN inputs map to the canonical bf16 qNaN (0x7FC0).
uint16_t f32_to_bf16_bits(float x) {
    uint bits = asuint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;

    if (exp_bits == 0xFFu && mant_low != 0u) {
        return uint16_t((bits & 0x80000000u) >> 16u | 0x7FC0u);
    }

    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return uint16_t((rounded >> 16u) & 0xFFFFu);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint sid = gid.x;
    if (sid >= p.slice_count) return;

    uint cs0 = (p.axis == 0u) ? 1u : p.shape0;
    uint cs1 = (p.axis == 1u) ? 1u : p.shape1;
    uint cs2 = (p.axis == 2u) ? 1u : p.shape2;
    uint cs3 = (p.axis == 3u) ? 1u : p.shape3;

    uint c3 = sid % cs3;
    uint r2 = sid / cs3;
    uint c2 = r2 % cs2;
    uint r1 = r2 / cs2;
    uint c1 = r1 % cs1;
    uint c0 = r1 / cs1;

    uint out_s2 = p.shape3;
    uint out_s1 = p.shape2 * out_s2;
    uint out_s0 = p.shape1 * out_s1;

    float running = 0.0;
    for (uint j = 0u; j < p.dim_size; ++j) {
        uint q0 = (p.axis == 0u) ? j : c0;
        uint q1 = (p.axis == 1u) ? j : c1;
        uint q2 = (p.axis == 2u) ? j : c2;
        uint q3 = (p.axis == 3u) ? j : c3;

        uint in_off = q0 * p.in_s0 + q1 * p.in_s1 + q2 * p.in_s2 + q3 * p.in_s3;
        running += bf16_bits_to_f32(input[in_off]);

        uint out_off = q0 * out_s0 + q1 * out_s1 + q2 * out_s2 + q3;
        output[out_off] = f32_to_bf16_bits(running);
    }
}
