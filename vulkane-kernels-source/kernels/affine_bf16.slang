// Affine: y = x * mul + add. bf16 variant. Pair-thread: each thread
// handles one u32 word = two bf16 lanes, requires contiguous input.
// (No strided KernelCaps — Fuel auto-Contiguizes before dispatch.)
// Math at f32 (matches the half-precision convention).

struct Params {
    uint  n_pairs;     // pair count (= n_elements / 2; wrapper guarantees n is even)
    uint  _pad0;
    float mul;
    float add;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // 2 bf16 / u32
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

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

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    if (pair_idx >= p.n_pairs) return;

    uint packed = input[pair_idx];
    float x0 = bf16_bits_to_f32(packed & 0xFFFFu);
    float x1 = bf16_bits_to_f32(packed >> 16u);

    float y0 = x0 * p.mul + p.add;
    float y1 = x1 * p.mul + p.add;

    uint out_lo = f32_to_bf16_bits(y0);
    uint out_hi = f32_to_bf16_bits(y1);
    output[pair_idx] = out_lo | (out_hi << 16u);
}
