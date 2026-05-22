// Element-wise binary ops on bf16. Mirrors binary.slang / binary_f16.slang's
// 6-op surface (Add/Sub/Mul/Div/Max/Min) and stride-aware param layout
// — only the bf16<->f32 round-trip differs.
//
// bf16 lanes are packed two-per-u32 and unpacked into f32 for the
// arithmetic; result is packed back via round-to-nearest-even.
//
// Wrapper guarantees `out_size` is even and the contiguous path
// (flags == 3) addresses pair-aligned u32 words. The strided path
// reads single lanes by addressing the input u32 then masking the
// requested half — this keeps the kernel correct under arbitrary
// strides (the typical broadcast / view cases of binary).

struct Params {
    uint out_size;     // element count
    uint op_id;
    uint rank;
    uint flags;        // bit 0: a contiguous; bit 1: b contiguous

    uint shape0, shape1, shape2, shape3;
    uint a_s0, a_s1, a_s2, a_s3;
    uint b_s0, b_s1, b_s2, b_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   a_buf;     // packed bf16 lanes
[[vk::binding(1, 0)]] StructuredBuffer<uint>   b_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

static const uint OP_ADD = 0;
static const uint OP_SUB = 1;
static const uint OP_MUL = 2;
static const uint OP_DIV = 3;
static const uint OP_MAX = 4;
static const uint OP_MIN = 5;

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

// Read a single bf16 lane (as f32) from a packed u32 buffer.
float read_lane(StructuredBuffer<uint> buf, uint elem_idx) {
    uint word = buf[elem_idx >> 1u];
    uint half_bits = (elem_idx & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
    return bf16_bits_to_f32(half_bits);
}

float compute(uint op, float x, float y) {
    switch (op) {
        case OP_ADD: return x + y;
        case OP_SUB: return x - y;
        case OP_MUL: return x * y;
        case OP_DIV: return x / y;
        case OP_MAX: return max(x, y);
        case OP_MIN: return min(x, y);
        default:     return x;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint base = pair_idx * 2u;
    if (base >= p.out_size) return;

    bool a_contig = (p.flags & 1u) != 0u;
    bool b_contig = (p.flags & 2u) != 0u;

    float x0, x1, y0, y1;

    if (a_contig && b_contig) {
        // Fast path — read both as packed u32 pairs.
        uint a_packed = a_buf[pair_idx];
        uint b_packed = b_buf[pair_idx];
        x0 = bf16_bits_to_f32(a_packed & 0xFFFFu);
        x1 = bf16_bits_to_f32(a_packed >> 16u);
        y0 = bf16_bits_to_f32(b_packed & 0xFFFFu);
        y1 = bf16_bits_to_f32(b_packed >> 16u);
    } else {
        // Strided path — read each lane individually.
        for (uint lane = 0u; lane < 2u; lane++) {
            uint idx = base + lane;
            uint c3 = idx % p.shape3;
            uint r2 = idx / p.shape3;
            uint c2 = r2 % p.shape2;
            uint r1 = r2 / p.shape2;
            uint c1 = r1 % p.shape1;
            uint c0 = r1 / p.shape1;

            uint a_off = c0 * p.a_s0 + c1 * p.a_s1 + c2 * p.a_s2 + c3 * p.a_s3;
            uint b_off = c0 * p.b_s0 + c1 * p.b_s1 + c2 * p.b_s2 + c3 * p.b_s3;

            float ax = a_contig ? bf16_bits_to_f32(
                (a_buf[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xFFFFu) : read_lane(a_buf, a_off);
            float by = b_contig ? bf16_bits_to_f32(
                (b_buf[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xFFFFu) : read_lane(b_buf, b_off);
            if (lane == 0u) { x0 = ax; y0 = by; }
            else            { x1 = ax; y1 = by; }
        }
    }

    float z0 = compute(p.op_id, x0, y0);
    float z1 = (base + 1u) < p.out_size ? compute(p.op_id, x1, y1) : z0;

    uint out_lo = f32_to_bf16_bits(z0);
    uint out_hi = f32_to_bf16_bits(z1);
    // For odd out_size the wrapper allocates a padded buffer so the
    // half-word write here is always safe; the trailing lane is ignored
    // downstream.
    out_buf[pair_idx] = out_lo | (out_hi << 16u);
}
