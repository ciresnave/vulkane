// Parallel reduction over all elements — bf16. One workgroup, tree
// reduction.
//
// Storage: bf16 packed two-per-u32 (lane 0 = low 16 bits). `n` is the
// bf16 element count; n MUST be even (n / 2 = u32 pair count).
//
// Mixed precision: input/output bf16, accumulation + tree reduction
// in f32.
//
// Output is a single bf16 written to the low 16 bits of `output[0]`
// (the high 16 bits are deliberately written as 0, not as a "ghost"
// bf16 — the downstream reader only sees the first 2 bytes).
// Single-writer; no atomic or zero-fill needed.

struct Params {
    uint n;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

static const uint OP_SUM  = 0;
static const uint OP_MAX  = 1;
static const uint OP_MIN  = 2;
static const uint OP_MEAN = 3;

groupshared float wg_data[256];

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

float identity(uint op) {
    switch (op) {
        case OP_SUM:  return 0.0f;
        case OP_MEAN: return 0.0f;
        case OP_MAX:  return -3.402823e+38f;
        case OP_MIN:  return  3.402823e+38f;
        default:      return 0.0f;
    }
}

float combine(uint op, float a, float b) {
    switch (op) {
        case OP_SUM:  return a + b;
        case OP_MEAN: return a + b;
        case OP_MAX:  return max(a, b);
        case OP_MIN:  return min(a, b);
        default:      return a;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 lid : SV_GroupThreadID)
{
    uint tid = lid.x;
    uint op = p.op_id;
    uint n_pairs = p.n >> 1u;

    // Each thread processes one u32 pair (two bf16 lanes) per iter.
    float acc = identity(op);
    uint i = tid;
    while (i < n_pairs) {
        uint packed = input[i];
        float v0 = bf16_bits_to_f32(packed & 0xFFFFu);
        float v1 = bf16_bits_to_f32(packed >> 16u);
        acc = combine(op, acc, v0);
        acc = combine(op, acc, v1);
        i += 256u;
    }
    wg_data[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    uint stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_data[tid] = combine(op, wg_data[tid], wg_data[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        float r = wg_data[0];
        if (op == OP_MEAN) {
            r = r / float(p.n);
        }
        // Single bf16 result; explicit 0 in upper half so downstream
        // readers that look at the full u32 see a well-defined value.
        output[0] = f32_to_bf16_bits(r);
    }
}
