// Fused rotary position embedding — bf16.
//
// Storage: bf16 packed two-per-u32 (lane 0 = low 16 bits). x, cos,
// sin, out all use the same packing.
//
// Pair-thread layout: each thread T handles a pair of i values
// (i, i+1). For positions (i, i+1) and (i+h, i+h+1), each pair forms
// one u32 (adjacent bf16 lanes), so each thread writes EXACTLY two
// u32 words — one for the low half (positions i, i+1) and one for
// the high half (positions i+h, i+h+1). No inter-thread write race,
// no atomics, no zero-fill needed.
//
// PRECONDITION: `head_dim % 4 == 0` (so h = head_dim/2 is even, which
// means our (2T, 2T+1) base lands on a u32 boundary for both the
// low-half pair and the high-half pair). LLM head_dims are typically
// 64/128/256, all divisible by 4 — the wrapper enforces.
//
// Mixed precision: math in f32 (multiplies + adds compound rounding
// past 1 ULP of bf16 quickly); bf16 round at the store.

struct Params {
    uint outer;
    uint seq;
    uint head_dim;
    uint pairs_total;    // outer * seq * (head_dim / 4)
    uint x_s0;
    uint x_s1;
    uint x_s_seq;
    uint x_s_hd;
    uint x_outer1;
    uint x_contiguous;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   x;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   cos_tab;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   sin_tab;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> out;
[[vk::binding(4, 0)]] ConstantBuffer<Params>   p;

float bf16_bits_to_f32(uint b) { return asfloat(b << 16u); }

uint f32_to_bf16_bits(float v) {
    uint bits = asuint(v);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

// Read a single bf16 lane (as f32) from a packed u32 buffer at element
// index `elem_idx`. Lane 0 = low 16, lane 1 = high 16.
float read_lane(StructuredBuffer<uint> buf, uint elem_idx) {
    uint word = buf[elem_idx >> 1u];
    uint half_bits = (elem_idx & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
    return bf16_bits_to_f32(half_bits);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_tid = gid.x;
    if (pair_tid >= p.pairs_total) return;

    uint h = p.head_dim / 2;
    uint h_pairs = h >> 1u;           // pairs per "low half" of one row
    uint i_pair = pair_tid % h_pairs;
    uint os = pair_tid / h_pairs;
    uint s  = os % p.seq;
    uint o  = os / p.seq;

    uint i = i_pair * 2u;             // base i (even, guaranteed by head_dim%4==0)
    uint row_elem   = (o * p.seq + s) * p.head_dim;     // bf16 element offset
    uint row_u32    = row_elem >> 1u;                   // u32 word offset
    uint table_elem = s * p.head_dim;
    uint table_u32  = table_elem >> 1u;

    // u32 word indices we'll touch:
    //   lo  word: positions i, i+1 → u32 at row_u32 + i_pair
    //   hi  word: positions i+h, i+1+h → u32 at row_u32 + (i+h)/2 = row_u32 + i_pair + h/2
    uint lo_word_idx_off = i_pair;
    uint hi_word_idx_off = i_pair + h_pairs;

    // Read x: contiguous fast path reads u32 directly; strided path
    // reads each lane via read_lane(x, elem_idx) using bf16 element
    // strides (x_s_hd is in bf16-element units).
    float x_lo_0, x_lo_1, x_hi_0, x_hi_1;
    if (p.x_contiguous == 1) {
        uint x_lo_word = x[row_u32 + lo_word_idx_off];
        uint x_hi_word = x[row_u32 + hi_word_idx_off];
        x_lo_0 = bf16_bits_to_f32(x_lo_word & 0xFFFFu);
        x_lo_1 = bf16_bits_to_f32(x_lo_word >> 16u);
        x_hi_0 = bf16_bits_to_f32(x_hi_word & 0xFFFFu);
        x_hi_1 = bf16_bits_to_f32(x_hi_word >> 16u);
    } else {
        uint o1 = o % p.x_outer1;
        uint o0 = o / p.x_outer1;
        uint x_base = o0 * p.x_s0 + o1 * p.x_s1 + s * p.x_s_seq;
        // x_s_hd is in bf16-element units.
        x_lo_0 = read_lane(x, x_base +  i        * p.x_s_hd);
        x_lo_1 = read_lane(x, x_base + (i + 1u)  * p.x_s_hd);
        x_hi_0 = read_lane(x, x_base + (i + h)        * p.x_s_hd);
        x_hi_1 = read_lane(x, x_base + (i + 1u + h)   * p.x_s_hd);
    }

    // cos/sin tables are always contiguous.
    uint c_lo_word = cos_tab[table_u32 + lo_word_idx_off];
    uint s_lo_word = sin_tab[table_u32 + lo_word_idx_off];
    uint c_hi_word = cos_tab[table_u32 + hi_word_idx_off];
    uint s_hi_word = sin_tab[table_u32 + hi_word_idx_off];

    float c0_0 = bf16_bits_to_f32(c_lo_word & 0xFFFFu);
    float c0_1 = bf16_bits_to_f32(c_lo_word >> 16u);
    float s0_0 = bf16_bits_to_f32(s_lo_word & 0xFFFFu);
    float s0_1 = bf16_bits_to_f32(s_lo_word >> 16u);
    float c1_0 = bf16_bits_to_f32(c_hi_word & 0xFFFFu);
    float c1_1 = bf16_bits_to_f32(c_hi_word >> 16u);
    float s1_0 = bf16_bits_to_f32(s_hi_word & 0xFFFFu);
    float s1_1 = bf16_bits_to_f32(s_hi_word >> 16u);

    // Apply the rotation per the rope formula:
    //   out[i]     = x[i]   * c0 - x[i+h] * s0
    //   out[i+h]   = x[i+h] * c1 + x[i]   * s1
    float out_lo_0 = x_lo_0 * c0_0 - x_hi_0 * s0_0;
    float out_lo_1 = x_lo_1 * c0_1 - x_hi_1 * s0_1;
    float out_hi_0 = x_hi_0 * c1_0 + x_lo_0 * s1_0;
    float out_hi_1 = x_hi_1 * c1_1 + x_lo_1 * s1_1;

    uint out_lo_packed = f32_to_bf16_bits(out_lo_0) | (f32_to_bf16_bits(out_lo_1) << 16u);
    uint out_hi_packed = f32_to_bf16_bits(out_hi_0) | (f32_to_bf16_bits(out_hi_1) << 16u);
    out[row_u32 + lo_word_idx_off] = out_lo_packed;
    out[row_u32 + hi_word_idx_off] = out_hi_packed;
}
