// GGML Q4_K_M super-block dequantization to f32.
//
// Block layout (144 bytes per 256-element super-block):
//   [0..2)    f16 d      (super-block scale)
//   [2..4)    f16 dmin   (super-block min scale)
//   [4..16)   u8[12]     (8 sub-block scales + 8 sub-block mins, 6-bit packed)
//   [16..144) u8[128]    (4-bit quants for 256 elements)
//
// Decoding algorithm (matches llama.cpp k_quants.c):
//   for j = 0..QK_K step 64:
//     get_scale_min_k4(is+0) -> (sc, m);  d1 = d*sc; m1 = dmin*m
//     get_scale_min_k4(is+1) -> (sc, m);  d2 = d*sc; m2 = dmin*m
//     for l = 0..32:
//       y[j + l]      = d1 * (q[j/2 + l] & 0xF) - m1
//       y[j + 32 + l] = d2 * (q[j/2 + l] >> 4)  - m2
//     is += 2
//
// One workgroup per super-block; 32 threads per workgroup. Each
// thread writes 8 output elements (4 low-nibble + 4 high-nibble,
// matching llama.cpp's mapping). 32 * 8 = 256 = QK_K.

struct Params {
    uint n_blocks;     // number of Q4_K super-blocks
    uint out_elements; // n_blocks * 256
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] ByteAddressBuffer         w_q4_km;   // block byte stream
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out_f32;   // dequantized output
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

static const uint BYTES_PER_BLOCK = 144;
static const uint QK_K = 256;

uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = w_q4_km.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

float load_f16_at(uint byte_off)
{
    uint lo = load_u8(byte_off);
    uint hi = load_u8(byte_off + 1);
    return f16tof32(lo | (hi << 8u));
}

// Matches llama.cpp get_scale_min_k4. `j` is the sub-block index
// (0..8). `scales_off` is the byte offset to the 12-byte packed
// scale array inside the super-block.
// Returns sub-block scale in x, sub-block min in y.
uint2 get_scale_min_k4(uint j, uint scales_off)
{
    uint sc;
    uint m;
    if (j < 4u) {
        sc = load_u8(scales_off + j)       & 63u;
        m  = load_u8(scales_off + j + 4u)  & 63u;
    } else {
        uint q_j4  = load_u8(scales_off + j + 4u);
        uint q_jm4 = load_u8(scales_off + j - 4u);
        uint q_j   = load_u8(scales_off + j);
        sc = (q_j4 & 0xFu) | ((q_jm4 >> 6u) << 4u);
        m  = (q_j4 >> 4u)  | ((q_j   >> 6u) << 4u);
    }
    uint2 r;
    r.x = sc;
    r.y = m;
    return r;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(
    uint3 gid  : SV_GroupID,
    uint3 gtid : SV_GroupThreadID)
{
    uint block_idx = gid.x;
    if (block_idx >= p.n_blocks) return;

    uint tid = gtid.x;  // 0..32
    uint block_byte_off = block_idx * BYTES_PER_BLOCK;

    float d    = load_f16_at(block_byte_off);
    float dmin = load_f16_at(block_byte_off + 2u);

    uint scales_off = block_byte_off + 4u;
    uint qs_off     = block_byte_off + 16u;

    // Map tid → (il, ir) per llama.cpp CUDA Q4_K dequant:
    //   il = tid / 8   (0..4)  — 128-element stride (j-pair index)
    //   ir = tid & 7   (0..8)  — 4-element offset within pair
    //   is = 2*il              — sub-block index (8 sub-blocks)
    //   out_base = 64*il + 4*ir
    //   writes y[out_base + l] = d1 * (q[32*il + 4*ir + l] & 0xF) - m1
    //          y[out_base + 32 + l] = d2 * (q[...] >> 4) - m2
    // 4 (il values) × 8 (ir values) × 8 (elems per thread) = 256 = QK_K.
    uint il = tid >> 3u;       // 0..4
    uint ir = tid & 7u;        // 0..8
    uint is = 2u * il;
    uint2 sm0 = get_scale_min_k4(is,      scales_off);
    uint2 sm1 = get_scale_min_k4(is + 1u, scales_off);
    float d1 = d * float(sm0.x); float m1 = dmin * float(sm0.y);
    float d2 = d * float(sm1.x); float m2 = dmin * float(sm1.y);

    uint q_start = qs_off + 32u * il + 4u * ir;
    uint y_base  = block_idx * QK_K + 64u * il + 4u * ir;

    [unroll]
    for (uint l = 0u; l < 4u; l++) {
        uint q = load_u8(q_start + l);
        out_f32[y_base + l]       = d1 * float(q & 0xFu) - m1;
        out_f32[y_base + 32u + l] = d2 * float(q >> 4u)  - m2;
    }
}
