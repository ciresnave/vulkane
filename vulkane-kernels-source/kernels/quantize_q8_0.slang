// Quantize an F32 tensor to GGML Q8_0 blocks (byte stream).
//
// Q8_0 block (34 bytes, 32 elements):
//   [0:2)   f16 scale d = max(|x|) / 127
//   [2:34)  32 bytes of signed i8 quants: round(x / d)
//
// One thread per block (serial over 32 elements). Quantize + write
// happens in one thread, avoiding byte-write races within a 4-byte
// word. Launched as a 1D dispatch with one thread per block and
// workgroup size 64; with one workgroup per 64 blocks (2048 elements).
//
// Used for KV-cache quantization — happens once per decode step over a
// small F32 buffer (fresh K/V), so the non-cooperative write pattern
// is fine compared to the win from 4× smaller cache storage.

struct Params {
    uint n_elements;      // total F32 input elements (multiple of 32)
    uint n_blocks;        // n_elements / 32
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   src_f32;
[[vk::binding(1, 0)]] RWByteAddressBuffer       out_blocks;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

static const uint BLCK_SIZE = 32;
static const uint BYTES_PER_BLOCK = 34;

// Write a single byte into `out_blocks` at `byte_off` via RMW on the
// containing aligned u32 word. Safe when only ONE thread writes any
// given byte-offset (which is the case here: one thread per block,
// and blocks occupy disjoint byte ranges).
void write_byte(uint byte_off, uint byte_val)
{
    uint word_off = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint mask = 0xFFu << shift;
    uint new_bits = (byte_val & 0xFFu) << shift;
    // Two different blocks may share the SAME 4-byte word at the
    // boundary (block 0 ends at byte 33, block 1 starts at byte 34 →
    // bytes 32 & 33 of block 0 are in the word at offset 32; byte 0
    // of block 1 starts the next word at offset 36). So the word at
    // offset 32 contains only bytes from block 0.
    //
    // BUT: every pair of adjacent blocks, the LAST byte of block N
    // (offset N*34 + 33) and the FIRST byte of block N+1 (offset
    // (N+1)*34 = N*34 + 34) may share a word when (N*34 + 33) >>2 ==
    // ((N+1)*34) >> 2 — i.e., they differ by 1 which is always the
    // case. Check N=0: byte 33 is in word 32 (word_off 32). Byte 34
    // is in word 32 too! So byte 33 (last of block 0) and byte 34
    // (first of block 1) SHARE a word.
    //
    // Workaround: use atomic InterlockedOr after clearing the
    // target bits. We do the AND+OR as two atomics; between them,
    // another thread may write a DIFFERENT byte in the same word
    // (which we don't want to clobber). To avoid that, we use a
    // single InterlockedXor that applies `(old_byte ^ new_byte) <<
    // shift`:
    //
    //   new_word = old_word ^ ((old_byte ^ new_byte) << shift)
    //            = (old_word & ~mask) | (new_byte << shift)
    //
    // That requires reading the old byte first. Do that via a
    // non-atomic Load (racing with concurrent writes is benign
    // because we'll XOR away the difference).
    uint old_word = out_blocks.Load(word_off);
    uint old_byte = (old_word >> shift) & 0xFFu;
    uint xor_bits = ((old_byte ^ (byte_val & 0xFFu)) & 0xFFu) << shift;
    uint _out;
    out_blocks.InterlockedXor(word_off, xor_bits, _out);
    // Silence unused warnings.
    mask = mask; new_bits = new_bits;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint block_idx = gid.x;
    if (block_idx >= p.n_blocks) return;

    uint block_byte_off = block_idx * BYTES_PER_BLOCK;
    uint elem_base = block_idx * BLCK_SIZE;

    // Pass 1: find absmax over the 32 elements.
    float block_max = 0.0;
    for (uint k = 0; k < BLCK_SIZE; k++) {
        float ax = abs(src_f32[elem_base + k]);
        if (ax > block_max) block_max = ax;
    }
    // Scale chosen so ±max maps to ±127.
    float d = (block_max > 0.0) ? (block_max / 127.0) : 0.0;
    float inv_d = (d > 0.0) ? (1.0 / d) : 0.0;

    // Write scale (f16) at byte offset 0-1.
    uint scale_bits = f32tof16(d);
    write_byte(block_byte_off, scale_bits & 0xFFu);
    write_byte(block_byte_off + 1, (scale_bits >> 8) & 0xFFu);

    // Write 32 quant bytes at byte offset 2..34.
    for (uint k = 0; k < BLCK_SIZE; k++) {
        float x = src_f32[elem_base + k];
        int q = int(round(x * inv_d));
        q = clamp(q, -128, 127);
        write_byte(block_byte_off + 2 + k, uint(q & 0xFF));
    }
}
