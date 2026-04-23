// Dequantize GGML Q4_0 blocks to f32.
//
// Block layout (18 bytes each, 32 elements per block):
//   [0:2)   f16 scale d
//   [2:18)  16 bytes of packed u4 quants (low nibble of byte k → element k,
//            high nibble of byte k → element k+16)
//
// Formula: y[b*32 + k]    = (qs_lo - 8) * d
//          y[b*32 + k+16] = (qs_hi - 8) * d
//
// One thread per output pair (k, k+16). Blocks are not 4-byte aligned
// (18 bytes each), so all reads use `load_u8` which fetches the
// containing 4-byte word and extracts the requested byte.

struct Params {
    uint n_blocks;
    uint out_elements;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] ByteAddressBuffer       q4_0_blocks;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out_buf;
[[vk::binding(2, 0)]] ConstantBuffer<Params>  p;

static const uint BLCK_SIZE = 32;
static const uint BYTES_PER_BLOCK = 18;

// Load one byte at an arbitrary byte offset.
uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = q4_0_blocks.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    uint total_pairs = p.n_blocks * (BLCK_SIZE / 2);
    if (tid >= total_pairs) return;

    uint block_idx = tid / (BLCK_SIZE / 2);
    uint k = tid % (BLCK_SIZE / 2);

    uint block_byte_off = block_idx * BYTES_PER_BLOCK;

    // Read f16 scale (bytes [0:2) of block).
    uint scale_lo = load_u8(block_byte_off);
    uint scale_hi = load_u8(block_byte_off + 1);
    uint scale_bits = scale_lo | (scale_hi << 8u);
    float d = f16tof32(scale_bits);

    // Read packed nibble byte at offset 2 + k.
    uint packed = load_u8(block_byte_off + 2 + k);
    int x0 = int(packed & 0x0Fu) - 8;
    int x1 = int((packed >> 4u) & 0x0Fu) - 8;

    uint out_off = block_idx * BLCK_SIZE + k;
    out_buf[out_off] = float(x0) * d;
    out_buf[out_off + BLCK_SIZE / 2u] = float(x1) * d;
}
