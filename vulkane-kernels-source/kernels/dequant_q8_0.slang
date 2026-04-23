// Dequantize GGML Q8_0 blocks to f32.
//
// Block layout (34 bytes each, 32 elements per block):
//   [0:2)   f16 scale d
//   [2:34)  32 bytes of signed i8 quants
//
// Formula: y[b*32 + k] = qs[k] * d
//
// Blocks are not 4-byte aligned (34 bytes each), so all reads use
// `load_u8` to extract bytes from 4-byte-aligned word loads.

struct Params {
    uint n_blocks;
    uint out_elements;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] ByteAddressBuffer       q8_0_blocks;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out_buf;
[[vk::binding(2, 0)]] ConstantBuffer<Params>  p;

static const uint BLCK_SIZE = 32;
static const uint BYTES_PER_BLOCK = 34;

uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = q8_0_blocks.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.out_elements) return;

    uint block_idx = tid / BLCK_SIZE;
    uint k = tid % BLCK_SIZE;

    uint block_byte_off = block_idx * BYTES_PER_BLOCK;

    // Read f16 scale.
    uint scale_lo = load_u8(block_byte_off);
    uint scale_hi = load_u8(block_byte_off + 1);
    uint scale_bits = scale_lo | (scale_hi << 8u);
    float d = f16tof32(scale_bits);

    // Read signed i8 at offset 2 + k.
    uint u = load_u8(block_byte_off + 2 + k);
    int x = int(u);
    if (x >= 128) { x -= 256; }

    out_buf[tid] = float(x) * d;
}
