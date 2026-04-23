// Fused Q4_0 × F32 gemv.
//
// Computes output[n] = sum_k A[k] * dequant(W_block[n, k/32][k%32]) for
// n in [0, N). Weight matrix W is stored as Q4_0 blocks in [N, K/32]
// layout (N rows of K/32 blocks of 18 bytes each = 32 elements each).
//
// This is the decode-phase (M=1) hot path for inference on quantized
// LLMs — 97%+ of matmul dispatches during decode are gemv, so a fused
// dequant-in-kernel here lets the Q4_0 blocks stay resident at ~4×
// compressed size vs F32 (2× vs BF16), unlocking 7B models on 8 GB
// cards.
//
// Block layout (18 bytes per block, 32 elements):
//   [0:2)   f16 scale d
//   [2:18)  16 bytes of packed u4 quants (low nibble of byte j → element j,
//            high nibble of byte j → element j+16)
// Formula: elem = (nibble - 8) * d.
//
// One workgroup per output column (n). 128 threads per workgroup.
// Each thread accumulates a partial sum over its K slice, then
// subgroup + shared-memory reduction produces the final output[n].

struct Params {
    uint n;              // output length (N)
    uint k;              // contracted dim length (K, must be multiple of 32)
    uint blocks_per_row; // K / 32 — number of Q4_0 blocks per output row
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   a_f32;      // [K]
[[vk::binding(1, 0)]] ByteAddressBuffer         w_q4_0;     // [N, K/32] blocks (raw bytes)
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_f32;      // [N]
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

static const uint BYTES_PER_BLOCK = 18;
static const uint BLCK_SIZE = 32;

// Shared memory for cross-subgroup reduction (up to 32 subgroups at
// workgroup size 128, way more than we need — NVIDIA is 4 subgroups
// of 32 lanes at this size).
groupshared float subgroup_partials[16];

// Load one byte at an arbitrary byte offset.
uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = w_q4_0.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

// Load an f16 scale at an arbitrary byte offset (2 bytes).
float load_f16_scale(uint byte_off)
{
    uint lo = load_u8(byte_off);
    uint hi = load_u8(byte_off + 1);
    return f16tof32(lo | (hi << 8u));
}

// Dequantize the `elem`th element (0..32) of block `block_byte_off`
// given the already-loaded scale `d`.
float dequant_elem(uint block_byte_off, uint elem, float d)
{
    uint nibble_idx;
    bool is_high;
    if (elem < 16) {
        nibble_idx = elem;
        is_high = false;
    } else {
        nibble_idx = elem - 16;
        is_high = true;
    }
    uint packed = load_u8(block_byte_off + 2 + nibble_idx);
    uint raw = is_high ? (packed >> 4u) : (packed & 0x0Fu);
    int x = int(raw) - 8;
    return float(x) * d;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(
    uint3 gid  : SV_GroupID,
    uint3 gtid : SV_GroupThreadID)
{
    uint n = gid.x;
    if (n >= p.n) return;

    uint tid = gtid.x;

    // Each output row n starts at byte offset:
    //   n * blocks_per_row * BYTES_PER_BLOCK
    uint row_byte_off = n * p.blocks_per_row * BYTES_PER_BLOCK;

    float partial = 0.0;

    // Thread-stride over K: 128 threads each step 128 in k.
    // Within each 32-element block, ~4 threads visit the same block —
    // good locality for the scale load.
    uint k_idx = tid;
    while (k_idx < p.k) {
        uint block_idx = k_idx / BLCK_SIZE;
        uint elem_in_block = k_idx % BLCK_SIZE;
        uint block_byte_off = row_byte_off + block_idx * BYTES_PER_BLOCK;
        float d = load_f16_scale(block_byte_off);
        float w_elem = dequant_elem(block_byte_off, elem_in_block, d);
        partial += a_f32[k_idx] * w_elem;
        k_idx += 128u;
    }

    // Subgroup reduction + cross-subgroup reduction through shared memory.
    float sg_sum = WaveActiveSum(partial);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        subgroup_partials[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_subgroups = 128u / WaveGetLaneCount();
    if (sg_id == 0) {
        float v = (tid < n_subgroups) ? subgroup_partials[tid] : 0.0;
        float total = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            c_f32[n] = total;
        }
    }
}
