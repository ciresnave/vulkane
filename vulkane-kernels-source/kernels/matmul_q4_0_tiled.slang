// Fused Q4_0 × F32 matmul for M > 1 (prefill path).
//
// Computes C[m, n] = sum_k A[m, k] * dequant(W[n, k]) for
// m in [0, M), n in [0, N). W is stored as Q4_0 blocks in [N, K/32]
// layout (same as qmatvec_q4_0).
//
// Structure is a direct extension of qmatvec_q4_0: one workgroup per
// (m_tile, n_col) pair, with TM m-rows per tile. Each thread keeps
// TM accumulators so one weight read feeds TM A rows — the payoff
// that distinguishes this from the row-loop (M separate gemv dispatches).
//
// TM = 8. Threads per workgroup = 128. For each K stride: load scale,
// dequant W element once, MAC into TM register partials. After the K
// loop, one `WaveActiveSum` per m row, cross-subgroup collapse via
// groupshared, one write to C[m_base + m_off, n] per m.
//
// At M=1 the overhead per workgroup is higher than qmatvec (extra loop
// unrolls over dead TM rows), so VulkanBackend::matmul_q4_0 dispatches
// qmatvec_q4_0 for M=1 and this kernel for M>1.

struct Params {
    uint m;              // M rows (batch * seq)
    uint n;              // N cols
    uint k;              // contracted dim
    uint blocks_per_row; // K / 32
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   a_f32;    // [M, K]
[[vk::binding(1, 0)]] ByteAddressBuffer         w_q4_0;   // [N, K/32] blocks
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_f32;    // [M, N]
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

static const uint BYTES_PER_BLOCK = 18;
static const uint BLCK_SIZE = 32;
static const uint TM = 8;     // M rows per workgroup tile
static const uint WG_SIZE = 128;

// Subgroup partials, one column per m row. 16 subgroups max at WG=128.
groupshared float sg_partials[TM][16];

uint load_u8(uint byte_off)
{
    uint word_off = byte_off & ~3u;
    uint byte_in_word = byte_off & 3u;
    uint word = w_q4_0.Load(word_off);
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

float load_f16_scale(uint byte_off)
{
    uint lo = load_u8(byte_off);
    uint hi = load_u8(byte_off + 1);
    return f16tof32(lo | (hi << 8u));
}

float dequant_elem(uint block_byte_off, uint elem, float d)
{
    uint nibble_idx;
    bool is_high;
    if (elem < 16) { nibble_idx = elem; is_high = false; }
    else           { nibble_idx = elem - 16; is_high = true; }
    uint packed = load_u8(block_byte_off + 2 + nibble_idx);
    uint raw = is_high ? (packed >> 4u) : (packed & 0x0Fu);
    int x = int(raw) - 8;
    return float(x) * d;
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 gid  : SV_GroupID,
    uint3 gtid : SV_GroupThreadID)
{
    uint n = gid.x;
    uint m_base = gid.y * TM;
    if (n >= p.n) return;

    uint tid = gtid.x;
    uint row_byte_off = n * p.blocks_per_row * BYTES_PER_BLOCK;

    // Per-thread register partials, one per m row in the tile.
    float partial[TM];
    for (uint i = 0; i < TM; i++) partial[i] = 0.0f;

    // Thread-stride over K.
    uint k_idx = tid;
    while (k_idx < p.k) {
        uint block_idx = k_idx / BLCK_SIZE;
        uint elem_in_block = k_idx % BLCK_SIZE;
        uint block_byte_off = row_byte_off + block_idx * BYTES_PER_BLOCK;
        float d = load_f16_scale(block_byte_off);
        float w_elem = dequant_elem(block_byte_off, elem_in_block, d);

        // Fan out the single weight load across TM A rows.
        [unroll]
        for (uint m_off = 0; m_off < TM; m_off++) {
            uint m = m_base + m_off;
            if (m < p.m) {
                partial[m_off] += a_f32[m * p.k + k_idx] * w_elem;
            }
        }
        k_idx += WG_SIZE;
    }

    // Subgroup + cross-subgroup reduction, one m row at a time.
    uint sg_id = tid / WaveGetLaneCount();
    uint n_subgroups = WG_SIZE / WaveGetLaneCount();

    [unroll]
    for (uint m_off = 0; m_off < TM; m_off++) {
        float sg_sum = WaveActiveSum(partial[m_off]);
        if (WaveIsFirstLane()) {
            sg_partials[m_off][sg_id] = sg_sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Cross-subgroup: subgroup 0 reduces the per-SG partials for each m.
    if (sg_id == 0) {
        [unroll]
        for (uint m_off = 0; m_off < TM; m_off++) {
            float v = (tid < n_subgroups) ? sg_partials[m_off][tid] : 0.0f;
            float total = WaveActiveSum(v);
            if (WaveIsFirstLane()) {
                uint m = m_base + m_off;
                if (m < p.m) {
                    c_f32[m * p.n + n] = total;
                }
            }
        }
    }
}
