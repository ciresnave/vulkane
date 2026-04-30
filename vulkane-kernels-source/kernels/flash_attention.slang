// Phase 8 Tier 2: FlashAttention v2 forward in Slang.
//
// Tile-based scaled-dot-product attention with online softmax. One
// workgroup processes one (batch, head, q_tile_start) triple — i.e.
// BR query rows for one (b, h) — and walks all K/V blocks.
//
// Layout (all NCHW-like, contiguous):
//   q: [B, Hq, Sq, D]
//   k: [B, Hkv, Sk, D]   Hkv ≤ Hq, GQA: head h_q reads kv-head h_q / (Hq/Hkv)
//   v: [B, Hkv, Sk, D]
//   o: [B, Hq, Sq, D]
//   alibi_slopes (optional via has_alibi flag): [Hq]
//
// Tile sizes are constants; correctness is independent of the choice.
// BR=BC=16 matches the Tier 1 reference; D≤128 is the maximum head_dim
// we lift into shared memory.
//
// Mask features (any combo): causal, sliding window left/right, ALiBi
// per-head slope, softcap (tanh logit cap).

struct Params {
    uint b;          // batch
    uint hq;         // query heads
    uint hkv;        // kv heads
    uint sq;         // query seq len
    uint sk;         // key/value seq len
    uint d;          // head dim
    uint groups;     // hq / hkv
    uint causal;     // 0 / 1
    uint window_left;     // sliding-window left (0 = unused)
    uint window_right;    // sliding-window right (0 = unused)
    uint has_window_left;
    uint has_window_right;
    uint has_alibi;
    uint has_softcap;
    float softmax_scale;
    float softcap;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   q_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float>   k_buf;
[[vk::binding(2, 0)]] StructuredBuffer<float>   v_buf;
[[vk::binding(3, 0)]] StructuredBuffer<float>   alibi_buf;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> o_buf;
[[vk::binding(5, 0)]] ConstantBuffer<Params>    p;

// One thread per (q_row in tile, output dim) pair = BR × D_max threads.
// 16 × 128 = 2048 threads total. We use 1D layout and split internally.
static const uint BR = 16;
static const uint BC = 16;
static const uint D_MAX = 128;

// Per-tile scratch.
groupshared float tile_q[BR * D_MAX];           // BR × D
groupshared float tile_k[BC * D_MAX];           // BC × D
groupshared float tile_v[BC * D_MAX];           // BC × D
groupshared float tile_s[BR * BC];              // BR × BC, raw scores then P
groupshared float row_m[BR];                    // running max
groupshared float row_l[BR];                    // running denominator
groupshared float tile_o[BR * D_MAX];           // BR × D, partial output

bool position_admissible(uint qi, uint kj) {
    if (p.causal != 0 && kj > qi) {
        return false;
    }
    if (p.has_window_left != 0) {
        // mask if kj + window_left < qi  (i.e. qi - window_left > kj)
        if (kj + p.window_left < qi) {
            return false;
        }
    }
    if (p.has_window_right != 0) {
        if (kj > qi + p.window_right) {
            return false;
        }
    }
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint bi = gid.x;
    uint hi = gid.y;
    uint q_tile_start = gid.z * BR;

    if (bi >= p.b || hi >= p.hq || q_tile_start >= p.sq) {
        return;
    }

    uint kv_h = (p.groups == 0u) ? hi : (hi / p.groups);

    uint d = p.d;
    uint q_row_stride = d;
    uint q_h_stride   = p.sq * d;
    uint q_b_stride   = p.hq * q_h_stride;
    uint k_row_stride = d;
    uint k_h_stride   = p.sk * d;
    uint k_b_stride   = p.hkv * k_h_stride;

    uint q_off = bi * q_b_stride + hi * q_h_stride;
    uint k_off = bi * k_b_stride + kv_h * k_h_stride;
    uint v_off = k_off;
    uint o_off = q_off;

    uint q_tile_end = min(q_tile_start + BR, p.sq);
    uint q_tile_rows = q_tile_end - q_tile_start;

    // 1. Load Q tile into shared memory (BR × D, zero-pad rows past end).
    //    Each thread handles a strided slice over BR*D.
    uint q_tile_size = q_tile_rows * d;
    for (uint i = tid; i < BR * D_MAX; i += 256u) {
        uint local_row = i / D_MAX;
        uint local_col = i % D_MAX;
        if (local_row < q_tile_rows && local_col < d) {
            uint qi_global = q_tile_start + local_row;
            tile_q[local_row * D_MAX + local_col] =
                q_buf[q_off + qi_global * q_row_stride + local_col];
        } else {
            tile_q[local_row * D_MAX + local_col] = 0.0;
        }
    }

    // 2. Initialize per-Q-row state (m=-inf, l=0, O=0).
    if (tid < BR) {
        row_m[tid] = -3.402823e+38;
        row_l[tid] = 0.0;
    }
    for (uint i = tid; i < BR * D_MAX; i += 256u) {
        tile_o[i] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    float alibi_slope = (p.has_alibi != 0) ? alibi_buf[hi] : 0.0;

    // 3. Walk K/V blocks.
    for (uint k_tile_start = 0u; k_tile_start < p.sk; k_tile_start += BC) {
        uint k_tile_end = min(k_tile_start + BC, p.sk);
        uint k_tile_cols = k_tile_end - k_tile_start;

        // 3a. Load K block (BC × D, padded).
        for (uint i = tid; i < BC * D_MAX; i += 256u) {
            uint local_row = i / D_MAX;
            uint local_col = i % D_MAX;
            if (local_row < k_tile_cols && local_col < d) {
                uint kj_global = k_tile_start + local_row;
                tile_k[local_row * D_MAX + local_col] =
                    k_buf[k_off + kj_global * k_row_stride + local_col];
                tile_v[local_row * D_MAX + local_col] =
                    v_buf[v_off + kj_global * k_row_stride + local_col];
            } else {
                tile_k[local_row * D_MAX + local_col] = 0.0;
                tile_v[local_row * D_MAX + local_col] = 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3b. Compute S_ij = Q_i · K_j^T · scale, plus mask + softcap + alibi.
        //     One thread per (qi_local, kj_local) cell: 16 × 16 = 256 threads.
        if (tid < BR * BC) {
            uint qi_local = tid / BC;
            uint kj_local = tid % BC;
            float s = -3.402823e+38;
            if (qi_local < q_tile_rows && kj_local < k_tile_cols) {
                uint qi = q_tile_start + qi_local;
                uint kj = k_tile_start + kj_local;
                if (position_admissible(qi, kj)) {
                    float acc = 0.0;
                    for (uint dd = 0u; dd < d; dd++) {
                        acc += tile_q[qi_local * D_MAX + dd]
                             * tile_k[kj_local * D_MAX + dd];
                    }
                    s = acc * p.softmax_scale;
                    if (p.has_softcap != 0) {
                        float t = tanh(s / p.softcap);
                        s = t * p.softcap;
                    }
                    if (p.has_alibi != 0) {
                        s += alibi_slope * (float(kj) - float(qi));
                    }
                }
            }
            tile_s[qi_local * BC + kj_local] = s;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3c. Online softmax update per Q row. One thread per Q row.
        if (tid < BR && tid < q_tile_rows) {
            uint qi_local = tid;
            // Find tile_max for this row.
            float tile_max = -3.402823e+38;
            for (uint kj = 0u; kj < k_tile_cols; kj++) {
                float s = tile_s[qi_local * BC + kj];
                if (s > tile_max) tile_max = s;
            }

            float m_old = row_m[qi_local];
            float m_new = max(m_old, tile_max);
            // Skip if no admissible keys yet.
            if (isinf(m_new)) {
                continue;
            }
            float scale_old = isinf(m_old) ? 0.0 : exp(m_old - m_new);

            // Compute P_ij = exp(S_ij - m_new), sum, accumulate into O.
            float row_sum = 0.0;
            for (uint kj = 0u; kj < k_tile_cols; kj++) {
                float s = tile_s[qi_local * BC + kj];
                float pij = isinf(s) ? 0.0 : exp(s - m_new);
                tile_s[qi_local * BC + kj] = pij;
                row_sum += pij;
            }
            row_l[qi_local] = scale_old * row_l[qi_local] + row_sum;

            // O_i = scale_old * O_i + Σ_j P_ij · V_j   (over this K block).
            for (uint dd = 0u; dd < d; dd++) {
                float acc = scale_old * tile_o[qi_local * D_MAX + dd];
                for (uint kj = 0u; kj < k_tile_cols; kj++) {
                    float pij = tile_s[qi_local * BC + kj];
                    acc += pij * tile_v[kj * D_MAX + dd];
                }
                tile_o[qi_local * D_MAX + dd] = acc;
            }
            row_m[qi_local] = m_new;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Finalize: O_i / l_i and write to global memory.
    if (tid < BR && tid < q_tile_rows) {
        uint qi_local = tid;
        float l = row_l[qi_local];
        float inv_l = (l > 0.0) ? (1.0 / l) : 0.0;
        uint qi_global = q_tile_start + qi_local;
        for (uint dd = 0u; dd < d; dd++) {
            o_buf[o_off + qi_global * q_row_stride + dd] =
                tile_o[qi_local * D_MAX + dd] * inv_l;
        }
    }
}
