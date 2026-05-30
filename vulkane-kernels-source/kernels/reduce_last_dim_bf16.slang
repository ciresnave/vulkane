// Per-row reduction along the last dimension — bf16.
//
// Input:  [n_rows, n_cols]  contiguous, row-major, bf16 packed
//                           two-per-u32 (lane 0 = low 16, lane 1 =
//                           high 16). `n_cols` MUST be even.
// Output: [n_rows]          one bf16 per row, packed two-per-u32 in
//                           an output buffer that the wrapper has
//                           ZERO-FILLED before dispatch. Workgroup
//                           `row` atomic-ORs its result into
//                           `output[row >> 1]` shifted by
//                           `(row & 1) * 16` bits.
//
// op_id: 0 = sum, 1 = max, 2 = min, 3 = mean.
//
// Mixed precision: storage bf16, accumulation + reduction in f32.
// Lane-pair scheme on input (each lane reads one u32 = two bf16
// values per iter); InterlockedOr on output to allow distinct
// workgroups to write distinct half-words of the same u32 without
// racing. The output buffer's zero initialization (done in the
// wrapper) is required for InterlockedOr to act as a half-word write
// — otherwise stale bits would corrupt the other lane.

struct Params {
    uint n_rows;
    uint n_cols;
    uint op_id;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;     // packed bf16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;    // zero-filled by wrapper
[[vk::binding(2, 0)]] ConstantBuffer<Params>   params;

groupshared float wg_data[32];

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

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

float identity_of(uint op) {
    if (op == 0u) return 0.0;
    if (op == 1u) return F32_NEG_INF;
    if (op == 2u) return F32_POS_INF;
    return 0.0;
}

float combine(uint op, float a, float b) {
    if (op == 0u) return a + b;
    if (op == 1u) return max(a, b);
    if (op == 2u) return min(a, b);
    return a + b;
}

float subgroup_reduce(uint op, float x) {
    if (op == 0u) return WaveActiveSum(x);
    if (op == 1u) return WaveActiveMax(x);
    if (op == 2u) return WaveActiveMin(x);
    return WaveActiveSum(x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint n_pairs = params.n_cols >> 1u;
    uint row_off_u32 = row * n_pairs;
    uint op = params.op_id;

    float acc = identity_of(op);
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        float v0 = bf16_bits_to_f32(packed & 0xFFFFu);
        float v1 = bf16_bits_to_f32(packed >> 16u);
        acc = combine(op, acc, v0);
        acc = combine(op, acc, v1);
    }

    float sg = subgroup_reduce(op, acc);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_data[sg_id] = sg;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_data[tid] : identity_of(op);
        float r = subgroup_reduce(op, v);
        if (WaveIsFirstLane()) {
            if (op == 3u) {
                r = r / float(params.n_cols);
            }
            // Pack into the correct half of output[row >> 1].
            uint bits = f32_to_bf16_bits(r);
            uint shifted = bits << ((row & 1u) * 16u);
            InterlockedOr(output[row >> 1u], shifted);
        }
    }
}
