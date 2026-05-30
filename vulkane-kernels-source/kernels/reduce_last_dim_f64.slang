// Per-row reduction along the last dimension — f64.
//
// Input:  [n_rows, n_cols]  contiguous, row-major, double.
// Output: [n_rows]          one double per row.
//
// op_id: 0 = sum, 1 = max, 2 = min, 3 = mean.
//
// Native f64 end-to-end. Requires shaderFloat64 +
// GroupNonUniformArithmetic. Uses GLSL.std.450 for max/min (NOT
// OpenCL.std).
//
// Binding layout matches the f32 version (16-byte Params).

struct Params {
    uint n_rows;
    uint n_cols;
    uint op_id;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     params;

groupshared double wg_data[32];

static const double F64_NEG_INF = -1.7976931348623157e+308;
static const double F64_POS_INF =  1.7976931348623157e+308;

double identity_of(uint op) {
    if (op == 0u) return 0.0;
    if (op == 1u) return F64_NEG_INF;
    if (op == 2u) return F64_POS_INF;
    return 0.0;
}

double combine(uint op, double a, double b) {
    if (op == 0u) return a + b;
    if (op == 1u) return max(a, b);
    if (op == 2u) return min(a, b);
    return a + b;
}

double subgroup_reduce(uint op, double x) {
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
    uint row_off = row * params.n_cols;
    uint op = params.op_id;

    double acc = identity_of(op);
    for (uint col = tid; col < params.n_cols; col += 256u) {
        acc = combine(op, acc, input[row_off + col]);
    }

    double sg = subgroup_reduce(op, acc);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_data[sg_id] = sg;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        double v = (tid < n_sg) ? wg_data[tid] : identity_of(op);
        double r = subgroup_reduce(op, v);
        if (WaveIsFirstLane()) {
            if (op == 3u) {
                r = r / double(params.n_cols);
            }
            output[row] = r;
        }
    }
}
