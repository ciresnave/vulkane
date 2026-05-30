// Parallel reduction over all elements — f64. One workgroup, tree
// reduction. Native f64 end-to-end. Requires shaderFloat64.

struct Params {
    uint n;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     p;

static const uint OP_SUM  = 0;
static const uint OP_MAX  = 1;
static const uint OP_MIN  = 2;
static const uint OP_MEAN = 3;

groupshared double wg_data[256];

double identity(uint op) {
    switch (op) {
        case OP_SUM:  return 0.0;
        case OP_MEAN: return 0.0;
        case OP_MAX:  return -1.7976931348623157e+308;
        case OP_MIN:  return  1.7976931348623157e+308;
        default:      return 0.0;
    }
}

double combine(uint op, double a, double b) {
    switch (op) {
        case OP_SUM:  return a + b;
        case OP_MEAN: return a + b;
        case OP_MAX:  return max(a, b);
        case OP_MIN:  return min(a, b);
        default:      return a;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 lid : SV_GroupThreadID)
{
    uint tid = lid.x;
    uint op = p.op_id;

    double acc = identity(op);
    uint i = tid;
    while (i < p.n) {
        acc = combine(op, acc, input[i]);
        i += 256u;
    }
    wg_data[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    uint stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_data[tid] = combine(op, wg_data[tid], wg_data[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        double r = wg_data[0];
        if (op == OP_MEAN) {
            r = r / double(p.n);
        }
        output[0] = r;
    }
}
