// Parallel reduction: sum, max, min, or mean of all elements → single value.
// One workgroup, tree reduction.
//
// op_id: 0=sum, 1=max, 2=min, 3=mean (sum / n).
//
// Replaces reduce.wgsl. Written in Slang per the shader policy.

struct Params {
    uint n;
    uint op_id; // 0=sum, 1=max, 2=min, 3=mean
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

static const uint OP_SUM  = 0;
static const uint OP_MAX  = 1;
static const uint OP_MIN  = 2;
static const uint OP_MEAN = 3;

groupshared float wg_data[256];

float identity(uint op) {
    switch (op) {
        case OP_SUM:  return 0.0f;
        case OP_MEAN: return 0.0f;
        case OP_MAX:  return -3.402823e+38f;
        case OP_MIN:  return 3.402823e+38f;
        default:      return 0.0f;
    }
}

float combine(uint op, float a, float b) {
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

    // Each thread accumulates a stride of the input.
    float acc = identity(op);
    uint i = tid;
    while (i < p.n) {
        acc = combine(op, acc, input[i]);
        i += 256u;
    }
    wg_data[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction.
    uint stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_data[tid] = combine(op, wg_data[tid], wg_data[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        float r = wg_data[0];
        if (op == OP_MEAN) {
            r = r / float(p.n);
        }
        output[0] = r;
    }
}
