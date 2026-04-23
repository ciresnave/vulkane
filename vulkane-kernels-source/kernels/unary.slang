// Element-wise unary operations. One thread per element.
//
// Replaces unary.wgsl. Written in Slang per the shader policy.

struct Params {
    uint n;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

static const uint OP_NEG     = 0;
static const uint OP_SQR     = 1;
static const uint OP_SQRT    = 2;
static const uint OP_EXP     = 3;
static const uint OP_LOG     = 4;
static const uint OP_SIN     = 5;
static const uint OP_COS     = 6;
static const uint OP_TANH    = 7;
static const uint OP_SIGMOID = 8;
static const uint OP_SILU    = 9;
static const uint OP_GELU    = 10;
static const uint OP_RELU    = 11;
static const uint OP_STEP    = 12;

static const float SQRT_2_OVER_PI = 0.7978845608f;

float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * x * (1.0f + 0.044715f * x * x)));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.n) return;

    float x = input[idx];
    float y;

    switch (p.op_id) {
        case OP_NEG:     y = -x; break;
        case OP_SQR:     y = x * x; break;
        case OP_SQRT:    y = sqrt(x); break;
        case OP_EXP:     y = exp(x); break;
        case OP_LOG:     y = log(x); break;
        case OP_SIN:     y = sin(x); break;
        case OP_COS:     y = cos(x); break;
        case OP_TANH:    y = tanh(x); break;
        case OP_SIGMOID: y = sigmoid(x); break;
        case OP_SILU:    y = x * sigmoid(x); break;
        case OP_GELU:    y = gelu_tanh(x); break;
        case OP_RELU:    y = max(x, 0.0f); break;
        case OP_STEP:    y = x > 0.0f ? 1.0f : 0.0f; break;
        default:         y = x; break;
    }

    output[idx] = y;
}
