// Element-wise unary operations with per-input stride support.
//
// Output is always contiguous. The input is read via its own per-dim
// strides, which supports:
//   * Lazy-permuted inputs (non-contiguous strides)
//   * Implicit broadcasting (stride=0 on a dim means "repeat")
//   * Sliced views (any strides; auto-Contiguize handles non-zero
//     start_offset separately)
//
// Supports ranks 1-4. Output shape padded to rank 4 with leading 1s
// (and matching stride=0) so the decomposition is uniform.
//
// Fast path: when the input is contiguous AND matches output shape
// exactly, uses simple linear indexing (zero overhead). Mirrors
// binary.slang's stride-aware pattern.
//
// Replaces unary.wgsl. Written in Slang per the shader policy.

struct Params {
    uint out_size;        // output element count
    uint op_id;
    uint rank;            // effective rank (1-4)
    uint flags;           // bit 0: input_contiguous

    // Output shape, padded to rank 4 with 1s in leading dims.
    uint shape0, shape1, shape2, shape3;

    // Input strides in output-space (0 means broadcast along that dim).
    uint in_s0, in_s1, in_s2, in_s3;
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
    if (idx >= p.out_size) return;

    float x;
    bool in_contig = (p.flags & 1u) != 0u;

    if (in_contig) {
        // Fast path: input contiguous and same shape as output.
        x = input[idx];
    } else {
        // Decompose linear idx into per-dim coordinates, then apply
        // per-input strides. Output is contiguous, so we use the
        // output shape's contiguous strides to decompose.
        uint c3 = idx % p.shape3;
        uint r2 = idx / p.shape3;
        uint c2 = r2 % p.shape2;
        uint r1 = r2 / p.shape2;
        uint c1 = r1 % p.shape1;
        uint c0 = r1 / p.shape1;

        uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
        x = input[in_off];
    }

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
