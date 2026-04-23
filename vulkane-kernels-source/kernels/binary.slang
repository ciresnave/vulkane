// Element-wise binary operations with per-operand stride support.
//
// Output is always contiguous. Each operand is read via its own
// per-dim strides, which supports:
//   * Lazy-permuted inputs (non-contiguous strides)
//   * Implicit broadcasting (stride=0 on a dim means "repeat")
//
// Supports ranks 1-4. Output shape/strides padded with (1, 0) for
// ranks < 4 so the decomposition is uniform.
//
// Replaces binary.wgsl. Written in Slang per the shader policy.
//
// Fast path: when both operands are contiguous and match output shape
// exactly, uses simple linear indexing (zero overhead).

struct Params {
    uint out_size;        // output element count
    uint op_id;
    uint rank;            // effective rank (1-4)
    uint flags;           // bit 0: a_contiguous, bit 1: b_contiguous

    // Output shape, padded to rank 4 with 1s in leading dims.
    uint shape0, shape1, shape2, shape3;

    // A strides in output-space (0 means broadcast along that dim).
    uint a_s0, a_s1, a_s2, a_s3;

    // B strides in output-space (0 means broadcast along that dim).
    uint b_s0, b_s1, b_s2, b_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   a_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float>   b_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

static const uint OP_ADD = 0;
static const uint OP_SUB = 1;
static const uint OP_MUL = 2;
static const uint OP_DIV = 3;
static const uint OP_MAX = 4;
static const uint OP_MIN = 5;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    float x, y;

    bool a_contig = (p.flags & 1u) != 0u;
    bool b_contig = (p.flags & 2u) != 0u;

    if (a_contig && b_contig) {
        // Fast path: both operands contiguous and same shape as output.
        x = a_buf[idx];
        y = b_buf[idx];
    } else {
        // Decompose linear idx into per-dim coordinates, then apply
        // per-operand strides. Output is contiguous, so we use the
        // output shape's contiguous strides to decompose.
        uint c3 = idx % p.shape3;
        uint r2 = idx / p.shape3;
        uint c2 = r2 % p.shape2;
        uint r1 = r2 / p.shape2;
        uint c1 = r1 % p.shape1;
        uint c0 = r1 / p.shape1;

        uint a_off = c0 * p.a_s0 + c1 * p.a_s1 + c2 * p.a_s2 + c3 * p.a_s3;
        uint b_off = c0 * p.b_s0 + c1 * p.b_s1 + c2 * p.b_s2 + c3 * p.b_s3;

        x = a_buf[a_off];
        y = b_buf[b_off];
    }

    float z;
    switch (p.op_id) {
        case OP_ADD: z = x + y; break;
        case OP_SUB: z = x - y; break;
        case OP_MUL: z = x * y; break;
        case OP_DIV: z = x / y; break;
        case OP_MAX: z = max(x, y); break;
        case OP_MIN: z = min(x, y); break;
        default:     z = x; break;
    }

    out_buf[idx] = z;
}
