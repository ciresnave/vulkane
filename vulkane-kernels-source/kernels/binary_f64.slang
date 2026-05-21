// Element-wise binary ops on f64 (Add/Sub/Mul/Div/Max/Min). Mirrors
// binary.slang exactly; only buffer element type changes.

struct Params {
    uint out_size;
    uint op_id;
    uint rank;
    uint flags;

    uint shape0, shape1, shape2, shape3;
    uint a_s0, a_s1, a_s2, a_s3;
    uint b_s0, b_s1, b_s2, b_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   a_buf;
[[vk::binding(1, 0)]] StructuredBuffer<double>   b_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> out_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>     p;

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

    double x, y;

    bool a_contig = (p.flags & 1u) != 0u;
    bool b_contig = (p.flags & 2u) != 0u;

    if (a_contig && b_contig) {
        x = a_buf[idx];
        y = b_buf[idx];
    } else {
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

    double z;
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
