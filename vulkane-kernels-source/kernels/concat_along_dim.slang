// Single-dispatch concat of two tensors along an arbitrary dim, with
// per-operand stride support so either input may be a lazy view
// (permute, broadcast, etc.) without needing explicit materialization.
//
// Logical shapes (both inputs agree on all dims except `concat_dim`):
//   a: out_d0 × out_d1 × out_d2 × out_d3, with a_dim at concat_dim
//   b: same, with b_dim at concat_dim
//   dst: same, with out_d[concat_dim] = a_dim + b_dim (always contiguous)
//
// One thread per output element. Thread decomposes its linear tid into
// (c0, c1, c2, c3) via the output shape's contiguous strides, decides
// which side of the split it falls on (a or b), and reads through the
// appropriate per-dim strides.
//
// Replaces concat_along_dim.wgsl. Written in Slang per the shader policy.

struct Params {
    // Output shape (padded to rank 4 with leading 1s).
    uint out_d0, out_d1, out_d2, out_d3;
    // Which dim (0..3) is the concat dim. For rank < 4 the caller
    // pads leading dims with 1, so concat_dim is always indexed
    // against the rank-4 padded shape.
    uint concat_dim;
    uint a_dim;      // size of a along concat_dim
    uint b_dim;      // size of b along concat_dim (a_dim + b_dim == out_d[concat_dim])
    uint total;

    // Per-dim strides for a and b. Zero strides are valid for
    // broadcast-view inputs. The stride at the concat dim applies
    // to coord positions within [0, a_dim) for a / [0, b_dim) for b.
    uint a_s0, a_s1, a_s2, a_s3;
    uint b_s0, b_s1, b_s2, b_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   a_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float>   b_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.total) return;

    // Decompose tid into per-dim coords via output shape.
    uint c3 = tid % p.out_d3;
    uint r2 = tid / p.out_d3;
    uint c2 = r2 % p.out_d2;
    uint r1 = r2 / p.out_d2;
    uint c1 = r1 % p.out_d1;
    uint c0 = r1 / p.out_d1;

    uint coord_at_concat;
    if      (p.concat_dim == 0) coord_at_concat = c0;
    else if (p.concat_dim == 1) coord_at_concat = c1;
    else if (p.concat_dim == 2) coord_at_concat = c2;
    else                        coord_at_concat = c3;

    bool from_a = coord_at_concat < p.a_dim;
    uint c_adjusted = from_a ? coord_at_concat : (coord_at_concat - p.a_dim);

    // Reassemble coords with the adjusted concat-dim coord.
    uint k0 = (p.concat_dim == 0) ? c_adjusted : c0;
    uint k1 = (p.concat_dim == 1) ? c_adjusted : c1;
    uint k2 = (p.concat_dim == 2) ? c_adjusted : c2;
    uint k3 = (p.concat_dim == 3) ? c_adjusted : c3;

    float value;
    if (from_a) {
        uint off = k0 * p.a_s0 + k1 * p.a_s1 + k2 * p.a_s2 + k3 * p.a_s3;
        value = a_buf[off];
    } else {
        uint off = k0 * p.b_s0 + k1 * p.b_s1 + k2 * p.b_s2 + k3 * p.b_s3;
        value = b_buf[off];
    }

    dst[tid] = value;
}
