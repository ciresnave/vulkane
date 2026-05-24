// Inclusive prefix sum along a single axis. Rank-N input (padded to
// rank 4 with leading 1s) + per-input strides; output is contig over
// the input's shape.
//
// One thread per "slice" — i.e., one combination of coords for every
// dim except `axis`. Each thread serially accumulates `dim_size`
// elements along the axis. Total threads = product(shape[d]) for d
// != axis. Sequential per-slice keeps the kernel simple and correct;
// block-scan is a follow-up if profiles show large dim_size on the
// hot path.
//
// f32 variant — accumulator stays in f32 (matches CPU/CUDA semantics).

struct Params {
    uint slice_count;       // product of shape over non-axis dims
    uint axis;              // which dim is scanned (0..3)
    uint dim_size;          // shape[axis]
    uint _pad;
    uint shape0, shape1, shape2, shape3;
    uint in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint sid = gid.x;
    if (sid >= p.slice_count) return;

    // Decompose sid into per-dim coords over the shape with the axis
    // dim treated as size 1 (we'll fill that coord inside the loop).
    uint cs0 = (p.axis == 0u) ? 1u : p.shape0;
    uint cs1 = (p.axis == 1u) ? 1u : p.shape1;
    uint cs2 = (p.axis == 2u) ? 1u : p.shape2;
    uint cs3 = (p.axis == 3u) ? 1u : p.shape3;

    uint c3 = sid % cs3;
    uint r2 = sid / cs3;
    uint c2 = r2 % cs2;
    uint r1 = r2 / cs2;
    uint c1 = r1 % cs1;
    uint c0 = r1 / cs1;

    // Contig output strides over the full shape (computed inline).
    uint out_s2 = p.shape3;
    uint out_s1 = p.shape2 * out_s2;
    uint out_s0 = p.shape1 * out_s1;

    float running = 0.0;
    for (uint j = 0u; j < p.dim_size; ++j) {
        // Fill the axis coord with j.
        uint q0 = (p.axis == 0u) ? j : c0;
        uint q1 = (p.axis == 1u) ? j : c1;
        uint q2 = (p.axis == 2u) ? j : c2;
        uint q3 = (p.axis == 3u) ? j : c3;

        uint in_off = q0 * p.in_s0 + q1 * p.in_s1 + q2 * p.in_s2 + q3 * p.in_s3;
        running += input[in_off];

        uint out_off = q0 * out_s0 + q1 * out_s1 + q2 * out_s2 + q3;
        output[out_off] = running;
    }
}
