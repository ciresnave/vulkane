// Triu / Tril for 8-byte element types (f64, i64).
//
// Treat each element as a (lo, hi) pair of u32. The pair is always
// adjacent in memory so we read input[2*tid] / input[2*tid+1] for the
// element, and write through unchanged on keep / write zeros on mask.
//
// entries: triu_b8, tril_b8

struct Params {
    uint  batch_count;
    uint  rows;
    uint  cols;
    int   diagonal;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

void body(uint3 gid, bool keep_upper) {
    uint tid = gid.x;
    uint plane = p.rows * p.cols;
    uint total = p.batch_count * plane;
    if (tid >= total) return;

    uint within_plane = tid % plane;
    int  i = int(within_plane / p.cols);
    int  j = int(within_plane % p.cols);

    bool keep = keep_upper
        ? (j >= i + p.diagonal)
        : (j <= i + p.diagonal);

    uint w = tid * 2u;
    if (keep) {
        output[w]      = input[w];
        output[w + 1u] = input[w + 1u];
    } else {
        output[w]      = 0u;
        output[w + 1u] = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void triu_b8(uint3 gid : SV_DispatchThreadID) { body(gid, true); }

[shader("compute")]
[numthreads(256, 1, 1)]
void tril_b8(uint3 gid : SV_DispatchThreadID) { body(gid, false); }
