// Triu / Tril for 2-byte element types (f16, bf16).
//
// Read storage as `uint` (4 bytes = pair of 2-byte elements); each
// thread processes ONE pair. The pair straddles two columns, which
// can fall on different sides of the triangle predicate — handle the
// two halves independently and re-pack.
//
// `cols` MUST be even on entry (wrapper validates and falls back to
// CPU otherwise — same constraint as write_slice_b2). With even cols
// the (i, 2k) and (i, 2k+1) elements live in the same u32 and the
// pair-index decomposition is unambiguous.
//
// entries: triu_b2, tril_b2

struct Params {
    uint  batch_count;
    uint  rows;
    uint  cols;        // must be even
    int   diagonal;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // 2 elems / u32
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

void body(uint3 gid, bool keep_upper) {
    uint tid = gid.x;
    uint half_cols = p.cols >> 1u;
    uint pairs_per_plane = p.rows * half_cols;
    uint total_pairs = p.batch_count * pairs_per_plane;
    if (tid >= total_pairs) return;

    uint within_plane = tid % pairs_per_plane;
    int  i  = int(within_plane / half_cols);
    int  jp = int(within_plane % half_cols);
    int  j0 = jp * 2;
    int  j1 = j0 + 1;

    bool keep0 = keep_upper
        ? (j0 >= i + p.diagonal)
        : (j0 <= i + p.diagonal);
    bool keep1 = keep_upper
        ? (j1 >= i + p.diagonal)
        : (j1 <= i + p.diagonal);

    uint src = input[tid];
    uint lo  = keep0 ? (src & 0x0000FFFFu) : 0u;
    uint hi  = keep1 ? (src & 0xFFFF0000u) : 0u;
    output[tid] = lo | hi;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void triu_b2(uint3 gid : SV_DispatchThreadID) { body(gid, true); }

[shader("compute")]
[numthreads(256, 1, 1)]
void tril_b2(uint3 gid : SV_DispatchThreadID) { body(gid, false); }
