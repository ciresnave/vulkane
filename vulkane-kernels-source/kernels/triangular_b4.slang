// Triu / Tril for 4-byte element types (f32, i32, u32).
//
// Mask-style op: walk each (i, j) of the [..., rows, cols] tail; per
// (batch, i, j) keep the input element when the triangle predicate
// passes, otherwise emit a zero. Reading the input as `uint` makes the
// kernel dtype-agnostic — the byte pattern of f32 0.0, i32 0, u32 0 is
// all-zero, so a single zero-write covers every 4-byte dtype.
//
// Predicate (signed):
//   Triu (keep_upper = true ):  j >= i + diagonal
//   Tril (keep_upper = false):  j <= i + diagonal
// CPU reference: fuel-cpu-backend/src/byte_kernels.rs `triangular_cpu`.
//
// entries: triu_b4, tril_b4

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

    output[tid] = keep ? input[tid] : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void triu_b4(uint3 gid : SV_DispatchThreadID) { body(gid, true); }

[shader("compute")]
[numthreads(256, 1, 1)]
void tril_b4(uint3 gid : SV_DispatchThreadID) { body(gid, false); }
