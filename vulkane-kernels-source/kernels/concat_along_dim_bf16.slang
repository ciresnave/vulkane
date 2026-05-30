// Concat along arbitrary dim — bf16.
//
// Storage: bf16 packed two-per-u32. The challenge for bf16 concat is
// that when `concat_dim` is the last (innermost) dim and `a_dim` is
// odd, adjacent output positions can come from DIFFERENT source
// buffers (a vs b) — so a direct packed-pair write would either
// race (multi-thread per u32) or require complex inter-input
// reads.
//
// Solution: one thread per output bf16 element. Each thread reads a
// single bf16 lane from the appropriate source and writes a single
// bf16 lane to the output via `InterlockedOr`. The wrapper
// zero-fills the output buffer first (so the OR acts as a clean
// half-word write) and binds the output descriptor with u32-rounded
// size (so writes to the last partial u32 aren't discarded by
// Vulkan robust-access).
//
// This is slower than packed-pair writes when no race exists, but
// correct universally. Worth keeping simple — concat isn't usually
// in the hot loop.

struct Params {
    uint out_d0, out_d1, out_d2, out_d3;
    uint concat_dim;
    uint a_dim;
    uint b_dim;
    uint total;
    uint a_s0, a_s1, a_s2, a_s3;
    uint b_s0, b_s1, b_s2, b_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   a_buf;     // packed bf16
[[vk::binding(1, 0)]] StructuredBuffer<uint>   b_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> dst;       // zero-filled by wrapper
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.total) return;

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

    uint k0 = (p.concat_dim == 0) ? c_adjusted : c0;
    uint k1 = (p.concat_dim == 1) ? c_adjusted : c1;
    uint k2 = (p.concat_dim == 2) ? c_adjusted : c2;
    uint k3 = (p.concat_dim == 3) ? c_adjusted : c3;

    // Read source half-word (single bf16 lane).
    uint src_elem_off;
    uint bf16_bits;
    if (from_a) {
        src_elem_off = k0 * p.a_s0 + k1 * p.a_s1 + k2 * p.a_s2 + k3 * p.a_s3;
        uint word = a_buf[src_elem_off >> 1u];
        bf16_bits = (src_elem_off & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
    } else {
        src_elem_off = k0 * p.b_s0 + k1 * p.b_s1 + k2 * p.b_s2 + k3 * p.b_s3;
        uint word = b_buf[src_elem_off >> 1u];
        bf16_bits = (src_elem_off & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
    }

    // Write into the appropriate half of dst[tid >> 1].
    uint shifted = bf16_bits << ((tid & 1u) * 16u);
    InterlockedOr(dst[tid >> 1u], shifted);
}
