// Fused rotary position embedding (rotate_half convention).
//
// x:   [..., seq, head_dim]  (head_dim even; let h = head_dim/2)
// cos: [seq, head_dim]       (always contiguous)
// sin: [seq, head_dim]       (always contiguous)
// out: same shape as x       (always contiguous)
//
// x may be non-contiguous (e.g. after a lazy [0,2,1,3] permute).
// When x_contiguous == 0, reads decompose the linear thread index
// into per-dim coordinates and use explicit strides.
//
// Formula:
//   out[o, s, i]     = x[o, s, i]     * cos[s, i]     - x[o, s, i+h] * sin[s, i]
//   out[o, s, i+h]   = x[o, s, i+h]   * cos[s, i+h]   + x[o, s, i]   * sin[s, i+h]
//
// Replaces rope.wgsl. Written in Slang for cross-backend source
// compatibility and future autodiff support.
//
// One thread per (o, s, i) with i in [0, h). Each thread writes two
// output positions: i and i+h. Dispatch count: ceil(outer*seq*h / 64).

struct Params {
    uint outer;         // product of dims before the seq dim
    uint seq;
    uint head_dim;      // full head_dim (h = head_dim/2)
    uint total;         // outer * seq * (head_dim/2) — iteration bound
    // x stride support (for non-contiguous inputs)
    uint x_s0;          // stride of outermost outer dim
    uint x_s1;          // stride of inner outer dim
    uint x_s_seq;       // stride of seq dim
    uint x_s_hd;        // stride of head_dim dim
    uint x_outer1;      // size of inner outer dim (outer = outer0 * outer1)
    uint x_contiguous;  // 1 = contiguous fast path, 0 = strided
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   x;
[[vk::binding(1, 0)]] StructuredBuffer<float>   cos_tab;
[[vk::binding(2, 0)]] StructuredBuffer<float>   sin_tab;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out;
[[vk::binding(4, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.total) return;

    uint h = p.head_dim / 2;
    // Decode (o, s, i) from linear tid.
    uint i  = tid % h;
    uint os = tid / h;
    uint s  = os % p.seq;
    uint o  = os / p.seq;

    // Output is always contiguous.
    uint row_off   = (o * p.seq + s) * p.head_dim;
    // cos/sin are always contiguous [seq, head_dim].
    uint table_off = s * p.head_dim;

    // Read x: contiguous fast path or strided decomposition.
    float x0, x1;
    if (p.x_contiguous == 1) {
        x0 = x[row_off + i];
        x1 = x[row_off + i + h];
    } else {
        uint o1 = o % p.x_outer1;
        uint o0 = o / p.x_outer1;
        uint x_base = o0 * p.x_s0 + o1 * p.x_s1 + s * p.x_s_seq;
        x0 = x[x_base + i * p.x_s_hd];
        x1 = x[x_base + (i + h) * p.x_s_hd];
    }

    float c0 = cos_tab[table_off + i];
    float s0 = sin_tab[table_off + i];
    float c1 = cos_tab[table_off + i + h];
    float s1 = sin_tab[table_off + i + h];

    out[row_off + i]     = x0 * c0 - x1 * s0;
    out[row_off + i + h] = x1 * c1 + x0 * s1;
}
