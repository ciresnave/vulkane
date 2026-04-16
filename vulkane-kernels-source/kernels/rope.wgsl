// Fused rotary position embedding (rotate_half convention).
//
// x:   [..., seq, head_dim]  (head_dim even; let h = head_dim/2)
// cos: [seq, head_dim]
// sin: [seq, head_dim]
// out: same shape as x
//
// Formula:
//   out[o, s, i]     = x[o, s, i]     * cos[s, i]     - x[o, s, i+h] * sin[s, i]
//   out[o, s, i+h]   = x[o, s, i+h]   * cos[s, i+h]   + x[o, s, i]   * sin[s, i+h]
//
// One thread per (o, s, i) with i in [0, h). Each thread writes two
// output positions: i and i+h. Dispatch count: ceil(outer*seq*h / 64).

struct Params {
    outer: u32,     // product of dims before the seq dim
    seq: u32,
    head_dim: u32,  // full head_dim (h = head_dim/2)
    total: u32,     // outer * seq * (head_dim/2) — iteration bound
};

@group(0) @binding(0) var<storage, read>       x:    array<f32>;
@group(0) @binding(1) var<storage, read>       cos:  array<f32>;
@group(0) @binding(2) var<storage, read>       sin:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out:  array<f32>;
@group(0) @binding(4) var<uniform>             p:    Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= p.total { return; }

    let h = p.head_dim / 2u;
    // Decode (o, s, i) from linear tid.
    let i = tid % h;
    let os = tid / h;
    let s = os % p.seq;
    let o = os / p.seq;

    let row_off = (o * p.seq + s) * p.head_dim;
    let table_off = s * p.head_dim;

    let x0 = x[row_off + i];
    let x1 = x[row_off + i + h];
    let c0 = cos[table_off + i];
    let s0 = sin[table_off + i];
    let c1 = cos[table_off + i + h];
    let s1 = sin[table_off + i + h];

    out[row_off + i]     = x0 * c0 - x1 * s0;
    out[row_off + i + h] = x1 * c1 + x0 * s1;
}
