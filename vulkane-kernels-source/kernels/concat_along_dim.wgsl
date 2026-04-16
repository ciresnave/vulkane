// Single-dispatch concat of two tensors along an arbitrary dim.
//
// Inputs (all row-major, contiguous):
//   a: outer x a_dim x inner
//   b: outer x b_dim x inner
//   dst: outer x (a_dim + b_dim) x inner
//
// One thread per output element. Thread picks its (o, d, i) coords
// from its linear id and reads from a or b depending on d's side of
// the split.
//
// This replaces the host-side loop in GraphExecutor::do_concat that
// dispatched `outer * 2` tiny copy kernels per concat — ~176 per
// TinyLlama token from the KV cache concats alone.

struct Params {
    outer: u32,
    a_dim: u32,
    b_dim: u32,
    inner: u32,
    total: u32,  // outer * (a_dim + b_dim) * inner
};

@group(0) @binding(0) var<storage, read>       a:   array<f32>;
@group(0) @binding(1) var<storage, read>       b:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform>             p:   Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= p.total { return; }

    let out_dim = p.a_dim + p.b_dim;
    let i = tid % p.inner;
    let rest = tid / p.inner;
    let d = rest % out_dim;
    let o = rest / out_dim;

    if d < p.a_dim {
        let a_idx = (o * p.a_dim + d) * p.inner + i;
        dst[tid] = a[a_idx];
    } else {
        let b_idx = (o * p.b_dim + (d - p.a_dim)) * p.inner + i;
        dst[tid] = b[b_idx];
    }
}
