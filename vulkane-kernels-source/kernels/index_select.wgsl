// Row-wise lookup along a specified dim.
//
// Semantics (matches fuel-core/candle `index_select`):
//   output[..., j_0, ..., j_{dim-1}, k, j_{dim+1}, ..., j_{N-1}]
//     = input[..., j_0, ..., j_{dim-1}, ids[k], j_{dim+1}, ..., j_{N-1}]
//
// We flatten input and output to 3D [outer, axis, inner] where axis is
// the selected dim. `inner` is the product of dims after `dim`, `outer`
// is the product of dims before `dim`. Each output element at
// (o, k, i) reads input at (o, ids[k], i).
//
// One thread per output element. For the common embeddings case
// (dim == 0, outer == 1, inner == hidden_size, ids length == seq_len)
// this parallelizes across the seq_len * hidden_size output tensor.

struct Params {
    out_size: u32,       // total output elements
    outer: u32,          // product of dims before `dim` in the input
    axis_out: u32,       // length of `ids` (= new size of `dim`)
    inner: u32,          // product of dims after `dim` in the input
    axis_in: u32,        // original size of `dim` in the input
    _pad0: u32, _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.out_size { return; }

    // Unflatten idx into (o, k, i) over output dims [outer, axis_out, inner].
    let inner = params.inner;
    let axis_out = params.axis_out;
    let i = idx % inner;
    let ko = idx / inner;
    let k = ko % axis_out;
    let o = ko / axis_out;

    let src_axis = ids[k];
    // Clamp defensively — out-of-range indices would otherwise read
    // past the input buffer and produce undefined values.
    let safe_axis = min(src_axis, params.axis_in - 1u);
    let src_idx = (o * params.axis_in + safe_axis) * inner + i;
    output[idx] = input[src_idx];
}
