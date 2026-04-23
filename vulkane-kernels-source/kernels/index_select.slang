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
// One thread per output element.
//
// Replaces index_select.wgsl. Written in Slang per the shader policy.

struct Params {
    uint out_size;   // total output elements
    uint outer;      // product of dims before `dim` in the input
    uint axis_out;   // length of `ids` (= new size of `dim`)
    uint inner;      // product of dims after `dim` in the input
    uint axis_in;    // original size of `dim` in the input
    uint _pad0, _pad1, _pad2;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    ids;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    uint inner = p.inner;
    uint axis_out = p.axis_out;
    uint i = idx % inner;
    uint ko = idx / inner;
    uint k = ko % axis_out;
    uint o = ko / axis_out;

    uint src_axis = ids[k];
    // Clamp defensively — out-of-range indices would otherwise read
    // past the input buffer and produce undefined values.
    uint safe_axis = min(src_axis, p.axis_in - 1u);
    uint src_idx = (o * p.axis_in + safe_axis) * inner + i;
    output[idx] = input[src_idx];
}
