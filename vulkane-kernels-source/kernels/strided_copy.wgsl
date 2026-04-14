// Strided copy: read `out_size` elements from `input` using the
// given source strides + start offset, write contiguously into
// `output` starting at `dst_offset`.
//
// Handles permute, broadcast (via stride=0 on broadcast dims),
// concat (via dst_offset), and slice (via src_offset).
//
// `shape_strides` storage buffer layout:
//   [shape[0], shape[1], ..., shape[rank-1],
//    stride[0], stride[1], ..., stride[rank-1]]

struct Params {
    out_size: u32,
    rank: u32,
    src_offset: u32,
    dst_offset: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> shape_strides: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    if out_idx >= params.out_size { return; }

    // Unflatten out_idx in row-major order using the output shape.
    // Output is contiguous; compute coord per dim by successive division.
    var remainder = out_idx;
    var src_flat = params.src_offset;

    // Compute the contiguous stride for each dim of the output shape.
    // We build strides[d] = product of shape[d+1..rank].
    // Then coord = remainder / strides[d], remainder %= strides[d].
    for (var d = 0u; d < params.rank; d++) {
        var dim_stride: u32 = 1u;
        for (var e = d + 1u; e < params.rank; e++) {
            dim_stride *= shape_strides[e];
        }
        let coord = remainder / dim_stride;
        remainder -= coord * dim_stride;
        src_flat += coord * shape_strides[params.rank + d];
    }

    output[params.dst_offset + out_idx] = input[src_flat];
}
