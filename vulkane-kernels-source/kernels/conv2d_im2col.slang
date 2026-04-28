// Conv2D im2col: rearrange NCHW input into the patches matrix that
// the existing matmul shader can consume to produce conv2d output.
//
// Input layout:
//   x: [batch, c_in, h, w]  (NCHW row-major)
//
// Output layout (matches fuel_conv::im2col on the CPU side, used as
// the parity oracle for this shader):
//   patches: [batch * groups, c_in_per_group * k_h * k_w, h_out * w_out]
//
// Each thread writes exactly one element of `patches`. With groups=1
// this is a standard im2col rearrangement; groups > 1 produces the
// per-group sub-blocks the matmul step reads independently.
//
// After this kernel:
//   for ni in 0..batch:
//       for g in 0..groups:
//           bg = ni * groups + g
//           weight_g  = weight[g*cout_per_g..(g+1)*cout_per_g, :, :, :]
//                       reshaped to [cout_per_g, cin_per_g * k_h * k_w]
//           patches_g = patches[bg, :, :]  // [cin_per_g * k_h * k_w, h_out * w_out]
//           out[ni, g*cout_per_g..(g+1)*cout_per_g, :, :]
//               = (weight_g @ patches_g) reshaped to [cout_per_g, h_out, w_out]
//
// Asymmetric stride and padding are supported (separate h / w fields).

struct Params {
    // Input shape.
    uint batch;
    uint c_in;
    uint h;
    uint w;
    // Output spatial shape.
    uint h_out;
    uint w_out;
    // Kernel.
    uint k_h;
    uint k_w;
    // Stride and padding (h, w).
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    // Grouped conv: groups must divide c_in and c_out.
    uint groups;
    uint cin_per_g;
    // Total output element count = batch * groups * (cin_per_g * k_h * k_w) * (h_out * w_out).
    // Stored as a u32 because Slang loop induction variables are u32.
    uint total_elements;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> patches;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.total_elements) return;

    // Decompose tid into (bg, patch_row, spatial_idx).
    uint spatial = p.h_out * p.w_out;
    uint patch_dim = p.cin_per_g * p.k_h * p.k_w;

    uint spatial_idx = tid % spatial;
    uint t1 = tid / spatial;
    uint patch_row = t1 % patch_dim;
    uint bg = t1 / patch_dim;

    // bg = ni * groups + g
    uint ni = bg / p.groups;
    uint g = bg % p.groups;

    // patch_row = ((ci_in_g * k_h) + ky) * k_w + kx
    uint kx = patch_row % p.k_w;
    uint t2 = patch_row / p.k_w;
    uint ky = t2 % p.k_h;
    uint ci_in_g = t2 / p.k_h;
    uint ci = g * p.cin_per_g + ci_in_g;

    // spatial_idx = oh * w_out + ow
    uint ow = spatial_idx % p.w_out;
    uint oh = spatial_idx / p.w_out;

    // Compute the input position this thread reads. Use signed math
    // for the padding subtraction so we can detect out-of-bounds via
    // the unsigned wrap that would otherwise be hidden.
    int ih_signed = int(oh * p.stride_h + ky) - int(p.pad_h);
    int iw_signed = int(ow * p.stride_w + kx) - int(p.pad_w);
    bool in_bounds =
        ih_signed >= 0 && ih_signed < int(p.h) &&
        iw_signed >= 0 && iw_signed < int(p.w);

    float val;
    if (in_bounds) {
        uint ih = uint(ih_signed);
        uint iw = uint(iw_signed);
        uint x_off = ((ni * p.c_in + ci) * p.h + ih) * p.w + iw;
        val = x[x_off];
    } else {
        val = 0.0f;
    }

    // Write to patches[bg, patch_row, spatial_idx].
    uint out_off = (bg * patch_dim + patch_row) * spatial + spatial_idx;
    patches[out_off] = val;
}
