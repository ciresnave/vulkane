// Row-wise lookup along a specified dim — bf16.
//
// Storage: bf16 packed two-per-u32. Pair-thread layout — each thread
// handles 2 inner positions (i, i+1) at once and reads/writes a full
// u32 from input/output. Pure data movement; no math; no bf16↔f32
// conversion needed.
//
// PRECONDITION: `inner % 2 == 0` (so each (o, k, *) row of `inner`
// bf16 elements packs into `inner / 2` u32 words and our pair-thread
// indexing stays u32-aligned). For embedding-lookup workloads
// (LLM tied embeddings, T5 / BERT vocabularies) `inner` is the embed
// dim which is always even. The wrapper enforces.

struct Params {
    uint out_size;     // bf16-element count in the output
    uint outer;
    uint axis_out;
    uint inner;        // bf16-element count along inner dim (MUST be even)
    uint axis_in;
    uint _pad0, _pad1, _pad2;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;     // packed bf16
[[vk::binding(1, 0)]] StructuredBuffer<uint>   ids;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint inner_pairs = p.inner >> 1u;
    uint out_pairs = p.out_size >> 1u;
    if (pair_idx >= out_pairs) return;

    uint i_pair = pair_idx % inner_pairs;
    uint ko = pair_idx / inner_pairs;
    uint k = ko % p.axis_out;
    uint o = ko / p.axis_out;

    uint src_axis = ids[k];
    uint safe_axis = min(src_axis, p.axis_in - 1u);
    uint src_u32 = (o * p.axis_in + safe_axis) * inner_pairs + i_pair;
    // dst layout: (o * axis_out + k) * inner_pairs + i_pair
    output[pair_idx] = input[src_u32];
}
