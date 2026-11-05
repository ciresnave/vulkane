// Pure HLSL: struct with semantics and a tbuffer; a helper function plus a kernel.
struct Vertex
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

tbuffer Table
{
    float4 weights[4];
};

RWStructuredBuffer<float> output : register(u0);

float scale(Vertex v) { return v.pos.x * v.uv.y; }

[numthreads(64, 1, 1)]
void k(uint3 tid : SV_DispatchThreadID)
{
    Vertex v;
    v.pos = weights[tid.x & 3];
    v.uv = float2(1.0f, 2.0f);
    output[tid.x] = scale(v);
}
