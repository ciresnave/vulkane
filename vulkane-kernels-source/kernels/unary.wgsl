// Element-wise unary operations. One thread per element.

struct Params {
    n: u32,
    op_id: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const OP_NEG: u32     = 0u;
const OP_SQR: u32     = 1u;
const OP_SQRT: u32    = 2u;
const OP_EXP: u32     = 3u;
const OP_LOG: u32     = 4u;
const OP_SIN: u32     = 5u;
const OP_COS: u32     = 6u;
const OP_TANH: u32    = 7u;
const OP_SIGMOID: u32 = 8u;
const OP_SILU: u32    = 9u;
const OP_GELU: u32    = 10u;
const OP_RELU: u32    = 11u;
const OP_STEP: u32    = 12u;

const SQRT_2_OVER_PI: f32 = 0.7978845608;

fn gelu_tanh(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let x = input[idx];
    var y: f32;

    switch params.op_id {
        case OP_NEG     { y = -x; }
        case OP_SQR     { y = x * x; }
        case OP_SQRT    { y = sqrt(x); }
        case OP_EXP     { y = exp(x); }
        case OP_LOG     { y = log(x); }
        case OP_SIN     { y = sin(x); }
        case OP_COS     { y = cos(x); }
        case OP_TANH    { y = tanh(x); }
        case OP_SIGMOID { y = sigmoid(x); }
        case OP_SILU    { y = x * sigmoid(x); }
        case OP_GELU    { y = gelu_tanh(x); }
        case OP_RELU    { y = max(x, 0.0); }
        case OP_STEP    { y = select(0.0, 1.0, x > 0.0); }
        default         { y = x; }
    }

    output[idx] = y;
}
