// SPDX-License-Identifier: MIT OR Apache-2.0
//! Module-level declarations: capabilities, types, the invocation-id input and the buffers.

use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, MemoryModel, StorageClass, Word,
};

use crate::ops::Family;

/// Ids of the module-level declarations the function body refers to.
pub(crate) struct Skeleton {
    pub(crate) b: Builder,
    pub(crate) glsl: Word,
    pub(crate) void: Word,
    pub(crate) fn_ty: Word,
    pub(crate) t: Word,
    pub(crate) boolean: Word,
    pub(crate) u32_t: Word,
    pub(crate) vec3: Word,
    pub(crate) gid: Word,
    pub(crate) zero_u32: Word,
    pub(crate) inputs: Vec<Word>,
    pub(crate) output: Word,
}

/// A builder with the version, capabilities and memory model a `fam` kernel needs.
fn header(fam: Family) -> Builder {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    if fam == Family::F64 {
        b.capability(Capability::Float64);
    }
    if fam.is_float() {
        b.capability(Capability::DenormPreserve);
        b.extension("SPV_KHR_float_controls");
    }
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    b
}

/// The element type for `fam`.
fn element_type(b: &mut Builder, fam: Family, u32_t: Word) -> Word {
    match fam {
        Family::F32 => b.type_float(32, None),
        Family::F64 => b.type_float(64, None),
        Family::I32 => b.type_int(32, 1),
        Family::U32 => u32_t,
    }
}

/// The `GlobalInvocationId` input variable.
fn invocation_id(b: &mut Builder, vec3: Word) -> Word {
    let ptr = b.type_pointer(None, StorageClass::Input, vec3);
    let gid = b.variable(ptr, None, StorageClass::Input, None);
    b.decorate(
        gid,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
    );
    gid
}

/// `n_in + 1` storage-buffer blocks (`struct { T data[]; }`, set 0, bindings `0..=n_in`); the
/// last is the output. Returns `(inputs, output)`.
fn buffers(b: &mut Builder, fam: Family, t: Word, n_in: usize) -> (Vec<Word>, Word) {
    let rt_array = b.type_runtime_array(t);
    b.decorate(
        rt_array,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(fam.bytes())],
    );
    let block = b.type_struct([rt_array]);
    b.decorate(block, Decoration::Block, []);
    b.member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let ptr = b.type_pointer(None, StorageClass::StorageBuffer, block);
    let mut all = Vec::with_capacity(n_in + 1);
    for binding in 0..=n_in {
        let v = b.variable(ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        let slot = u32::try_from(binding).expect("binding fits u32");
        b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(slot)]);
        all.push(v);
    }
    let output = all.pop().expect("output buffer");
    (all, output)
}

/// Declare everything outside the function body for a `fam` kernel with `n_in` inputs.
pub(crate) fn declare(fam: Family, n_in: usize) -> Skeleton {
    let mut b = header(fam);
    let glsl = b.ext_inst_import("GLSL.std.450");
    let void = b.type_void();
    let fn_ty = b.type_function(void, []);
    let boolean = b.type_bool();
    let u32_t = b.type_int(32, 0);
    let t = element_type(&mut b, fam, u32_t);
    let vec3 = b.type_vector(u32_t, 3);
    let gid = invocation_id(&mut b, vec3);
    let (inputs, output) = buffers(&mut b, fam, t, n_in);
    let zero_u32 = b.constant_bit32(u32_t, 0);
    Skeleton {
        b,
        glsl,
        void,
        fn_ty,
        t,
        boolean,
        u32_t,
        vec3,
        gid,
        zero_u32,
        inputs,
        output,
    }
}
