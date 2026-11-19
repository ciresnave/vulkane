// SPDX-License-Identifier: MIT OR Apache-2.0
//! Module-level declarations for a kernel: types, builtins, storage buffers, workgroup arrays.

use rspirv::dr::Operand;
use rspirv::spirv::{BuiltIn, Decoration, StorageClass, Word};
use unpopped::exec::Extent;
use unpopped::kernel::{BufferRole, KernelFn, ParamKind};

use crate::ops::Family;
use crate::skeleton::{element_type, header};

/// An `Input` builtin variable of `vec3` type.
pub(super) fn builtin_var(b: &mut rspirv::dr::Builder, vec3: Word, which: BuiltIn) -> Word {
    let ptr = b.type_pointer(None, StorageClass::Input, vec3);
    let v = b.variable(ptr, None, StorageClass::Input, None);
    b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(which)]);
    v
}

/// The storage-buffer variable of a parameter at binding `binding`.
pub(super) fn buffer_var(
    b: &mut rspirv::dr::Builder,
    ptr: Word,
    binding: u32,
    role: BufferRole,
) -> Word {
    let v = b.variable(ptr, None, StorageClass::StorageBuffer, None);
    b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
    b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    // On the variable, not the block member: rspirv shares one block type across the buffers, and
    // the roles differ per buffer.
    if role == BufferRole::Read {
        b.decorate(v, Decoration::NonWritable, []);
    }
    v
}

/// The module-level ids the function body refers to.
pub(super) struct Module {
    pub(super) b: rspirv::dr::Builder,
    pub(super) fam: Family,
    pub(super) glsl: Word,
    pub(super) void: Word,
    pub(super) fn_ty: Word,
    pub(super) boolean: Word,
    pub(super) u32_t: Word,
    pub(super) t: Word,
    pub(super) vec3: Word,
    pub(super) local_id: Word,
    pub(super) group_id: Word,
    pub(super) bufs: Vec<Word>,
    pub(super) allocs: Vec<Word>,
    /// `(local, spec constant id)` of each scalar parameter lowered as a specialization constant.
    pub(super) specs: Vec<(u32, Word)>,
}

/// What `declare_storage` made.
pub(super) struct Storage {
    bufs: Vec<Word>,
    allocs: Vec<Word>,
    specs: Vec<(u32, Word)>,
}

/// The storage buffers (one per parameter) and the workgroup arrays (one per allocation).
pub(super) fn declare_storage(
    b: &mut rspirv::dr::Builder,
    kernel: &KernelFn,
    fam: Family,
    u32_t: Word,
    t: Word,
) -> Storage {
    // One run-time array and one block type (rspirv shares them), each decorated exactly once.
    let rt = b.type_runtime_array(t);
    b.decorate(
        rt,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(fam.bytes())],
    );
    let blk = b.type_struct([rt]);
    b.decorate(blk, Decoration::Block, []);
    b.member_decorate(blk, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let buf_ptr = b.type_pointer(None, StorageClass::StorageBuffer, blk);
    let (mut bufs, mut specs) = (Vec::new(), Vec::new());
    for (i, p) in kernel.params.iter().enumerate() {
        let id = u32::try_from(i).expect("parameter index fits u32");
        match &p.kind {
            ParamKind::Buffer { role, .. } => bufs.push(buffer_var(b, buf_ptr, id, *role)),
            ParamKind::Scalar { local, .. } => {
                // A specialization constant: its SpecId is the parameter's index, its default 1.
                let spec = b.spec_constant_bit32(u32_t, 1);
                b.decorate(spec, Decoration::SpecId, [Operand::LiteralBit32(id)]);
                specs.push((*local, spec));
                bufs.push(spec); // placeholder: a scalar parameter has no buffer
            }
            _ => unreachable!("check_scope admits buffers and extent scalars only"),
        }
    }
    let mut allocs = Vec::new();
    for a in &kernel.allocs {
        let len = match a.extent {
            Extent::Static(n) => b.constant_bit32(u32_t, n),
            Extent::Param(l) => specs.iter().find(|(loc, _)| *loc == l).expect("checked").1,
            _ => unreachable!("check_scope admits static and parameter extents only"),
        };
        let arr = b.type_array(t, len);
        let ptr = b.type_pointer(None, StorageClass::Workgroup, arr);
        allocs.push(b.variable(ptr, None, StorageClass::Workgroup, None));
    }
    Storage {
        bufs,
        allocs,
        specs,
    }
}

/// Declare types, builtins, buffers and workgroup arrays for `kernel`.
pub(super) fn declare(kernel: &KernelFn, fam: Family) -> Module {
    let mut b = header(fam);
    let glsl = b.ext_inst_import("GLSL.std.450");
    let void = b.type_void();
    let fn_ty = b.type_function(void, []);
    let boolean = b.type_bool();
    let u32_t = b.type_int(32, 0);
    let t = element_type(&mut b, fam, u32_t);
    let vec3 = b.type_vector(u32_t, 3);
    let local_id = builtin_var(&mut b, vec3, BuiltIn::LocalInvocationId);
    let group_id = builtin_var(&mut b, vec3, BuiltIn::WorkgroupId);
    let Storage {
        bufs,
        allocs,
        specs,
    } = declare_storage(&mut b, kernel, fam, u32_t, t);
    Module {
        b,
        fam,
        glsl,
        void,
        fn_ty,
        boolean,
        u32_t,
        t,
        vec3,
        local_id,
        group_id,
        bufs,
        allocs,
        specs,
    }
}
