// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel`: the kernel layer (`unpopped::kernel::KernelFn`) lowered to a SPIR-V
//! compute module. ROADMAP rows 8a (workgroup memory) and 8b (barriers and fences), first slice.
//!
//! ## What v1 lowers
//!
//! - **Integer kernels only**: `compute` is `I32` or `U32`, and every buffer, shared array and
//!   local has that element type. A float kernel is a typed decline (the thread ids are integers,
//!   and a float kernel needs casts this slice does not spell).
//! - **Parameters**: `Buffer` only, one storage buffer per parameter at set 0, binding = the
//!   parameter's index; a `Read` buffer is decorated `NonWritable`. `Scalar` and `Struct`
//!   parameters, multi-coordinate places and peer/multicast places are declined.
//! - **Launch**: the kernel must name a [`LaunchShape`](unpopped::exec::LaunchShape); its `block`
//!   is the `LocalSize`, `BlockDim`/`GridDim` are constants from it. `grid_dims` that depend on a
//!   parameter are declined.
//! - **Workgroup memory** (row 8a): `MemorySpace::Shared` arrays of `Extent::Static` become
//!   `OpVariable Workgroup` of array type. A dynamic extent (`Extent::Param`) is a typed decline:
//!   SPIR-V has no per-dispatch extent (it needs a specialization constant, fixed per pipeline).
//! - **Statements**: `Let`, `Load`, `Assign`, `Store`, `Barrier`, `Fence` and the structured control
//!   flow of row 8g (`If`, `For`, `While`, `Break`, `Continue`, `Return`; see the `flow` module). Atomics,
//!   collectives and everything else is declined naming the statement.
//!
//! ## Barriers and fences (row 8b)
//!
//! `Barrier { Block, Full }` is `OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory|UniformMemory`.
//! **Deviation from the ROADMAP text, deliberate:** the semantics also name `UniformMemory`
//! (storage buffers), because Unpopped's `Barrier` is "the only ordering a kernel has" and a
//! CUDA `__syncthreads()` orders global writes within the block too; naming only
//! `WorkgroupMemory` would leave a buffer write unordered. `Barrier { Subgroup, Full }` is the same
//! with the `Subgroup` scope. `Fence` is `OpMemoryBarrier`: `Acquire`/`Release`/`AcqRel` map to
//! their SPIR-V semantics, and **`SeqCst` collapses to `AcquireRelease`** (Vulkan's SPIR-V
//! environment has no sequentially consistent barrier). The `Device` fence scope is `Device`.
//!
//! A barrier under thread-divergent control is refused by `check_kernel` (`BarrierDivergent`), which
//! runs first, so it is a typed decline carrying the checker's answer and is never silently moved.

mod flow;

use std::cell::RefCell;
use std::collections::HashMap;

use rspirv::binary::{Assemble, Disassemble};
use rspirv::dr::Operand;
use rspirv::spirv::{
    BuiltIn, Decoration, ExecutionMode, ExecutionModel, FunctionControl, MemorySemantics,
    Scope as SpvScope, StorageClass, Word,
};
use unpopped::backend::{Decline, DeclinedOp, GeneratedKernel, LowerError, Spelling};
use unpopped::exec::{ExecId, Extent, MemorySpace, Scope};
use unpopped::kernel::{
    BarrierKind, BufferRole, KernelFn, MemoryOrder, ParamKind, Place, Stmt, check_kernel,
};
use unpopped::types::Ty;

use crate::ops::{Em, Family, lo};
use crate::skeleton::{element_type, header};
use crate::{KernelSeams, lower_body};

/// A typed refusal of the kernel layer: `UnsupportedKernel` carrying `what`.
fn refuse(kernel: &KernelFn, what: impl std::fmt::Display) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("spirv backend: {what} (kernel `{}`)", kernel.name),
    }
}

/// The variant name of a statement: the `Debug` text up to the first non-identifier character.
fn stmt_name(s: &Stmt) -> String {
    let text = format!("{s:?}");
    text.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Everything the statement walk needs: the expression emitter and the module-level ids.
struct Kc<'k> {
    kernel: &'k KernelFn,
    em: RefCell<Em>,
    u32_t: Word,
    vec3: Word,
    local_id: Word,
    group_id: Word,
    /// One storage-buffer variable per parameter.
    bufs: Vec<Word>,
    /// One workgroup variable per allocation.
    allocs: Vec<Word>,
    /// One function-scope variable per local.
    locals: Vec<Word>,
    block: [u32; 3],
    grid: [u32; 3],
    /// `(merge, continue)` labels of the enclosing loops, innermost last.
    loops: RefCell<Vec<(Word, Word)>>,
}

/// The checks that decide whether this slice can lower `kernel`, before any id is allocated.
fn check_scope(kernel: &KernelFn) -> Result<(Family, [u32; 3], [u32; 3]), LowerError> {
    check_kernel(kernel).map_err(|e| LowerError::UnsupportedKernel {
        detail: format!("check_kernel refused kernel `{}`: {e:?}", kernel.name),
    })?;
    let fam = match Family::of(kernel.compute) {
        Some(f @ (Family::I32 | Family::U32)) => f,
        _ => {
            return Err(refuse(
                kernel,
                format!(
                    "v1 lowers integer kernels only; compute dtype is {:?}",
                    kernel.compute
                ),
            ));
        }
    };
    let Some(launch) = kernel.launch.as_ref() else {
        return Err(refuse(
            kernel,
            "a kernel without a launch shape has no workgroup size",
        ));
    };
    if launch.grid_dims.is_some() {
        return Err(refuse(kernel, "a parameter-dependent grid is not lowered"));
    }
    check_allocs(kernel, fam)?;
    check_params_and_locals(kernel, fam)?;
    Ok((fam, launch.block, launch.grid))
}

/// Every allocation must be a static shared array of the compute dtype.
fn check_allocs(kernel: &KernelFn, fam: Family) -> Result<(), LowerError> {
    for a in &kernel.allocs {
        match (a.space, a.extent) {
            (MemorySpace::Shared, Extent::Static(_)) if Family::of(a.elem) == Some(fam) => {}
            (MemorySpace::Shared, Extent::Param(_)) => {
                return Err(refuse(
                    kernel,
                    format!(
                        "allocation `{}` has a dynamic extent: SPIR-V needs a specialization constant, fixed per pipeline (row 8a, not yet lowered)",
                        a.name
                    ),
                ));
            }
            _ => {
                return Err(refuse(
                    kernel,
                    format!(
                        "allocation `{}` is not a static shared array of the compute dtype",
                        a.name
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Every parameter must be a buffer of the compute dtype, every local a scalar of it.
fn check_params_and_locals(kernel: &KernelFn, fam: Family) -> Result<(), LowerError> {
    for p in &kernel.params {
        match &p.kind {
            ParamKind::Buffer { elem, .. } if Family::of(*elem) == Some(fam) => {}
            ParamKind::Buffer { elem, .. } => {
                return Err(refuse(
                    kernel,
                    format!(
                        "buffer `{}` has element {elem:?}, not the compute dtype",
                        p.name
                    ),
                ));
            }
            _ => {
                return Err(refuse(
                    kernel,
                    format!("parameter `{}` is not a buffer", p.name),
                ));
            }
        }
    }
    for (i, l) in kernel.locals.iter().enumerate() {
        if l.lanes != 1 || Family::of(l.elem) != Some(fam) {
            return Err(refuse(
                kernel,
                format!("local {i} is not a scalar of the compute dtype"),
            ));
        }
    }
    Ok(())
}

/// An `Input` builtin variable of `vec3` type.
fn builtin_var(b: &mut rspirv::dr::Builder, vec3: Word, which: BuiltIn) -> Word {
    let ptr = b.type_pointer(None, StorageClass::Input, vec3);
    let v = b.variable(ptr, None, StorageClass::Input, None);
    b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(which)]);
    v
}

/// The storage-buffer variable of a parameter at binding `binding`.
fn buffer_var(b: &mut rspirv::dr::Builder, ptr: Word, binding: u32, role: BufferRole) -> Word {
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

impl Kc<'_> {
    fn t(&self) -> Word {
        self.em.borrow().t
    }

    /// A `u32` constant.
    fn u32_const(&self, v: u32) -> Word {
        self.em.borrow_mut().b.constant_bit32(self.u32_t, v)
    }

    /// The `axis` component of a builtin vector as the compute dtype.
    fn builtin_axis(&self, var: Word, axis: u8) -> Result<Spelling, LowerError> {
        let mut e = self.em.borrow_mut();
        let v = lo(e.b.load(self.vec3, None, var, None, []))?;
        let x =
            lo(e.b.composite_extract(self.u32_t, None, v, [u32::from(axis)]))?;
        let t = e.t;
        let x = if t == self.u32_t {
            x
        } else {
            lo(e.b.bitcast(t, None, x))?
        };
        Ok(Spelling::Spelled(x.to_string()))
    }

    /// The spelling of an execution id.
    fn exec(&self, id: ExecId) -> Result<Spelling, LowerError> {
        let axis = |a: u8| (a < 3).then_some(usize::from(a));
        match id {
            ExecId::ThreadIdx(a) if axis(a).is_some() => self.builtin_axis(self.local_id, a),
            ExecId::BlockIdx(a) if axis(a).is_some() => self.builtin_axis(self.group_id, a),
            ExecId::BlockDim(a) | ExecId::GridDim(a) if axis(a).is_some() => {
                let v = if matches!(id, ExecId::BlockDim(_)) {
                    self.block
                } else {
                    self.grid
                };
                let dtype = self.em.borrow().dtype;
                Ok(self
                    .em
                    .borrow_mut()
                    .constant_int(i64::from(v[usize::from(a)]), dtype))
            }
            other => Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Kernel,
                why: format!("execution id {other:?} is not lowered by the spirv backend v1"),
            })),
        }
    }

    /// The value of local `i`.
    fn local(&self, i: u32) -> Result<Spelling, LowerError> {
        let mut e = self.em.borrow_mut();
        let t = e.t;
        let var = self.locals[usize::try_from(i).expect("local id fits usize")];
        Ok(Spelling::Spelled(
            lo(e.b.load(t, None, var, None, []))?.to_string(),
        ))
    }

    /// Lower an expression to its result id.
    fn expr(&self, e: &unpopped::ir::ScalarExpr) -> Result<Word, LowerError> {
        let tys: Vec<Ty> = self.kernel.locals.iter().map(|l| l.ty()).collect();
        let (local, exec) = (|i| self.local(i), |id| self.exec(id));
        let dtype = self.em.borrow().dtype;
        let seams = KernelSeams {
            local: &local,
            exec: &exec,
            tys: &tys,
            dtype,
        };
        lower_body(&self.em, e, Some(&seams))
    }

    /// A pointer to the element `place` names.
    fn pointer(&self, place: &Place) -> Result<Word, LowerError> {
        let t = self.t();
        match place {
            Place::Logical { buffer, coords } if coords.len() == 1 => {
                let idx = self.expr(&coords[0])?;
                let zero = self.u32_const(0);
                let mut e = self.em.borrow_mut();
                let ptr_t = e.b.type_pointer(None, StorageClass::StorageBuffer, t);
                let base = self.bufs[usize::try_from(*buffer).expect("buffer id fits usize")];
                lo(e.b.access_chain(ptr_t, None, base, [zero, idx]))
            }
            Place::Alloc { alloc, index } => {
                let idx = self.expr(index)?;
                let mut e = self.em.borrow_mut();
                let ptr_t = e.b.type_pointer(None, StorageClass::Workgroup, t);
                let base = self.allocs[usize::try_from(*alloc).expect("alloc id fits usize")];
                lo(e.b.access_chain(ptr_t, None, base, [idx]))
            }
            other => Err(refuse(
                self.kernel,
                format!("place {other:?} is not lowered (v1: one-coordinate global or shared)"),
            )),
        }
    }

    fn store_local(&self, local: u32, value: Word) -> Result<(), LowerError> {
        let var = self.locals[usize::try_from(local).expect("local id fits usize")];
        lo(self.em.borrow_mut().b.store(var, value, None, []))
    }

    /// `OpControlBarrier` at `scope` ordering workgroup and buffer memory.
    fn barrier(&self, scope: SpvScope) -> Result<(), LowerError> {
        let sem = MemorySemantics::ACQUIRE_RELEASE
            | MemorySemantics::WORKGROUP_MEMORY
            | MemorySemantics::UNIFORM_MEMORY;
        let (s, m) = (self.u32_const(scope as u32), self.u32_const(scope as u32));
        let sem = self.u32_const(sem.bits());
        lo(self.em.borrow_mut().b.control_barrier(s, m, sem))
    }

    /// `OpMemoryBarrier` at `scope` with `order` (`SeqCst` collapses to `AcquireRelease`).
    fn fence(&self, scope: Scope, order: MemoryOrder) -> Result<(), LowerError> {
        let spv = match scope {
            Scope::Subgroup => SpvScope::Subgroup,
            Scope::Block => SpvScope::Workgroup,
            Scope::Device => SpvScope::Device,
            other => {
                return Err(refuse(
                    self.kernel,
                    format!("a fence at scope {other:?} is not lowered"),
                ));
            }
        };
        let ord = match order {
            MemoryOrder::Acquire => MemorySemantics::ACQUIRE,
            MemoryOrder::Release => MemorySemantics::RELEASE,
            MemoryOrder::AcqRel | MemoryOrder::SeqCst => MemorySemantics::ACQUIRE_RELEASE,
            other => {
                return Err(refuse(
                    self.kernel,
                    format!("a {other:?} fence is not lowered"),
                ));
            }
        };
        let sem = ord | MemorySemantics::WORKGROUP_MEMORY | MemorySemantics::UNIFORM_MEMORY;
        let (s, sem) = (self.u32_const(spv as u32), self.u32_const(sem.bits()));
        lo(self.em.borrow_mut().b.memory_barrier(s, sem))
    }

    fn stmt(&self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init: value } | Stmt::Assign { local, value } => {
                let v = self.expr(value)?;
                self.store_local(*local, v)
            }
            Stmt::Load { local, place } => {
                let ptr = self.pointer(place)?;
                let t = self.t();
                let v = lo(self.em.borrow_mut().b.load(t, None, ptr, None, []))?;
                self.store_local(*local, v)
            }
            Stmt::Store { place, value } => {
                let ptr = self.pointer(place)?;
                let v = self.expr(value)?;
                lo(self.em.borrow_mut().b.store(ptr, v, None, []))
            }
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            } => self.barrier(SpvScope::Workgroup),
            Stmt::Barrier {
                scope: Scope::Subgroup,
                kind: BarrierKind::Full,
            } => self.barrier(SpvScope::Subgroup),
            Stmt::Fence { scope, order } => self.fence(*scope, *order),
            Stmt::If { .. }
            | Stmt::For { .. }
            | Stmt::While { .. }
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Return => self.flow(s),
            other => Err(refuse(
                self.kernel,
                format!(
                    "`Stmt::{}` is not lowered (v1: Let, Load, Assign, Store, block/subgroup Barrier, Fence)",
                    stmt_name(other)
                ),
            )),
        }
    }
}

/// The module-level ids the function body refers to.
struct Module {
    b: rspirv::dr::Builder,
    fam: Family,
    glsl: Word,
    void: Word,
    fn_ty: Word,
    boolean: Word,
    u32_t: Word,
    t: Word,
    vec3: Word,
    local_id: Word,
    group_id: Word,
    bufs: Vec<Word>,
    allocs: Vec<Word>,
}

/// The storage buffers (one per parameter) and the workgroup arrays (one per allocation).
fn declare_storage(
    b: &mut rspirv::dr::Builder,
    kernel: &KernelFn,
    fam: Family,
    u32_t: Word,
    t: Word,
) -> (Vec<Word>, Vec<Word>) {
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
    let mut bufs = Vec::new();
    for (i, p) in kernel.params.iter().enumerate() {
        let ParamKind::Buffer { role, .. } = &p.kind else {
            unreachable!("check_scope admits buffers only")
        };
        let binding = u32::try_from(i).expect("binding fits u32");
        bufs.push(buffer_var(b, buf_ptr, binding, *role));
    }
    let mut allocs = Vec::new();
    for a in &kernel.allocs {
        let Extent::Static(n) = a.extent else {
            unreachable!("check_scope admits static extents only")
        };
        let len = b.constant_bit32(u32_t, n);
        let arr = b.type_array(t, len);
        let ptr = b.type_pointer(None, StorageClass::Workgroup, arr);
        allocs.push(b.variable(ptr, None, StorageClass::Workgroup, None));
    }
    (bufs, allocs)
}

/// Declare types, builtins, buffers and workgroup arrays for `kernel`.
fn declare(kernel: &KernelFn, fam: Family) -> Module {
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
    let (bufs, allocs) = declare_storage(&mut b, kernel, fam, u32_t, t);
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
    }
}

/// Open the entry function and declare one function-scope variable per local, then build the
/// statement emitter over the module.
fn begin_body<'k>(
    kernel: &'k KernelFn,
    mut m: Module,
    block: [u32; 3],
    grid: [u32; 3],
) -> Result<(Kc<'k>, Word), LowerError> {
    let zero_u32 = m.b.constant_bit32(m.u32_t, 0);
    let main =
        lo(m.b.begin_function(m.void, None, FunctionControl::NONE, m.fn_ty))?;
    lo(m.b.begin_block(None))?;
    let fn_ptr = m.b.type_pointer(None, StorageClass::Function, m.t);
    let locals: Vec<Word> = kernel
        .locals
        .iter()
        .map(|_| m.b.variable(fn_ptr, None, StorageClass::Function, None))
        .collect();
    let em = RefCell::new(Em {
        b: m.b,
        fam: m.fam,
        dtype: kernel.compute,
        glsl: m.glsl,
        t: m.t,
        boolean: m.boolean,
        zero_u32,
        inputs: Vec::new(),
        index: zero_u32,
        temps: HashMap::new(),
    });
    let kc = Kc {
        kernel,
        em,
        u32_t: m.u32_t,
        vec3: m.vec3,
        local_id: m.local_id,
        group_id: m.group_id,
        bufs: m.bufs,
        allocs: m.allocs,
        locals,
        block,
        grid,
        loops: RefCell::new(Vec::new()),
    };
    Ok((kc, main))
}

/// Lower `kernel` to a SPIR-V module, or say why this slice cannot.
pub(crate) fn lower(kernel: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    let (fam, block, grid) = check_scope(kernel)?;
    let (kc, main) = begin_body(kernel, declare(kernel, fam), block, grid)?;
    let ended = kc.block(&kernel.body)?;
    let (local_id, group_id) = (kc.local_id, kc.group_id);
    let mut e = kc.em.into_inner();
    if !ended {
        lo(e.b.ret())?;
    }
    lo(e.b.end_function())?;
    e.b.entry_point(
        ExecutionModel::GLCompute,
        main,
        &kernel.name,
        [local_id, group_id],
    );
    e.b.execution_mode(main, ExecutionMode::LocalSize, block);
    let module = e.b.module();
    let source = module.disassemble();
    let words = module.assemble();
    Ok(GeneratedKernel::new(kernel.name.clone(), source).with_words(words))
}
