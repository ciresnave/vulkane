// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checks that decide whether the walker can lower a kernel, before any text is written.

use unpopped::backend::LowerError;
use unpopped::exec::{Extent, MemorySpace};
use unpopped::kernel::{BufferRole, KernelFn, ParamKind, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

use crate::dialect::{Decl, Dialect, Layout};

/// What the checks learned about a kernel.
#[derive(Debug)]
pub(crate) struct Scope {
    pub(crate) signed: bool,
    pub(crate) block: [u32; 3],
    pub(crate) grid: [u32; 3],
    pub(crate) buffers: Vec<Decl>,
    pub(crate) shared: Vec<Decl>,
    pub(crate) subgroup: bool,
    pub(crate) backend: &'static str,
}

/// A typed refusal of the kernel layer carrying `what`.
pub(crate) fn refuse(backend: &str, kernel: &KernelFn, what: impl std::fmt::Display) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("{backend} backend: {what} (kernel `{}`)", kernel.name),
    }
}

impl Scope {
    /// The prologue's view of this scope.
    pub(crate) fn layout(&self, kernel: &KernelFn) -> Layout {
        Layout {
            name: kernel.name.clone(),
            signed: self.signed,
            block: self.block,
            buffers: self.buffers.clone(),
            shared: self.shared.clone(),
            locals: kernel.locals.len(),
            subgroup: self.subgroup,
        }
    }
}

/// The dtype of the kernel: `I32` or `U32`.
fn dtype(d: &dyn Dialect, kernel: &KernelFn) -> Result<bool, LowerError> {
    match kernel.compute {
        ElementKind::I32 => Ok(true),
        ElementKind::U32 => Ok(false),
        other => Err(refuse(
            d.name(),
            kernel,
            format!("v1 lowers integer kernels only; compute dtype is {other:?}"),
        )),
    }
}

/// One buffer parameter, or a decline.
fn buffer(d: &dyn Dialect, kernel: &KernelFn, i: usize) -> Result<Decl, LowerError> {
    let p = &kernel.params[i];
    match &p.kind {
        ParamKind::Buffer { elem, role, rank }
            if *elem == kernel.compute && rank.is_none_or(|r| r == 1) =>
        {
            Ok(Decl {
                name: format!("b{i}"),
                binding: u32::try_from(i).expect("binding fits u32"),
                role: *role,
                len: 0,
            })
        }
        _ => Err(refuse(
            d.name(),
            kernel,
            format!(
                "parameter `{}` is not a flat buffer of the compute dtype",
                p.name
            ),
        )),
    }
}

/// One shared allocation, or a decline.
fn shared(d: &dyn Dialect, kernel: &KernelFn, i: usize) -> Result<Decl, LowerError> {
    let a = &kernel.allocs[i];
    match (a.space, a.extent) {
        (MemorySpace::Shared, Extent::Static(n)) if a.elem == kernel.compute => Ok(Decl {
            name: format!("a{i}"),
            binding: 0,
            role: BufferRole::ReadWrite,
            len: n,
        }),
        _ => Err(refuse(
            d.name(),
            kernel,
            format!(
                "allocation `{}` is not a static shared array of the compute dtype (a dynamic extent is not lowered by the text walker yet)",
                a.name
            ),
        )),
    }
}

/// Whether `place` is one the walker spells (one coordinate, or a shared element).
pub(crate) fn place_ok(place: &Place) -> bool {
    match place {
        Place::Logical { coords, .. } => coords.len() == 1,
        Place::Alloc { .. } => true,
        _ => false,
    }
}

/// Whether any statement (at any depth) is a subgroup barrier or fence.
fn uses_subgroup(block: &unpopped::kernel::Block) -> bool {
    use unpopped::exec::Scope as S;
    block.0.iter().any(|s| match s {
        Stmt::Barrier { scope, .. } | Stmt::Fence { scope, .. } => *scope == S::Subgroup,
        Stmt::If { then, els, .. } => uses_subgroup(then) || uses_subgroup(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => uses_subgroup(body),
        _ => false,
    })
}

/// The launch shape: present, with a static grid.
fn launch<'k>(
    d: &dyn Dialect,
    kernel: &'k KernelFn,
) -> Result<&'k unpopped::exec::LaunchShape, LowerError> {
    let Some(launch) = kernel.launch.as_ref() else {
        return Err(refuse(
            d.name(),
            kernel,
            "a kernel without a launch shape has no workgroup size",
        ));
    };
    if launch.grid_dims.is_some() {
        return Err(refuse(
            d.name(),
            kernel,
            "a parameter-dependent grid is not lowered",
        ));
    }
    Ok(launch)
}

/// Every local must be a scalar of the compute dtype.
fn locals(d: &dyn Dialect, kernel: &KernelFn) -> Result<(), LowerError> {
    match kernel
        .locals
        .iter()
        .position(|l| l.lanes != 1 || l.elem != kernel.compute)
    {
        Some(i) => Err(refuse(
            d.name(),
            kernel,
            format!("local {i} is not a scalar of the compute dtype"),
        )),
        None => Ok(()),
    }
}

/// Run every check; see the crate docs for the scope.
pub(crate) fn check(d: &dyn Dialect, kernel: &KernelFn) -> Result<Scope, LowerError> {
    check_kernel(kernel).map_err(|e| LowerError::UnsupportedKernel {
        detail: format!("check_kernel refused kernel `{}`: {e:?}", kernel.name),
    })?;
    let signed = dtype(d, kernel)?;
    let launch = launch(d, kernel)?;
    locals(d, kernel)?;
    let buffers = (0..kernel.params.len())
        .map(|i| buffer(d, kernel, i))
        .collect::<Result<_, _>>()?;
    let shared = (0..kernel.allocs.len())
        .map(|i| shared(d, kernel, i))
        .collect::<Result<_, _>>()?;
    Ok(Scope {
        signed,
        block: launch.block,
        grid: launch.grid,
        buffers,
        shared,
        subgroup: uses_subgroup(&kernel.body),
        backend: d.name(),
    })
}
