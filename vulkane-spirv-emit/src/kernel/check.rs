// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checks that decide whether this slice can lower a kernel, before any id is allocated.

use unpopped::backend::LowerError;
use unpopped::exec::{Extent, MemorySpace};
use unpopped::kernel::{KernelFn, Param, ParamKind, check_kernel};

use super::refuse;
use crate::ops::Family;

/// Whether this slice can lower `kernel`: the dtype family, launch block and grid, or a decline.
pub(super) fn check_scope(kernel: &KernelFn) -> Result<(Family, [u32; 3], [u32; 3]), LowerError> {
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

/// The index of the scalar parameter that declares `local`, when its type is the compute dtype.
pub(super) fn extent_param(kernel: &KernelFn, local: u32, fam: Family) -> Option<usize> {
    kernel.params.iter().position(|p| {
        matches!(&p.kind, ParamKind::Scalar { ty, local: l }
            if *l == local && Family::of(*ty) == Some(fam))
    })
}

/// Every allocation must be a shared array of the compute dtype whose extent is static or is a
/// scalar parameter of the compute dtype (lowered as a specialization constant).
pub(super) fn check_allocs(kernel: &KernelFn, fam: Family) -> Result<(), LowerError> {
    for a in &kernel.allocs {
        match (a.space, a.extent) {
            (MemorySpace::Shared, Extent::Static(_)) if Family::of(a.elem) == Some(fam) => {}
            (MemorySpace::Shared, Extent::Param(l))
                if Family::of(a.elem) == Some(fam) && extent_param(kernel, l, fam).is_some() => {}
            (MemorySpace::Shared, Extent::Param(l)) => {
                return Err(refuse(
                    kernel,
                    format!(
                        "allocation `{}` has a dynamic extent from local {l}, which is no scalar parameter of the compute dtype",
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

/// Whether some shared allocation takes its extent from the scalar parameter declaring `local`.
fn sizes_array(kernel: &KernelFn, local: u32) -> bool {
    kernel
        .allocs
        .iter()
        .any(|a| a.extent == Extent::Param(local))
}

/// One parameter: a buffer of the compute dtype, or an extent scalar of it.
fn check_param(kernel: &KernelFn, p: &Param, fam: Family) -> Result<(), LowerError> {
    match &p.kind {
        ParamKind::Buffer { elem, .. } if Family::of(*elem) == Some(fam) => Ok(()),
        ParamKind::Buffer { elem, .. } => Err(refuse(
            kernel,
            format!(
                "buffer `{}` has element {elem:?}, not the compute dtype",
                p.name
            ),
        )),
        ParamKind::Scalar { local, .. }
            if sizes_array(kernel, *local) && extent_param(kernel, *local, fam).is_some() =>
        {
            Ok(())
        }
        _ => Err(refuse(
            kernel,
            format!(
                "parameter `{}` is not lowered: only a buffer, or a scalar of the compute dtype that sizes a shared array (a specialization constant), is",
                p.name
            ),
        )),
    }
}

/// Every parameter must be a buffer of the compute dtype or an extent scalar, every local a scalar of the dtype.
pub(super) fn check_params_and_locals(kernel: &KernelFn, fam: Family) -> Result<(), LowerError> {
    for p in &kernel.params {
        check_param(kernel, p, fam)?;
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
