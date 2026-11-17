// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statements this emitter refuses **by name** because lowering them to a hardware operation on non-NVIDIA
//! hardware risks infringing NVIDIA's patents (board 434 C, extended by CireSnave 2026-11-16).
//!
//! CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if
//! non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for
//! other manufacturers' hardware know about that and we consider ourselves not infringing."* and, for the second
//! patent, *"I think we treat that patent like the others that are Nvidia-only. Add a note and specifically notify
//! our non-Nvidia-emitter builders about the potential for infringement and not to."*
//!
//! - NVIDIA US11803380B2 (a synchronization barrier with an arrival counter, a phase and a hardware copy operator):
//!   `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` (completes on an mbarrier) and
//!   `Tcgen05Commit` (arrives on an mbarrier).
//! - NVIDIA US12204897B2 and its continuation US12710961B2 (a wait on a count of pending matrix-multiply
//!   operations): `WgmmaWait`.
//!
//! **Do not lower any of these to a hardware operation in this crate.** The walk below matches every [`Stmt`]
//! variant explicitly and has no wildcard arm, so a new statement fails to compile here until someone decides which
//! side of this list it is on.

use unpopped::backend::LowerError;
use unpopped::kernel::{Block, KernelFn, Stmt};

/// The reason text every refusal carries, so a caller and a test can recognise it.
pub(crate) const DECLINED: &str = "declined: possible exposure, board 434 C extended";

/// The name of the first declined statement in `block`, looking inside `If`, `For` and `While` at any depth.
fn find(block: &Block) -> Option<&'static str> {
    block.0.iter().find_map(find_stmt)
}

fn find_stmt(stmt: &Stmt) -> Option<&'static str> {
    match stmt {
        Stmt::MbarInit { .. } => Some("MbarInit"),
        Stmt::MbarArrive { .. } => Some("MbarArrive"),
        Stmt::MbarCompleteTx { .. } => Some("MbarCompleteTx"),
        Stmt::MbarWait { .. } => Some("MbarWait"),
        Stmt::CopyBulk { .. } => Some("CopyBulk"),
        Stmt::Tcgen05Commit { .. } => Some("Tcgen05Commit"),
        Stmt::WgmmaWait { .. } => Some("WgmmaWait"),
        Stmt::If { then, els, .. } => find(then).or_else(|| find(els)),
        Stmt::For { body, .. } | Stmt::While { body, .. } => find(body),
        // Every other statement is not declined by name here; the kernel layer is not lowered by this emitter at all
        // yet, so the default refusal applies. Listed explicitly, never `_`. (`CopyAsync`, `AsyncCommit`, `AsyncWait` and
        // `Collective` are the board 434 C declines of ROADMAP rows 8c/8i; naming them here is a separate change.)
        Stmt::Let { .. }
        | Stmt::Load { .. }
        | Stmt::Assign { .. }
        | Stmt::Store { .. }
        | Stmt::MaskedLoad { .. }
        | Stmt::MaskedStore { .. }
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Return
        | Stmt::Barrier { .. }
        | Stmt::AtomicRmw { .. }
        | Stmt::AtomicCas { .. }
        | Stmt::MulticastLoadReduce { .. }
        | Stmt::GridDepWait
        | Stmt::GridDepLaunchDependents
        | Stmt::Fence { .. }
        | Stmt::Collective { .. }
        | Stmt::CollectiveTuple { .. }
        | Stmt::Mma { .. }
        | Stmt::MatrixLoad { .. }
        | Stmt::MatrixStore { .. }
        | Stmt::Wgmma { .. }
        | Stmt::WgmmaCommit
        | Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::CopyAsync { .. }
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::Call { .. }
        | Stmt::CallTuple { .. }
        | Stmt::ExternCall { .. } => None,
    }
}

/// `Backend::lower_kernel` for an emitter that does not lower kernel-layer kernels yet: the typed refusal
/// [`LowerError::UnsupportedKernel`], and when the kernel holds a declined statement the refusal NAMES it.
pub(crate) fn refuse_kernel(backend: &str, kernel: &KernelFn) -> LowerError {
    let detail = match find(&kernel.body) {
        Some(name) => format!(
            "backend `{backend}` refuses `Stmt::{name}` by name ({DECLINED}); it is not lowered to a hardware operation on non-NVIDIA hardware (kernel `{}`)",
            kernel.name
        ),
        None => format!(
            "backend `{backend}` does not lower kernel-layer kernels (kernel `{}`)",
            kernel.name
        ),
    };
    LowerError::UnsupportedKernel { detail }
}
