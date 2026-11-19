// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statements this emitter refuses **by name** because lowering them to a hardware operation on non-NVIDIA
//! hardware risks infringing NVIDIA's patents. Twelve names: the four of board 434 C plus `CollectiveTuple` (CireSnave 2026-11-13) and the
//! seven of its extension (CireSnave 2026-11-16).
//!
//! - Board 434 C, *"emitters may lower AsyncWait/CopyAsync/AsyncCommit/Collective for NVIDIA devices ... and decline by
//!   name for non-NVIDIA devices"* (CireSnave 2026-11-13, "434: C."): `CopyAsync`, `AsyncCommit`, `AsyncWait` (NVIDIA
//!   US11294713B2 and US12118382B2, an API that blocks a thread's instructions around an asynchronous shared-memory
//!   copy; `unpopped` `docs/PATENT-SWEEP.md` hit 22) and `Collective`, and `CollectiveTuple`. `CollectiveTuple` is the tuple form of `Collective`: 434 C read as covering it; not named in the
//!   ruling's list (pm-kernels 2026-11-17, PM no objection; rockfish, bonefish and fishstringer decline the family).
//!
//! CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if
//! non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for
//! other manufacturers' hardware know about that and we consider ourselves not infringing."* and, for the second
//! patent, *"I think we treat that patent like the others that are Nvidia-only. Add a note and specifically notify
//! our non-Nvidia-emitter builders about the potential for infringement and not to."*
//!
//! Board 434 C extended (2026-11-16):
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

/// The reason text the four board 434 C refusals carry, so a caller and a test can recognise it.
pub const DECLINED_434C: &str = "declined: possible exposure, board 434 C";
/// The reason text the seven board 434 C extended refusals carry.
pub const DECLINED_434C_EXT: &str = "declined: possible exposure, board 434 C extended";

/// The name of the first declined statement in `block` and the ruling text it carries, looking inside `If`, `For`
/// and `While` at any depth.
fn find(block: &Block) -> Option<Declined> {
    block.0.iter().find_map(find_stmt)
}

/// A declined statement's name and the ruling text its refusal carries.
type Declined = (&'static str, &'static str);

fn find_stmt(stmt: &Stmt) -> Option<Declined> {
    declined_434c(stmt)
        .or_else(|| declined_extension(stmt))
        .or_else(|| nested(stmt))
}

/// The five of board 434 C. The wildcard is fine here: exhaustiveness is enforced by [`nested`].
fn declined_434c(stmt: &Stmt) -> Option<Declined> {
    let name = match stmt {
        Stmt::CopyAsync { .. } => "CopyAsync",
        Stmt::AsyncCommit => "AsyncCommit",
        Stmt::AsyncWait { .. } => "AsyncWait",
        Stmt::Collective { .. } => "Collective",
        Stmt::CollectiveTuple { .. } => "CollectiveTuple",
        _ => return None,
    };
    Some((name, DECLINED_434C))
}

/// The seven of the 2026-11-16 extension.
fn declined_extension(stmt: &Stmt) -> Option<Declined> {
    let name = match stmt {
        Stmt::MbarInit { .. } => "MbarInit",
        Stmt::MbarArrive { .. } => "MbarArrive",
        Stmt::MbarCompleteTx { .. } => "MbarCompleteTx",
        Stmt::MbarWait { .. } => "MbarWait",
        Stmt::CopyBulk { .. } => "CopyBulk",
        Stmt::Tcgen05Commit { .. } => "Tcgen05Commit",
        Stmt::WgmmaWait { .. } => "WgmmaWait",
        _ => return None,
    };
    Some((name, DECLINED_434C_EXT))
}

/// Look inside `If`, `For` and `While`. **This match has no wildcard arm on purpose**: it names every [`Stmt`] variant,
/// so a new statement in `unpopped` fails to compile here until someone decides which side of the declined list it is
/// on. The declined ones are listed (and answered `None`: they were already found by the two functions above).
fn nested(stmt: &Stmt) -> Option<Declined> {
    match stmt {
        Stmt::If { then, els, .. } => find(then).or_else(|| find(els)),
        Stmt::For { body, .. } | Stmt::While { body, .. } => find(body),
        // The declined statements (found above, and none holds a block).
        Stmt::CopyAsync { .. }
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::Collective { .. }
        | Stmt::CollectiveTuple { .. }
        | Stmt::MbarInit { .. }
        | Stmt::MbarArrive { .. }
        | Stmt::MbarCompleteTx { .. }
        | Stmt::MbarWait { .. }
        | Stmt::CopyBulk { .. }
        | Stmt::Tcgen05Commit { .. }
        | Stmt::WgmmaWait { .. }
        // Every other statement is not declined by name here.
        | Stmt::Let { .. }
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
        | Stmt::Mma { .. }
        | Stmt::MatrixLoad { .. }
        | Stmt::MatrixStore { .. }
        | Stmt::Wgmma { .. }
        | Stmt::WgmmaCommit
        | Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::Call { .. }
        | Stmt::CallTuple { .. }
        | Stmt::ExternCall { .. } => None,
    }
}

/// The by-name refusal of the first declined statement in `kernel` (at any depth), or `None`.
///
/// `backend` is the name used in the text (`glsl`, `wgsl`).
#[must_use]
pub fn declined_by_name(backend: &str, kernel: &KernelFn) -> Option<LowerError> {
    let (name, ruling) = find(&kernel.body)?;
    Some(LowerError::UnsupportedKernel {
        detail: format!(
            "backend `{backend}` refuses `Stmt::{name}` by name ({ruling}); it is not lowered to a hardware operation on non-NVIDIA hardware (kernel `{}`)",
            kernel.name
        ),
    })
}
