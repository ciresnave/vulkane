// SPDX-License-Identifier: MIT OR Apache-2.0
//! Structured control flow for `Backend::lower_kernel` (ROADMAP row 8g): `If`, `For`, `While`,
//! `Break`, `Continue` and `Return`.
//!
//! SPIR-V wants every selection and loop to name its merge (and a loop its continue) block, so each
//! IR construct maps to a fixed block pattern:
//!
//! - `If`: `OpSelectionMerge` + `OpBranchConditional` to a then and an else block, both branching to
//!   the merge block unless they already ended in a `Break`/`Continue`/`Return`.
//! - `While`: header (`OpLoopMerge merge continue`) -> check (`OpBranchConditional body merge`) ->
//!   body -> continue block (branch back to the header) -> merge.
//! - `For`: the same, with the counter in a function variable: the bounds are evaluated once before
//!   the header (their ids dominate the loop), the check compares the counter with `end`, and the
//!   continue block adds `step`. **A zero step leaves the loop at once** (no pass runs): the IR says
//!   it traps, and a SPIR-V invocation cannot trap. A negative step is meaningful at `I32` only.
//! - `Break` branches to the innermost loop's merge block, `Continue` to its continue block,
//!   `Return` is `OpReturn` (legal from inside a loop).
//!
//! A statement after a terminator in the same block is dead and is not emitted. The unroll hint is
//! non-semantic and ignored. Barrier uniformity (a barrier under thread-divergent control) is
//! `check_kernel`'s `BarrierDivergent`, which this backend runs first, so such a kernel is a typed
//! decline carrying the checker's answer, never a silently moved barrier.

use rspirv::spirv::{LoopControl, SelectionControl, Word};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, Stmt};

use super::{Kc, refuse};
use crate::ops::{Family, lo};
use unpopped::backend::LowerError;

/// What keeps a loop running: a counter against its bound, or a condition.
enum Head<'a> {
    /// `(local, end, step)` of a `For`.
    Counted(u32, Word, Word),
    /// A `While`'s condition.
    Cond(&'a ScalarExpr),
}

impl Kc<'_> {
    /// Lower a block; `true` when it ended in a terminator (`Break`, `Continue`, `Return`).
    pub(super) fn block(&self, block: &Block) -> Result<bool, LowerError> {
        for s in &block.0 {
            self.stmt(s)?;
            if matches!(s, Stmt::Break | Stmt::Continue | Stmt::Return) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn id(&self) -> Word {
        self.em.borrow_mut().b.id()
    }

    fn branch(&self, to: Word) -> Result<(), LowerError> {
        lo(self.em.borrow_mut().b.branch(to))
    }

    fn begin(&self, label: Word) -> Result<(), LowerError> {
        lo(self.em.borrow_mut().b.begin_block(Some(label))).map(drop)
    }

    /// `cond != 0` as a boolean id.
    fn truthy(&self, cond: &ScalarExpr) -> Result<Word, LowerError> {
        let v = self.expr(cond)?;
        let mut e = self.em.borrow_mut();
        let zero = e.iconst(0);
        let bl = e.boolean;
        lo(e.b.i_not_equal(bl, None, v, zero))
    }

    /// Lower `body`, then branch to `to` unless it ended in a terminator.
    fn arm(&self, body: &Block, to: Word) -> Result<(), LowerError> {
        if !self.block(body)? {
            self.branch(to)?;
        }
        Ok(())
    }

    pub(super) fn flow(&self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::If { cond, then, els } => self.lower_if(cond, then, els),
            Stmt::While { cond, body } => self.lower_loop(&Head::Cond(cond), body),
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                ..
            } => self.lower_for(*local, [start, end, step], body),
            Stmt::Break | Stmt::Continue => self.jump(matches!(s, Stmt::Break)),
            _ => lo(self.em.borrow_mut().b.ret()),
        }
    }

    /// `Break` (to the innermost loop's merge block) or `Continue` (to its continue block).
    fn jump(&self, is_break: bool) -> Result<(), LowerError> {
        let &(merge, cont) = self
            .loops
            .borrow()
            .last()
            .ok_or_else(|| refuse(self.kernel, "Break/Continue outside a loop"))?;
        self.branch(if is_break { merge } else { cont })
    }

    /// A counted loop: the bounds are evaluated once, before the header.
    fn lower_for(
        &self,
        local: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        let (start, end, step) = (self.expr(start)?, self.expr(end)?, self.expr(step)?);
        self.store_local(local, start)?;
        self.lower_loop(&Head::Counted(local, end, step), body)
    }

    fn lower_if(&self, cond: &ScalarExpr, then: &Block, els: &Block) -> Result<(), LowerError> {
        let c = self.truthy(cond)?;
        let (t, f, merge) = (self.id(), self.id(), self.id());
        {
            let mut e = self.em.borrow_mut();
            lo(e.b.selection_merge(merge, SelectionControl::NONE))?;
            lo(e.b.branch_conditional(c, t, f, []))?;
        }
        self.begin(t)?;
        self.arm(then, merge)?;
        self.begin(f)?;
        self.arm(els, merge)?;
        self.begin(merge)
    }

    /// The loop test for a counted loop: `i < end` for a positive step, `i > end` for a negative
    /// one (`I32` only); a zero step is false.
    fn counted_test(&self, local: u32, end: Word, step: Word) -> Result<Word, LowerError> {
        let var = self.locals[usize::try_from(local).expect("local id fits usize")];
        let (t, fam) = {
            let e = self.em.borrow();
            (e.t, e.fam)
        };
        let i = lo(self.em.borrow_mut().b.load(t, None, var, None, []))?;
        if fam == Family::U32 {
            self.unsigned_test(i, end, step)
        } else {
            let up = self.directed(i, end, step, true)?;
            let down = self.directed(i, end, step, false)?;
            let mut e = self.em.borrow_mut();
            let bl = e.boolean;
            lo(e.b.logical_or(bl, None, up, down))
        }
    }

    /// `step != 0 && i < end`: an unsigned step is never negative.
    fn unsigned_test(&self, i: Word, end: Word, step: Word) -> Result<Word, LowerError> {
        let mut e = self.em.borrow_mut();
        let (bl, zero) = (e.boolean, e.iconst(0));
        let moving = lo(e.b.i_not_equal(bl, None, step, zero))?;
        let below = lo(e.b.u_less_than(bl, None, i, end))?;
        lo(e.b.logical_and(bl, None, moving, below))
    }

    /// `step > 0 && i < end` when `up`, else `step < 0 && i > end` (signed).
    fn directed(&self, i: Word, end: Word, step: Word, up: bool) -> Result<Word, LowerError> {
        let mut e = self.em.borrow_mut();
        let (bl, zero) = (e.boolean, e.iconst(0));
        let (sign, order) = if up {
            (
                e.b.s_greater_than(bl, None, step, zero),
                e.b.s_less_than(bl, None, i, end),
            )
        } else {
            (
                e.b.s_less_than(bl, None, step, zero),
                e.b.s_greater_than(bl, None, i, end),
            )
        };
        let (sign, order) = (lo(sign)?, lo(order)?);
        lo(e.b.logical_and(bl, None, sign, order))
    }

    /// The loop's header (`OpLoopMerge`) and the start of its check block.
    fn open_loop(&self, [header, check, cont, merge]: [Word; 4]) -> Result<(), LowerError> {
        self.branch(header)?;
        self.begin(header)?;
        lo(self
            .em
            .borrow_mut()
            .b
            .loop_merge(merge, cont, LoopControl::NONE, []))?;
        self.branch(check)?;
        self.begin(check)
    }

    /// The boolean that keeps the loop running.
    fn loop_test(&self, head: &Head<'_>) -> Result<Word, LowerError> {
        match head {
            Head::Counted(local, end, step) => self.counted_test(*local, *end, *step),
            Head::Cond(cond) => self.truthy(cond),
        }
    }

    /// The body, with `(merge, cont)` the targets of `Break` and `Continue` inside it.
    fn loop_body(&self, body: &Block, merge: Word, cont: Word) -> Result<(), LowerError> {
        self.loops.borrow_mut().push((merge, cont));
        let ended = self.arm(body, cont);
        self.loops.borrow_mut().pop();
        ended
    }

    fn lower_loop(&self, head: &Head<'_>, body: &Block) -> Result<(), LowerError> {
        let [header, check, body_l, cont, merge] = [(); 5].map(|()| self.id());
        self.open_loop([header, check, cont, merge])?;
        let c = self.loop_test(head)?;
        lo(self
            .em
            .borrow_mut()
            .b
            .branch_conditional(c, body_l, merge, []))?;
        self.begin(body_l)?;
        self.loop_body(body, merge, cont)?;
        self.begin(cont)?;
        if let Head::Counted(local, _, step) = head {
            self.advance(*local, *step)?;
        }
        self.branch(header)?;
        self.begin(merge)
    }

    /// `local += step`.
    fn advance(&self, local: u32, step: Word) -> Result<(), LowerError> {
        let var = self.locals[usize::try_from(local).expect("local id fits usize")];
        let mut e = self.em.borrow_mut();
        let t = e.t;
        let i = lo(e.b.load(t, None, var, None, []))?;
        let next = lo(e.b.i_add(t, None, i, step))?;
        lo(e.b.store(var, next, None, []))
    }
}
