// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statements and structured control flow.
//!
//! `If` and `While` map to the native statements; a `While` condition that needs hoisted temporaries
//! is written `while (true) { temps; if (!cond) { break; } body }` so they run every pass. `For` is a
//! native `for (init; test; update)` inside its own scope that declares the bounds once, so a
//! `continue` still runs the update. Statements after a `Break`, `Continue` or `Return` in a block
//! are dead and are not written.

use unpopped::backend::LowerError;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, KernelFn, Place, Stmt};

use crate::dialect::Dialect;
use crate::expr;
use crate::scope::{Scope, place_ok, refuse};

/// The text being written.
struct W<'a> {
    d: &'a dyn Dialect,
    kernel: &'a KernelFn,
    sc: &'a Scope,
    out: &'a mut String,
    depth: usize,
    loops: u32,
}

/// Write the body of `kernel` at one level of indentation.
pub(crate) fn emit_body(
    d: &dyn Dialect,
    kernel: &KernelFn,
    sc: &Scope,
    out: &mut String,
) -> Result<(), LowerError> {
    let mut w = W {
        d,
        kernel,
        sc,
        out,
        depth: 1,
        loops: 0,
    };
    w.block(&kernel.body).map(drop)
}

/// The variant name of a statement: the `Debug` text up to the first non-identifier character.
fn stmt_name(s: &Stmt) -> String {
    let text = format!("{s:?}");
    text.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

impl W<'_> {
    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// [`Self::line`] as a statement result.
    fn emit(&mut self, text: &str) -> Result<(), LowerError> {
        self.line(text);
        Ok(())
    }

    fn no(&self, what: impl std::fmt::Display) -> LowerError {
        refuse(self.sc.backend, self.kernel, what)
    }

    /// The root spelling of `e`, its hoisted declarations written first.
    fn expr(&mut self, e: &ScalarExpr) -> Result<String, LowerError> {
        let (prelude, root) = expr::lower(self.d, self.kernel, self.sc, e)?;
        for p in prelude.iter().filter(|p| !p.is_empty()) {
            self.line(p);
        }
        Ok(root)
    }

    /// Lower a block; `true` when it ended in a terminator.
    fn block(&mut self, block: &Block) -> Result<bool, LowerError> {
        for s in &block.0 {
            self.stmt(s)?;
            if matches!(s, Stmt::Break | Stmt::Continue | Stmt::Return) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `body` one level deeper.
    fn nested(&mut self, body: &Block) -> Result<(), LowerError> {
        self.depth += 1;
        let r = self.block(body);
        self.depth -= 1;
        r.map(drop)
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { .. } | Stmt::Assign { .. } | Stmt::Load { .. } | Stmt::Store { .. } => self.data(s),
            Stmt::Barrier { .. } | Stmt::Fence { .. } => self.sync(s),
            Stmt::If { .. } | Stmt::For { .. } | Stmt::While { .. } => self.control(s),
            Stmt::Break => self.emit("break;"),
            Stmt::Continue => self.emit("continue;"),
            Stmt::Return => self.emit("return;"),
            other => Err(self.no(format!(
                "`Stmt::{}` is not lowered (v1: Let, Load, Assign, Store, Barrier, Fence, control flow)",
                stmt_name(other)
            ))),
        }
    }

    /// The element `place` names, as an lvalue or rvalue.
    fn element(&mut self, place: &Place) -> Result<String, LowerError> {
        if !place_ok(place) {
            return Err(self.no(format!(
                "place {place:?} is not lowered (v1: one-coordinate buffer or shared element)"
            )));
        }
        let (name, idx, is_buffer) = match place {
            Place::Logical { buffer, coords } => (format!("b{buffer}"), &coords[0], true),
            Place::Alloc { alloc, index } => (format!("a{alloc}"), index, false),
            _ => unreachable!("place_ok admits two forms"),
        };
        let idx = self.expr(idx)?;
        Ok(self.d.index(&name, &idx, is_buffer))
    }

    fn data(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init: v } | Stmt::Assign { local, value: v } => {
                let v = self.expr(v)?;
                self.emit(&format!("l{local} = {v};"))
            }
            Stmt::Load { local, place } => {
                let e = self.element(place)?;
                self.emit(&format!("l{local} = {e};"))
            }
            Stmt::Store { place, value } => {
                let e = self.element(place)?;
                let v = self.expr(value)?;
                self.emit(&format!("{e} = {v};"))
            }
            _ => unreachable!("`stmt` routes only data statements here"),
        }
    }

    fn sync(&mut self, s: &Stmt) -> Result<(), LowerError> {
        let text = match s {
            Stmt::Barrier {
                scope,
                kind: BarrierKind::Full,
            } => self.d.barrier(*scope),
            Stmt::Fence { scope, order } => self.d.fence(*scope, *order),
            _ => None,
        };
        let Some(text) = text else {
            return Err(self.no(format!(
                "`Stmt::{}` at this scope or kind has no {} spelling",
                stmt_name(s),
                self.d.name()
            )));
        };
        self.emit(&text)
    }

    fn control(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::If { cond, then, els } => self.lower_if(cond, then, els),
            Stmt::While { cond, body } => self.lower_while(cond, body),
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                ..
            } => self.lower_for(*local, [start, end, step], body),
            _ => unreachable!("`stmt` routes only control statements here"),
        }
    }

    fn lower_if(&mut self, cond: &ScalarExpr, then: &Block, els: &Block) -> Result<(), LowerError> {
        let c = self.expr(cond)?;
        let t = self.d.truthy(&c, self.sc.signed);
        self.line(&format!("if {t} {{"));
        self.nested(then)?;
        if !els.0.is_empty() {
            self.line("} else {");
            self.nested(els)?;
        }
        self.emit("}")
    }

    fn lower_while(&mut self, cond: &ScalarExpr, body: &Block) -> Result<(), LowerError> {
        self.line("while (true) {");
        self.depth += 1;
        let c = self.expr(cond);
        let r = c.and_then(|c| {
            let t = self.d.truthy(&c, self.sc.signed);
            self.line(&format!("if (!{t}) {{ break; }}"));
            self.block(body).map(drop)
        });
        self.depth -= 1;
        r?;
        self.emit("}")
    }

    /// The loop test: `i < end` for a positive step, `i > end` for a negative one (`I32` only); a
    /// zero step is false.
    fn counted_test(&self, i: &str, e: &str, s: &str) -> String {
        let zero = self.d.lit(0, self.sc.signed);
        if self.sc.signed {
            format!("(({s} > {zero} && {i} < {e}) || ({s} < {zero} && {i} > {e}))")
        } else {
            format!("({s} != {zero} && {i} < {e})")
        }
    }

    fn lower_for(
        &mut self,
        local: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        let n = self.loops;
        self.loops += 1;
        self.line("{");
        self.depth += 1;
        let r = self.for_in_scope(local, n, [start, end, step], body);
        self.depth -= 1;
        r?;
        self.emit("}")
    }

    fn for_in_scope(
        &mut self,
        local: u32,
        n: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        let signed = self.sc.signed;
        let (start, end, step) = (self.expr(start)?, self.expr(end)?, self.expr(step)?);
        self.line(&self.d.bound(&format!("e{n}"), &end, signed));
        self.line(&self.d.bound(&format!("s{n}"), &step, signed));
        let (i, e, s) = (format!("l{local}"), format!("e{n}"), format!("s{n}"));
        let test = self.counted_test(&i, &e, &s);
        self.line(&format!(
            "for ({i} = {start}; {test}; {i} = ({i} + {s})) {{"
        ));
        self.nested(body)?;
        self.emit("}")
    }
}
