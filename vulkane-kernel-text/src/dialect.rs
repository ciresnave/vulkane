// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a target language supplies to the walker.

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{ArithOp, BinaryOp};
use unpopped::kernel::{BufferRole, MemoryOrder};

/// A buffer or shared array the prologue declares.
#[derive(Clone, Debug)]
pub struct Decl {
    /// The identifier the body uses (`b0`, `a0`).
    pub name: String,
    /// The binding slot of a buffer; unused for a shared array.
    pub binding: u32,
    /// The access of a buffer; `ReadWrite` for a shared array.
    pub role: BufferRole,
    /// The element count of a shared array; `0` for a buffer.
    pub len: u32,
}

/// Everything the prologue needs.
#[derive(Clone, Debug)]
pub struct Layout {
    /// The kernel's name.
    pub name: String,
    /// `true` for an `I32` kernel, `false` for `U32`.
    pub signed: bool,
    /// The workgroup size.
    pub block: [u32; 3],
    /// The storage buffers, in parameter order.
    pub buffers: Vec<Decl>,
    /// The workgroup arrays, in allocation order.
    pub shared: Vec<Decl>,
    /// How many locals the body declares (`l0..`).
    pub locals: usize,
    /// Whether any statement is a subgroup barrier or fence (a GLSL extension line).
    pub subgroup: bool,
}

/// The spellings of one target language.
///
/// A method that cannot spell something returns `None` (or an `Err` text), and the walker turns that
/// into a typed decline naming it; nothing is approximated.
pub trait Dialect {
    /// The backend name used in decline texts (`glsl`, `wgsl`).
    fn name(&self) -> &'static str;
    /// The header, buffer and shared declarations and the opening of the entry function, with the
    /// locals declared (one line each) at the top of the entry.
    fn prologue(&self, layout: &Layout) -> String;
    /// The closing of the entry function.
    fn epilogue(&self) -> &'static str;
    /// An integer literal in the compute dtype.
    fn lit(&self, v: i64, signed: bool) -> String;
    /// The element `idx` of the buffer or shared array `name`.
    fn index(&self, name: &str, idx: &str, is_buffer: bool) -> String;
    /// The spelling of an execution id, or `None` when the dialect has none.
    fn exec(&self, id: ExecId, signed: bool) -> Option<String>;
    /// A binary op on two integers, or `None` when there is no spelling.
    fn binary(&self, op: BinaryOp, a: &str, b: &str, signed: bool) -> Option<String>;
    /// `a + b`, `a - b`, `a * b` (wrapping); `None` for a division.
    fn arith(&self, op: ArithOp, a: &str, b: &str) -> Option<String> {
        let sym = match op {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            _ => return None,
        };
        Some(format!("({a} {sym} {b})"))
    }
    /// `cond != 0 ? a : b`.
    fn select(&self, cond: &str, a: &str, b: &str, signed: bool) -> String;
    /// A boolean spelling of `value != 0`.
    fn truthy(&self, value: &str, signed: bool) -> String {
        format!("({value} != {})", self.lit(0, signed))
    }
    /// A hoisted temporary declaration line (no indentation, no newline).
    fn temp(&self, name: &str, rhs: &str, signed: bool) -> String;
    /// The `For` counter's bound and step constants, declared once before the loop.
    fn bound(&self, name: &str, rhs: &str, signed: bool) -> String;
    /// A statement that synchronizes `scope` (and orders memory), or `None` when there is none.
    fn barrier(&self, scope: Scope) -> Option<String>;
    /// A memory fence at `scope`, or `None` when the dialect has none without a barrier.
    fn fence(&self, scope: Scope, order: MemoryOrder) -> Option<String>;
}
