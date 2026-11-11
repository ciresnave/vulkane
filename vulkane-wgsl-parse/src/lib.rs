// SPDX-License-Identifier: MIT OR Apache-2.0
//! A WGSL front end: parse WGSL source and reflect its entry points and resource bindings.
//!
//! Vulkane owns WGSL (ROADMAP 3d). The parser is `wgsl-parse` (decision W1,
//! `docs/design/wgsl-parser-emitter.md`); this crate hides it behind its own owned public types so a
//! parser upgrade is not a breaking change here. It is a Rust-AST crate, not an Unpopped `Frontend`
//! (no tree-sitter tree, no `lift_*`).
//!
//! * [`parse_wgsl`] - source to a [`Module`], or a [`ParseError`] carrying the rendered diagnostic.
//! * [`Module::entry_points`] - `@compute` / `@vertex` / `@fragment` functions, with `@workgroup_size`.
//! * [`Module::bindings`] - module-scope `var` declarations carrying `@group` and `@binding`.

use std::fmt;
use wgsl_parse::span::Spanned;
use wgsl_parse::syntax::{
    Attribute, Declaration, DeclarationKind, Expression, ExpressionNode, GlobalDeclaration,
    LiteralExpression,
};

/// The pipeline stage of an entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `@compute`
    Compute,
    /// `@vertex`
    Vertex,
    /// `@fragment`
    Fragment,
}

/// One `@workgroup_size` dimension: a literal, or an expression this crate does not evaluate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    /// An integer literal.
    Literal(u32),
    /// Anything else (an `override` or `const` name); the source text of the expression.
    Expr(String),
}

/// An entry point function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// The function name.
    pub name: String,
    /// Its stage.
    pub stage: Stage,
    /// `@workgroup_size(x, y, z)`; absent dimensions are `None`. `None` overall for non-compute stages.
    pub workgroup_size: Option<[Option<Dim>; 3]>,
}

/// A module-scope resource: a `var` with `@group` and `@binding`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    /// The variable name.
    pub name: String,
    /// `@group(n)`
    pub group: u32,
    /// `@binding(n)`
    pub binding: u32,
    /// The declared type, as source text (e.g. `array<f32>`).
    pub ty: String,
    /// The address space and access mode as the parser spells them (Debug form), e.g.
    /// `Some("(Storage, Some(ReadWrite))")`; `None` for a handle (texture or sampler).
    pub space: Option<String>,
}

/// A parsed module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Module {
    /// Entry points in source order.
    pub entry_points: Vec<EntryPoint>,
    /// Bindings in source order.
    pub bindings: Vec<Binding>,
    /// Names of every function, entry points included, in source order.
    pub functions: Vec<String>,
}

/// A syntax error, or a binding attribute this crate cannot read as an integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// The rendered diagnostic, with the source location where the parser gave one.
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse WGSL source and reflect it.
///
/// # Errors
/// A [`ParseError`] for a syntax error, or for an `@group` / `@binding` that is not an integer
/// literal (this crate does not evaluate constants, and a guessed number would be a wrong answer).
pub fn parse_wgsl(source: &str) -> Result<Module, ParseError> {
    let unit = wgsl_parse::parse_str(source).map_err(|e| ParseError {
        message: e.with_source(source.into()).to_string(),
    })?;
    let mut module = Module::default();
    for node in &unit.global_declarations {
        match node.node() {
            GlobalDeclaration::Function(f) => {
                let name = f.ident.name().clone();
                module.functions.push(name.clone());
                if let Some(ep) = entry_point(&name, &f.attributes) {
                    module.entry_points.push(ep);
                }
            }
            GlobalDeclaration::Declaration(d) => {
                if let Some(b) = binding(d)? {
                    module.bindings.push(b);
                }
            }
            _ => {}
        }
    }
    Ok(module)
}

fn entry_point(name: &str, attrs: &[Spanned<Attribute>]) -> Option<EntryPoint> {
    let mut stage = None;
    let mut size = None;
    for a in attrs {
        match a.node() {
            Attribute::Compute => stage = Some(Stage::Compute),
            Attribute::Vertex => stage = Some(Stage::Vertex),
            Attribute::Fragment => stage = Some(Stage::Fragment),
            Attribute::WorkgroupSize(w) => {
                size = Some([
                    Some(dim(&w.x)),
                    w.y.as_ref().map(dim),
                    w.z.as_ref().map(dim),
                ]);
            }
            _ => {}
        }
    }
    stage.map(|stage| EntryPoint {
        name: name.to_owned(),
        stage,
        workgroup_size: if stage == Stage::Compute { size } else { None },
    })
}

fn literal_u32(e: &ExpressionNode) -> Option<u32> {
    match e.node() {
        Expression::Literal(LiteralExpression::AbstractInt(v)) => u32::try_from(*v).ok(),
        Expression::Literal(LiteralExpression::U32(v)) => Some(*v),
        Expression::Literal(LiteralExpression::I32(v)) => u32::try_from(*v).ok(),
        _ => None,
    }
}

fn dim(e: &ExpressionNode) -> Dim {
    literal_u32(e).map_or_else(|| Dim::Expr(e.to_string()), Dim::Literal)
}

fn binding(d: &Declaration) -> Result<Option<Binding>, ParseError> {
    let (mut group, mut bind) = (None, None);
    for a in &d.attributes {
        let (slot, what, e) = match a.node() {
            Attribute::Group(e) => (&mut group, "@group", e),
            Attribute::Binding(e) => (&mut bind, "@binding", e),
            _ => continue,
        };
        let v = literal_u32(e).ok_or_else(|| ParseError {
            message: format!(
                "{what} on `{}` is not an integer literal: `{e}`",
                d.ident.name()
            ),
        })?;
        *slot = Some(v);
    }
    let DeclarationKind::Var(space) = d.kind else {
        return Ok(None);
    };
    let (Some(group), Some(binding)) = (group, bind) else {
        return Ok(None);
    };
    Ok(Some(Binding {
        name: d.ident.name().clone(),
        group,
        binding,
        ty: d.ty.as_ref().map(ToString::to_string).unwrap_or_default(),
        space: space.map(|s| format!("{s:?}")),
    }))
}
