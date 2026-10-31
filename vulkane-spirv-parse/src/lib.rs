// SPDX-License-Identifier: MIT OR Apache-2.0
//! SPIR-V module model and reflection for Vulkane.
//!
//! [`Module::parse`] decodes a binary SPIR-V module (a `&[u32]`) into a typed model built on
//! [`rspirv`]; [`Module::to_words`] encodes it back. Reflection ([`Module::entry_points`],
//! [`Module::bindings`], [`Module::capabilities`]) reads the module without validating it:
//! a parsed module can still be invalid SPIR-V, so run `spirv-val` before trusting one.

use std::collections::HashMap;

use rspirv::binary::Assemble;
use rspirv::dr;
use rspirv::spirv::{self, Decoration, ExecutionMode, ExecutionModel, Op, StorageClass};

/// Why a module could not be decoded.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParseError {
    /// The words are not a well-formed SPIR-V binary.
    #[error("malformed SPIR-V: {0}")]
    Malformed(String),
}

/// A decoded SPIR-V module.
#[derive(Debug, Clone)]
pub struct Module {
    inner: dr::Module,
    /// Header word 2 (the tool that produced the module); `rspirv` re-encodes its own value.
    generator: u32,
}

/// One `OpEntryPoint` with the facts that belong to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    /// The execution model (`GLCompute`, `Vertex`, ...).
    pub model: ExecutionModel,
    /// The entry point's name.
    pub name: String,
    /// The id of the function it enters.
    pub function: u32,
    /// `LocalSize` execution mode (`x, y, z`), when the module declares one literally.
    pub local_size: Option<[u32; 3]>,
}

/// A resource variable decorated with a binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// The variable's result id.
    pub variable: u32,
    /// `DescriptorSet` decoration; `None` when the module does not decorate one.
    pub set: Option<u32>,
    /// `Binding` decoration.
    pub binding: u32,
    /// The variable's storage class (`StorageBuffer`, `Uniform`, `UniformConstant`, ...).
    pub storage_class: StorageClass,
    /// The variable's `OpName`, when the module carries debug names.
    pub name: Option<String>,
}

impl Module {
    /// Decode a binary module.
    pub fn parse(words: &[u32]) -> Result<Self, ParseError> {
        let generator = words.get(2).copied().unwrap_or(0);
        dr::load_words(words)
            .map(|inner| Self { inner, generator })
            .map_err(|e| ParseError::Malformed(e.to_string()))
    }

    /// Encode back to words. For a module produced by a standard compiler this reproduces the
    /// input exactly; the corpus test checks that over every module it is given.
    pub fn to_words(&self) -> Vec<u32> {
        let mut w = self.inner.assemble();
        w[2] = self.generator;
        w
    }

    /// The underlying `rspirv` model, for callers that need the full instruction stream.
    pub fn as_rspirv(&self) -> &dr::Module {
        &self.inner
    }

    /// The declared capabilities, in declaration order.
    pub fn capabilities(&self) -> Vec<spirv::Capability> {
        self.inner
            .capabilities
            .iter()
            .filter_map(|i| match i.operands.first() {
                Some(dr::Operand::Capability(c)) => Some(*c),
                _ => None,
            })
            .collect()
    }

    /// Every entry point, with its `LocalSize` when declared as a literal.
    pub fn entry_points(&self) -> Vec<EntryPoint> {
        self.inner
            .entry_points
            .iter()
            .filter_map(|ep| {
                let (model, function, name) = match ep.operands.as_slice() {
                    [
                        dr::Operand::ExecutionModel(m),
                        dr::Operand::IdRef(f),
                        dr::Operand::LiteralString(n),
                        ..,
                    ] => (*m, *f, n.clone()),
                    _ => return None,
                };
                let local_size =
                    self.inner
                        .execution_modes
                        .iter()
                        .find_map(|em| match em.operands.as_slice() {
                            [
                                dr::Operand::IdRef(f),
                                dr::Operand::ExecutionMode(ExecutionMode::LocalSize),
                                dr::Operand::LiteralBit32(x),
                                dr::Operand::LiteralBit32(y),
                                dr::Operand::LiteralBit32(z),
                            ] if *f == function => Some([*x, *y, *z]),
                            _ => None,
                        });
                Some(EntryPoint {
                    model,
                    name,
                    function,
                    local_size,
                })
            })
            .collect()
    }

    /// Variables carrying a `Binding` decoration, ordered by (set, binding, id).
    pub fn bindings(&self) -> Vec<Binding> {
        let mut set: HashMap<u32, u32> = HashMap::new();
        let mut bind: HashMap<u32, u32> = HashMap::new();
        for d in &self.inner.annotations {
            if d.class.opcode != Op::Decorate {
                continue;
            }
            if let [
                dr::Operand::IdRef(id),
                dr::Operand::Decoration(dec),
                dr::Operand::LiteralBit32(v),
            ] = d.operands.as_slice()
            {
                match dec {
                    Decoration::DescriptorSet => {
                        set.insert(*id, *v);
                    }
                    Decoration::Binding => {
                        bind.insert(*id, *v);
                    }
                    _ => {}
                }
            }
        }
        let names = self.names();
        let mut out: Vec<Binding> = self
            .inner
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::Variable)
            .filter_map(|i| {
                let id = i.result_id?;
                let binding = *bind.get(&id)?;
                let storage_class = match i.operands.first() {
                    Some(dr::Operand::StorageClass(s)) => *s,
                    _ => return None,
                };
                Some(Binding {
                    variable: id,
                    set: set.get(&id).copied(),
                    binding,
                    storage_class,
                    name: names.get(&id).cloned(),
                })
            })
            .collect();
        out.sort_by_key(|b| (b.set, b.binding, b.variable));
        out
    }

    fn names(&self) -> HashMap<u32, String> {
        self.inner
            .debug_names
            .iter()
            .filter_map(|i| match i.operands.as_slice() {
                [dr::Operand::IdRef(id), dr::Operand::LiteralString(n)] => Some((*id, n.clone())),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage() {
        assert!(Module::parse(&[1, 2, 3]).is_err());
        assert!(Module::parse(&[]).is_err());
    }
}
