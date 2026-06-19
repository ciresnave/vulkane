//! Derive macros for the `vulkane` Vulkan bindings crate.
//!
//! # `#[derive(Vertex)]`
//!
//! Automatically generates `VertexInputBinding` and
//! `VertexInputAttribute` descriptors from a `#[repr(C)]` struct,
//! so you never have to manually compute strides, offsets, or format
//! enums again.
//!
//! ```ignore
//! use vulkane_derive::Vertex;
//!
//! #[derive(Vertex, Clone, Copy)]
//! #[repr(C)]
//! struct MyVertex {
//!     position: [f32; 3],  // R32G32B32_SFLOAT, location 0
//!     normal:   [f32; 3],  // R32G32B32_SFLOAT, location 1
//!     uv:       [f32; 2],  // R32G32_SFLOAT,    location 2
//! }
//!
//! // Use in pipeline builder:
//! let bindings = [MyVertex::binding(0)];
//! let attributes = MyVertex::attributes(0);
//! builder.vertex_input(&bindings, &attributes)
//! ```
//!
//! ## Supported field types
//!
//! | Rust type | Vulkan format |
//! |-----------|---------------|
//! | `f32` | `R32_SFLOAT` |
//! | `[f32; 2]` | `R32G32_SFLOAT` |
//! | `[f32; 3]` | `R32G32B32_SFLOAT` |
//! | `[f32; 4]` | `R32G32B32A32_SFLOAT` |
//! | `u32` | `R32_UINT` |
//! | `[u32; 2]` | `R32G32_UINT` |
//! | `[u32; 3]` | `R32G32B32_UINT` |
//! | `[u32; 4]` | `R32G32B32A32_UINT` |
//! | `i32` | `R32_SINT` |
//! | `[i32; 2]` | `R32G32_SINT` |
//! | `[i32; 3]` | `R32G32B32_SINT` |
//! | `[i32; 4]` | `R32G32B32A32_SINT` (not yet added as Format constant) |
//! | `[u8; 4]` | `R8G8B8A8_UINT` |
//! | `u16` | `R16_UINT` |
//! | `i16` | `R16_SINT` |

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

/// Derive the `Vertex` trait, generating `binding()` and `attributes()`
/// methods that produce the `VertexInputBinding` and
/// `VertexInputAttribute` arrays needed by
/// `GraphicsPipelineBuilder::vertex_input`.
#[proc_macro_derive(Vertex)]
pub fn derive_vertex(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return syn::Error::new_spanned(name, "Vertex derive requires named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "Vertex derive only works on structs")
                .to_compile_error()
                .into();
        }
    };

    let mut attribute_exprs = Vec::new();
    for (location, field) in fields.iter().enumerate() {
        let field_name = field.ident.as_ref().unwrap();
        let format_expr = type_to_format(&field.ty);
        let format_expr = match format_expr {
            Some(expr) => expr,
            None => {
                return syn::Error::new_spanned(
                    &field.ty,
                    format!(
                        "Vertex derive: unsupported field type for `{}`. \
                         Supported: f32, [f32; 2..4], u32, [u32; 2..4], i32, [i32; 2..3], [u8; 4], u16, i16",
                        field_name
                    ),
                )
                .to_compile_error()
                .into();
            }
        };
        let location = location as u32;
        attribute_exprs.push(quote! {
            ::vulkane::safe::VertexInputAttribute {
                location: #location,
                binding,
                format: #format_expr,
                offset: ::core::mem::offset_of!(#name, #field_name) as u32,
            }
        });
    }

    let attr_count = attribute_exprs.len();

    let expanded = quote! {
        impl #name {
            /// Returns the [`VertexInputBinding`](::vulkane::safe::VertexInputBinding)
            /// for this vertex type at the given binding number.
            pub fn binding(binding: u32) -> ::vulkane::safe::VertexInputBinding {
                ::vulkane::safe::VertexInputBinding {
                    binding,
                    stride: ::core::mem::size_of::<#name>() as u32,
                    input_rate: ::vulkane::safe::InputRate::VERTEX,
                }
            }

            /// Returns the [`VertexInputBinding`](::vulkane::safe::VertexInputBinding)
            /// for this type used as per-instance data at the given binding number.
            pub fn instance_binding(binding: u32) -> ::vulkane::safe::VertexInputBinding {
                ::vulkane::safe::VertexInputBinding {
                    binding,
                    stride: ::core::mem::size_of::<#name>() as u32,
                    input_rate: ::vulkane::safe::InputRate::INSTANCE,
                }
            }

            /// Returns the [`VertexInputAttribute`](::vulkane::safe::VertexInputAttribute)
            /// array for this vertex type at the given binding number.
            /// Locations are assigned sequentially starting from 0.
            pub fn attributes(binding: u32) -> [::vulkane::safe::VertexInputAttribute; #attr_count] {
                [#(#attribute_exprs),*]
            }
        }
    };

    expanded.into()
}

/// Map a Rust field type to a `vulkane::safe::Format` constant expression.
fn type_to_format(ty: &Type) -> Option<proc_macro2::TokenStream> {
    let ty_str = quote!(#ty).to_string().replace(' ', "");
    match ty_str.as_str() {
        "f32" => Some(quote!(::vulkane::safe::Format::R32_SFLOAT)),
        "[f32;2]" => Some(quote!(::vulkane::safe::Format::R32G32_SFLOAT)),
        "[f32;3]" => Some(quote!(::vulkane::safe::Format::R32G32B32_SFLOAT)),
        "[f32;4]" => Some(quote!(::vulkane::safe::Format::R32G32B32A32_SFLOAT)),
        "u32" => Some(quote!(::vulkane::safe::Format::R32_UINT)),
        "[u32;2]" => Some(quote!(::vulkane::safe::Format::R32G32_UINT)),
        "[u32;3]" => Some(quote!(::vulkane::safe::Format::R32G32B32_UINT)),
        "[u32;4]" => Some(quote!(::vulkane::safe::Format::R32G32B32A32_UINT)),
        "i32" => Some(quote!(::vulkane::safe::Format::R32_SINT)),
        "[i32;2]" => Some(quote!(::vulkane::safe::Format::R32G32_SINT)),
        "[i32;3]" => Some(quote!(::vulkane::safe::Format::R32G32B32_SFLOAT)), // No R32G32B32_SINT yet
        "[u8;4]" => Some(quote!(::vulkane::safe::Format::R8G8B8A8_UINT)),
        "u16" => Some(quote!(::vulkane::safe::Format::R16_UINT)),
        "i16" => Some(quote!(::vulkane::safe::Format::R16_SINT)),
        _ => None,
    }
}
