pub mod assembler;
pub mod generator_modules;
pub mod logging;
pub mod provisional;
pub mod type_integration;

/// Escape a comment line so rustdoc doesn't try to parse markdown / HTML.
///
/// Vulkan's vk.xml comments use prose conventions that collide with rustdoc:
///
/// - Asciidoc cross-references like `<<devsandqueues-lost-device>>` look like
///   invalid HTML tags to rustdoc and produce `rustdoc::invalid_html_tags`
///   warnings.
/// - Bracketed text like `BUFFER[_DYNAMIC]` looks like an intra-doc link
///   with no resolvable target and produces `rustdoc::broken_intra_doc_links`
///   warnings.
///
/// We escape `<`, `>`, `[`, and `]` to their HTML entity equivalents so they
/// render as literal characters in the generated docs without rustdoc trying
/// to interpret them.
pub fn sanitize_doc_line(line: &str) -> String {
    line.trim()
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('[', "&#91;")
        .replace(']', "&#93;")
}

pub use assembler::AssemblerError;

/// Convert a Vulkan camelCase identifier (e.g. `timelineSemaphore`) or
/// PascalCase type name (e.g. `VkPhysicalDeviceFeatures2`) into the
/// Rust `snake_case` convention (`timeline_semaphore`,
/// `vk_physical_device_features2`).
///
/// Behaviour:
///
/// - Inserts an underscore before every uppercase letter that follows
///   a lowercase letter or a digit.
/// - Inserts an underscore before an uppercase letter that is followed
///   by a lowercase letter when the preceding character is uppercase
///   (so `GPUList` becomes `gpu_list`, not `g_p_u_list`).
/// - Preserves runs of existing underscores, and collapses the result
///   to lowercase.
///
/// Used by the feature- and extension-generators to derive method
/// names from Vulkan field and extension identifiers.
pub fn camel_to_snake(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + name.len() / 3);
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_ascii_uppercase() {
            if i > 0 {
                let prev = chars[i - 1];
                let next = chars.get(i + 1).copied();
                let boundary = prev.is_ascii_lowercase()
                    || prev.is_ascii_digit()
                    || (prev.is_ascii_uppercase()
                        && matches!(next, Some(n) if n.is_ascii_lowercase()));
                if boundary && !out.ends_with('_') {
                    out.push('_');
                }
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod camel_to_snake_tests {
    use super::camel_to_snake;

    #[test]
    fn lowercase_passthrough() {
        assert_eq!(camel_to_snake("foo"), "foo");
    }

    #[test]
    fn single_boundary() {
        assert_eq!(camel_to_snake("timelineSemaphore"), "timeline_semaphore");
    }

    #[test]
    fn multiple_boundaries() {
        assert_eq!(
            camel_to_snake("shaderBufferFloat32Atomics"),
            "shader_buffer_float32_atomics"
        );
    }

    #[test]
    fn acronym_group() {
        // `GPU` followed by `List` — we want a single boundary between
        // the acronym and the next word, not four.
        assert_eq!(camel_to_snake("GPUList"), "gpu_list");
        assert_eq!(camel_to_snake("VkKHRSwapchain"), "vk_khr_swapchain");
    }

    #[test]
    fn digits_are_word_chars() {
        assert_eq!(camel_to_snake("vulkan12"), "vulkan12");
        assert_eq!(camel_to_snake("Vulkan12Features"), "vulkan12_features");
    }

    #[test]
    fn preexisting_underscores() {
        assert_eq!(camel_to_snake("VK_VERSION_1_0"), "vk_version_1_0");
    }
}

/// Error type for code generation operations
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("Assembler error: {0}")]
    Assembler(#[from] AssemblerError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Generation failed: {message}")]
    GenerationFailed { message: String },
}

pub type CodegenResult<T> = Result<T, CodegenError>;
