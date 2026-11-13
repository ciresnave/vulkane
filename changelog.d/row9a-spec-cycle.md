kind: Fixed the `parser::tree_parser` <-> `vulkan_spec_parser` module cycle in `vulkan_gen` (ROADMAP row 9a)
---
`VulkanSpecification` moves to `parser::vk_types`, which both modules already read; `vulkan_spec_parser` re-exports it, so `vulkan_gen::VulkanSpecification` and `vulkan_gen::vulkan_spec_parser::VulkanSpecification` are unchanged. Removes the edge `tree_parser.rs:8` that closed the cycle. Verified: `cargo clippy -D warnings`, `cargo test -p vulkan_gen` (68 + 3 pass), docs with `-D warnings`. The rest of row 9a (vulkan_gen documentation and visibility) is a separate PR.
