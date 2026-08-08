# Contributing to Vulkane

Thank you for your interest in contributing to Vulkane! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Install development dependencies:
   - Rust 1.85 or later (the crate is edition 2024)
   - Vulkan SDK 1.4.316 or later
   - CMake 3.20 or later and a C++ toolchain — only needed for the `shaderc`
     and `slang` features, which build their compilers from source. The
     examples themselves are pure Rust and need neither.

2. Clone and build:

   ```bash
   git clone https://github.com/ciresnave/vulkane.git
   cd vulkane
   cargo build
   ```

3. Run tests:

   ```bash
   cargo test                          # core suite
   cargo test --features naga          # GLSL front-end
   cargo test --features kiss-target   # KISS `vulkan:` token derivation
   ```

   `cargo test --all-features` also works but pulls in `shaderc` and `slang`,
   so it needs the C++ toolchain above and takes considerably longer.

   Note that much of the suite enumerates and runs work on a **real Vulkan
   device** rather than mocking one. Expect it to fail on a machine with no
   Vulkan-capable GPU or no installed ICD, and avoid running two GPU-touching
   suites concurrently.

## Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` with the project's configuration
- Run `clippy` with all lints enabled
- Maintain comprehensive documentation

## Documentation Requirements

1. Every public API must have:
   - Clear purpose and usage examples
   - Parameter descriptions
   - Safety section for unsafe functions
   - Error conditions and handling

2. Include doc tests demonstrating usage:

```rust
/// Allocate `size` bytes of device memory from the given memory type.
///
/// # Examples
///
/// ```no_run
/// use vulkane::safe::{Device, DeviceMemory};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let device: Device = unimplemented!();
/// # let memory_type_index = 0;
/// let memory = DeviceMemory::allocate(&device, 1024, memory_type_index)?;
/// # Ok(())
/// # }
/// ```
```

Pick the type index with `PhysicalDevice::find_memory_type`, and prefer
`DeviceMemory::allocate_with` when you need to pass a `MemoryAllocateInfo`
(for a dedicated allocation or an exportable handle, say).

## Testing Requirements

1. Unit Tests:
   - Test each public API
   - Cover error cases
   - Mock external dependencies

2. Integration Tests:
   - Test realistic usage patterns
   - Verify component interactions
   - Test platform-specific features

3. Performance Tests:
   - Benchmark critical operations
   - Compare against baselines
   - Document performance characteristics

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes:
   - Follow code style guidelines
   - Add tests
   - Update documentation
4. Submit a pull request:
   - Describe the changes
   - Link related issues
   - Add test results

## Development Workflow

1. Check existing issues and PRs
2. Create an issue for new features
3. Write failing tests first
4. Implement the feature
5. Document thoroughly
6. Submit PR for review

## Safety Guidelines

1. Mark unsafe functions appropriately:

   ```rust
   /// Create a buffer from a raw handle
   ///
   /// # Safety
   ///
   /// The caller must ensure:
   /// - The handle is valid
   /// - The handle was created by the same device
   /// - The handle is not used elsewhere
   pub unsafe fn from_raw(
       device: &Arc<Device>,
       handle: VkBuffer,
   ) -> Buffer {
       // Implementation
   }
   ```

2. Document all safety requirements thoroughly
3. Validate inputs wherever possible
4. Use safe abstractions by default

## Performance Guidelines

1. Profile before optimizing
2. Document performance characteristics
3. Add benchmarks for changes
4. Compare against baselines

## Release Process

1. Update version numbers
2. Update changelog
3. Run full test suite
4. Build documentation
5. Create release tag
6. Publish to crates.io

## Getting Help

- Open an issue at <https://github.com/ciresnave/vulkane/issues>
- Check existing issues and discussions first — Vulkan questions recur
- Consult the API docs at <https://docs.rs/vulkane>

## License

By contributing, you agree to license your code under either:

- Apache License, Version 2.0
- MIT License

at your option.
