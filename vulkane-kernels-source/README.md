# vulkane-kernels-source

GPU kernel sources (Slang and GLSL) and the script that compiles them to SPIR-V.

Moved from Fuel's `fuel-kernels-source` (Fuel `e3ff3427`) with its git history, including the
history from before the files were renamed out of `fuel-graph-executor/src/shaders/`.
PTX/cubin kernels are not part of this crate.

Compile: `kernels/compile.sh [--out-dir DIR] [files...]`; needs `slangc` and `glslangValidator`
(Vulkan SDK). Default output is `spv/` next to `kernels/` (git-ignored).

Third-party origins: `THIRD-PARTY-NOTICES.md`.
