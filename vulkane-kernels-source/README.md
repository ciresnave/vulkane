# vulkane-kernels-source

GPU kernel sources in GLSL (27 files) and the script that compiles them to SPIR-V. The Slang kernels live in
FishStringer (`fishstringer-kernels-source`; FishStringer owns Slang, CireSnave 2026-11-06).

Moved from Fuel's `fuel-kernels-source` (Fuel `e3ff3427`) with its git history, including the
history from before the files were renamed out of `fuel-graph-executor/src/shaders/`.
PTX/cubin kernels are not part of this crate.

Compile: `kernels/compile.sh [--out-dir DIR] [files...]`; needs `glslangValidator`
(Vulkan SDK). Default output is `spv/` next to `kernels/` (git-ignored).

Third-party origins: `THIRD-PARTY-NOTICES.md`.
