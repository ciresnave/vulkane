#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Compile every shader source (*.wgsl, *.glsl) in this
# directory to SPIR-V, writing the output to --out-dir DIR (default ../spv/).
#
# Toolchain: requires the Vulkan SDK (for glslangValidator)
# and naga-cli (cargo install naga-cli) on PATH.
#
# Compiled SPIR-V is not kept in this crate; each consumer builds (and, if it
# wants, commits) its own from these sources.
#
# Usage:
#   ./compile.sh                         # compile all into ../spv/
#   ./compile.sh --out-dir out/spv       # compile all into out/spv/
#   ./compile.sh matvec.glsl             # compile a single file
#   ./compile.sh --out-dir out matvec.glsl
#
# Conventions:
#   .wgsl   -> naga (Vulkan 1.1 / SPIR-V 1.3)
#   .glsl   -> glslangValidator, compute stage, Vulkan 1.1
#
# GLSL files that need a specific entry or non-default SPIR-V version
# can override via a sidecar comment line `// compile: <flags>` at
# the top of the file.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
# Output directory: --out-dir DIR, else spv/ next to kernels/ (git-ignored).
# A relative DIR is resolved against the caller's working directory, before
# this script changes into kernels/.
OUT_DIR="$SRC_DIR/../spv"
if [[ "${1:-}" == "--out-dir" ]]; then
  [[ -n "${2:-}" ]] || { echo "error: --out-dir needs a directory" >&2; exit 2; }
  OUT_DIR="$2"
  shift 2
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# Resolve tools. glslangValidator comes from the Vulkan SDK.
GLSLANG="${GLSLANG:-${VULKAN_SDK:-}/Bin/glslangValidator}"
NAGA="${NAGA:-naga}"

# On non-Windows the SDK usually puts binaries in bin/ (lowercase).
if [[ ! -x "$GLSLANG" && -n "${VULKAN_SDK:-}" ]]; then
  [[ -x "$VULKAN_SDK/bin/glslangValidator" ]] && GLSLANG="$VULKAN_SDK/bin/glslangValidator"
fi

check_tool() {
  if ! command -v "$1" >/dev/null 2>&1 && [[ ! -x "$1" ]]; then
    echo "error: $2 not found (looked for '$1')" >&2
    echo "       Install the Vulkan SDK and/or run 'cargo install naga-cli'." >&2
    return 1
  fi
}

compile_wgsl() {
  local src="$1"
  local base; base="$(basename "$src" .wgsl)"
  local out="$OUT_DIR/$base.spv"
  echo "  wgsl  $base.wgsl -> $OUT_DIR/$base.spv"
  "$NAGA" "$src" "$out"
}

compile_glsl() {
  local src="$1"
  local base; base="$(basename "$src" .glsl)"
  local out="$OUT_DIR/$base.spv"
  echo "  glsl  $base.glsl -> $OUT_DIR/$base.spv"
  "$GLSLANG" -V -S comp --target-env vulkan1.1 "$src" -o "$out" >/dev/null
}

compile_one() {
  local f="$1"
  case "$f" in
    *.wgsl)  compile_wgsl  "$f" ;;
    *.glsl)  compile_glsl  "$f" ;;
    *) echo "warning: unknown shader type: $f" >&2 ;;
  esac
}

if (( $# > 0 )); then
  # Check only the tools we actually need.
  for arg in "$@"; do
    case "$arg" in
      *.wgsl)  check_tool "$NAGA" "naga-cli" ;;
      *.glsl)  check_tool "$GLSLANG" "glslangValidator" ;;
    esac
  done
  cd "$SRC_DIR"
  for f in "$@"; do compile_one "$f"; done
else
  # Compile everything. Need both tools.
  check_tool "$NAGA" "naga-cli" || true
  check_tool "$GLSLANG" "glslangValidator" || true
  cd "$SRC_DIR"
  shopt -s nullglob
  for f in *.wgsl; do compile_one "$f"; done
  for f in *.glsl; do compile_one "$f"; done
fi

echo "Done. Output: $OUT_DIR"
