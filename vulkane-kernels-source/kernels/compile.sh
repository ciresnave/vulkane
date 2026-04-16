#!/usr/bin/env bash
# Compile every shader source (*.wgsl, *.glsl, *.slang) in this
# directory to SPIR-V, writing the output to ../shaders_spirv/.
#
# Toolchain: requires the Vulkan SDK (for slangc + glslangValidator)
# and naga-cli (cargo install naga-cli) on PATH.
#
# SPIR-V files are committed to the repo — contributors who don't
# touch shader sources never need any of these tools installed.
#
# Usage:
#   cd fuel-graph-executor/src/shaders
#   ./compile.sh            # compile all
#   ./compile.sh rope.wgsl  # compile a single file
#
# Conventions:
#   .wgsl   -> naga (Vulkan 1.1 / SPIR-V 1.3)
#   .glsl   -> glslangValidator, compute stage, Vulkan 1.1
#   .slang  -> slangc, SPIR-V, entry point 'main', profile glsl_450
#
# GLSL files that need a specific entry or non-default SPIR-V version
# can override via a sidecar comment line `// compile: <flags>` at
# the top of the file.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$(cd "$SRC_DIR/.." && pwd)/shaders_spirv"
mkdir -p "$OUT_DIR"

# Resolve tools. slangc and glslangValidator come from the Vulkan SDK.
SLANGC="${SLANGC:-${VULKAN_SDK:-}/Bin/slangc}"
GLSLANG="${GLSLANG:-${VULKAN_SDK:-}/Bin/glslangValidator}"
NAGA="${NAGA:-naga}"

# On non-Windows the SDK usually puts binaries in bin/ (lowercase).
if [[ ! -x "$SLANGC" && -n "${VULKAN_SDK:-}" ]]; then
  [[ -x "$VULKAN_SDK/bin/slangc" ]] && SLANGC="$VULKAN_SDK/bin/slangc"
fi
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
  echo "  wgsl  $base.wgsl -> shaders_spirv/$base.spv"
  "$NAGA" "$src" "$out"
}

compile_glsl() {
  local src="$1"
  local base; base="$(basename "$src" .glsl)"
  local out="$OUT_DIR/$base.spv"
  echo "  glsl  $base.glsl -> shaders_spirv/$base.spv"
  "$GLSLANG" -V -S comp --target-env vulkan1.1 "$src" -o "$out" >/dev/null
}

compile_slang() {
  local src="$1"
  local base; base="$(basename "$src" .slang)"
  local out="$OUT_DIR/$base.spv"
  echo "  slang $base.slang -> shaders_spirv/$base.spv"
  "$SLANGC" "$src" -target spirv -profile glsl_450 -entry main -o "$out"
}

compile_one() {
  local f="$1"
  case "$f" in
    *.wgsl)  compile_wgsl  "$f" ;;
    *.glsl)  compile_glsl  "$f" ;;
    *.slang) compile_slang "$f" ;;
    *) echo "warning: unknown shader type: $f" >&2 ;;
  esac
}

if (( $# > 0 )); then
  # Check only the tools we actually need.
  for arg in "$@"; do
    case "$arg" in
      *.wgsl)  check_tool "$NAGA" "naga-cli" ;;
      *.glsl)  check_tool "$GLSLANG" "glslangValidator" ;;
      *.slang) check_tool "$SLANGC" "slangc" ;;
    esac
  done
  cd "$SRC_DIR"
  for f in "$@"; do compile_one "$f"; done
else
  # Compile everything. Need all three tools.
  check_tool "$NAGA" "naga-cli" || true
  check_tool "$GLSLANG" "glslangValidator" || true
  check_tool "$SLANGC" "slangc" || true
  cd "$SRC_DIR"
  shopt -s nullglob
  for f in *.wgsl; do compile_one "$f"; done
  for f in *.glsl; do compile_one "$f"; done
  for f in *.slang; do compile_one "$f"; done
fi

echo "Done. Output: $OUT_DIR"
