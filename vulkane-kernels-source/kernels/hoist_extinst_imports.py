#!/usr/bin/env python3
"""Post-process a SPIR-V file from slangc to hoist + dedupe
OpExtInstImport instructions that ended up inside function bodies
out to module scope.

Slang's `spirv_asm` block lets you emit arbitrary SPIR-V but does
NOT auto-hoist OpExtInstImport (it does hoist OpCapability and
OpExtension). When you write
    spirv_asm {
        %cl = OpExtInstImport "OpenCL.std";
        result:$$double = OpExtInst %cl 19 $x;
    }
the import lands inside the function — `spirv-val` then rejects it
with "OpExtInstImport cannot appear in the graph definitions
section". This script disassembles the SPV, moves all
OpExtInstImport lines up to module scope, dedupes by extension-set
name, rewrites references, and reassembles.

Usage:
    python hoist_extinst_imports.py path/to/file.spv

Modifies in place. Requires spirv-dis + spirv-as on PATH (Vulkan SDK).
"""

import re
import subprocess
import sys
from pathlib import Path


def hoist_imports(text: str) -> str:
    lines = text.splitlines()

    # Find every OpExtInstImport line + its identifier + extension name.
    # Slang emits e.g. `%cl = OpExtInstImport "OpenCL.std"` or
    # `%cl_0 = OpExtInstImport "OpenCL.std"` etc. The leading whitespace
    # varies (module-scope = column 16, function-body = indented).
    import_re = re.compile(r'^\s*(%\S+)\s*=\s*OpExtInstImport\s+"([^"]+)"\s*$')

    inside_function = False
    # Maps extension set name → canonical identifier (first one seen).
    # First-seen wins so module-scope imports (e.g. Slang's own
    # GLSL.std.450 import emitted at the top) keep their identifier.
    canonical: dict[str, str] = {}
    # Identifier rewrites: alias_id → canonical_id.
    alias: dict[str, str] = {}
    # Lines to delete (the in-function imports).
    delete_lines: set[int] = set()
    # New OpExtInstImport lines to add at module scope.
    new_imports: list[str] = []

    for idx, line in enumerate(lines):
        if line.lstrip().startswith("%") and " OpFunction " in line:
            inside_function = True
        if line.lstrip().startswith("OpFunctionEnd"):
            inside_function = False
            continue

        m = import_re.match(line)
        if not m:
            continue
        ident, ext_name = m.group(1), m.group(2)
        if ext_name in canonical:
            # Duplicate — alias to canonical, delete this line.
            alias[ident] = canonical[ext_name]
            delete_lines.add(idx)
        else:
            canonical[ext_name] = ident
            if inside_function:
                # First occurrence but it's in a function body — move
                # the line to module scope by recording it + deleting
                # the original.
                new_imports.append(f'         {ident} = OpExtInstImport "{ext_name}"')
                delete_lines.add(idx)
            # else: it's already at module scope; leave it.

    # Find insertion point for new imports: right after the first
    # existing OpExtInstImport at module scope (or after the last
    # OpCapability if no imports exist yet).
    insert_at: int | None = None
    for idx, line in enumerate(lines):
        if idx in delete_lines:
            continue
        if " OpExtInstImport " in line:
            insert_at = idx + 1
    if insert_at is None:
        # No module-scope imports yet — insert after the last
        # OpCapability or OpExtension instead.
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("OpCapability", "OpExtension")):
                insert_at = idx + 1
    if insert_at is None:
        # Fallback: put them right before OpMemoryModel.
        for idx, line in enumerate(lines):
            if line.lstrip().startswith("OpMemoryModel"):
                insert_at = idx
                break
    if insert_at is None:
        # Nothing recognizable — bail.
        raise RuntimeError(
            "hoist_extinst_imports: couldn't find a place to insert imports"
        )

    # Apply identifier rewrites first (so the inserted lines below
    # don't pick up rewrites that target their own canonical IDs).
    if alias:
        # Sort by length descending to avoid partial-prefix collisions
        # (e.g. %cl_10 should be rewritten before %cl_1).
        for old in sorted(alias.keys(), key=len, reverse=True):
            new = alias[old]
            # Word-boundary match: % is followed by [A-Za-z0-9_]+, so
            # we match the exact identifier with a trailing
            # non-identifier char.
            pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(old)
                             + r'(?![A-Za-z0-9_])')
            lines = [pat.sub(new, ln) for ln in lines]

    # Drop the deleted lines.
    lines = [ln for i, ln in enumerate(lines) if i not in delete_lines]

    # Recompute insertion point after deletions (line indices shifted).
    if new_imports:
        # Find insertion point in the now-mutated list. Use the same
        # "after last OpExtInstImport" rule.
        insert_at2: int | None = None
        for idx, line in enumerate(lines):
            if " OpExtInstImport " in line:
                insert_at2 = idx + 1
        if insert_at2 is None:
            for idx, line in enumerate(lines):
                stripped = line.lstrip()
                if stripped.startswith(("OpCapability", "OpExtension")):
                    insert_at2 = idx + 1
        if insert_at2 is None:
            for idx, line in enumerate(lines):
                if line.lstrip().startswith("OpMemoryModel"):
                    insert_at2 = idx
                    break
        if insert_at2 is None:
            raise RuntimeError(
                "hoist_extinst_imports: post-deletion insertion failed"
            )
        for offset, imp in enumerate(new_imports):
            lines.insert(insert_at2 + offset, imp)

    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: hoist_extinst_imports.py path/to/file.spv",
              file=sys.stderr)
        return 1
    spv = Path(sys.argv[1])
    if not spv.is_file():
        print(f"error: {spv} not found", file=sys.stderr)
        return 1

    dis = subprocess.run(
        ["spirv-dis", "--no-color", str(spv)],
        check=True, capture_output=True, text=True,
    )

    original = dis.stdout
    fixed = hoist_imports(original)

    # Skip the round-trip if the post-process didn't change anything —
    # spirv-as is non-trivial cost per shader (~50ms) and most kernels
    # don't use spirv_asm + OpExtInstImport at all.
    if fixed == original:
        return 0

    # Round-trip through spirv-as. Validate after.
    subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.1", "-o", str(spv), "-"],
        check=True, input=fixed, text=True, capture_output=True,
    )
    val = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.1", str(spv)],
        capture_output=True, text=True,
    )
    if val.returncode != 0:
        print("spirv-val failed after hoist:", val.stderr, file=sys.stderr)
        return val.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
