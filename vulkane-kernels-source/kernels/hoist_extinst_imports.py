#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
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


# Matches e.g. `%cl = OpExtInstImport "OpenCL.std"` or `%cl_0 = ...`. The leading
# whitespace varies (module-scope = column 16, function-body = indented).
IMPORT_RE = re.compile(r'^\s*(%\S+)\s*=\s*OpExtInstImport\s+"([^"]+)"\s*$')


def _function_state(line: str, inside: bool) -> bool:
    """Whether the next line is inside a function body, given this one."""
    stripped = line.lstrip()
    if stripped.startswith("%") and " OpFunction " in line:
        return True
    if stripped.startswith("OpFunctionEnd"):
        return False
    return inside


def _collect_imports(lines: list[str]):
    """Scan the disassembly. Returns (alias, delete_lines, new_imports):

    alias         identifier rewrites, alias_id -> canonical_id
    delete_lines  indices of the in-function / duplicate import lines to drop
    new_imports   module-scope lines to add for imports first seen in a function body
    """
    inside_function = False
    # Maps extension set name -> canonical identifier (first one seen).
    # First-seen wins so module-scope imports (e.g. Slang's own
    # GLSL.std.450 import emitted at the top) keep their identifier.
    canonical: dict[str, str] = {}
    alias: dict[str, str] = {}
    delete_lines: set[int] = set()
    new_imports: list[str] = []

    for idx, line in enumerate(lines):
        inside_function = _function_state(line, inside_function)
        m = IMPORT_RE.match(line)
        if not m:
            continue
        ident, ext_name = m.group(1), m.group(2)
        if ext_name in canonical:
            # Duplicate: alias to canonical, delete this line.
            alias[ident] = canonical[ext_name]
            delete_lines.add(idx)
        elif inside_function:
            # First occurrence but in a function body: move the line to
            # module scope by recording it + deleting the original.
            canonical[ext_name] = ident
            new_imports.append(f'         {ident} = OpExtInstImport "{ext_name}"')
            delete_lines.add(idx)
        else:
            # Already at module scope; leave it.
            canonical[ext_name] = ident
    return alias, delete_lines, new_imports


def _last_match(lines: list[str], pred, skip: set[int]) -> int | None:
    hits = [i for i, ln in enumerate(lines) if i not in skip and pred(ln)]
    return hits[-1] if hits else None


def _insertion_point(lines: list[str], skip: set[int]) -> int | None:
    """Where new module-scope imports go: right after the last existing
    OpExtInstImport (ignoring `skip`ped lines); else after the last
    OpCapability / OpExtension; else right before OpMemoryModel; else None."""
    last = _last_match(lines, lambda ln: " OpExtInstImport " in ln, skip)
    if last is None:
        last = _last_match(
            lines,
            lambda ln: ln.lstrip().startswith(("OpCapability", "OpExtension")),
            set(),
        )
    if last is not None:
        return last + 1
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("OpMemoryModel"):
            return idx
    return None


def _apply_aliases(lines: list[str], alias: dict[str, str]) -> list[str]:
    # Longest first, to avoid partial-prefix collisions (%cl_10 before %cl_1).
    for old in sorted(alias, key=len, reverse=True):
        # Word-boundary match: % is followed by [A-Za-z0-9_]+, so match the
        # exact identifier with a trailing non-identifier char.
        pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(old)
                         + r'(?![A-Za-z0-9_])')
        lines = [pat.sub(alias[old], ln) for ln in lines]
    return lines


def hoist_imports(text: str) -> str:
    lines = text.splitlines()
    alias, delete_lines, new_imports = _collect_imports(lines)

    # Fail early (even with nothing to add) if the module has no recognisable
    # place for imports.
    if _insertion_point(lines, delete_lines) is None:
        # Nothing recognizable: bail.
        raise RuntimeError(
            "hoist_extinst_imports: couldn't find a place to insert imports"
        )

    # Apply identifier rewrites first (so the inserted lines below
    # don't pick up rewrites that target their own canonical IDs).
    lines = _apply_aliases(lines, alias)

    # Drop the deleted lines.
    lines = [ln for i, ln in enumerate(lines) if i not in delete_lines]

    # Recompute the insertion point on the mutated list (indices shifted).
    if new_imports:
        insert_at = _insertion_point(lines, set())
        if insert_at is None:
            raise RuntimeError(
                "hoist_extinst_imports: post-deletion insertion failed"
            )
        for offset, imp in enumerate(new_imports):
            lines.insert(insert_at + offset, imp)

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
