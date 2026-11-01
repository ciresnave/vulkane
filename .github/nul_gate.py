#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""No tracked text file contains a NUL byte.

    python3 .github/nul_gate.py            # check
    python3 .github/nul_gate.py --self-test

WHY: a NUL in a source file is invisible in an editor and in most diffs, makes
git treat the file as binary (so review shows "Binary files differ"), and can
truncate a string a tool reads as C text. It arrives from a bad paste, an editor
saving UTF-16, or a tool writing a padded buffer.

HOW IT DECIDES WHAT IS TEXT: every tracked file is text unless its extension is
in BINARY below, each with a reason. The default is the strict one on purpose: a
new binary type fails the gate by name, and the fix is a decision on the record
rather than a silent skip. The population is enumerated from `git ls-files`
(never a disk walk, which sees other worktrees) and PRINTED, so a number can
be told from a different population of the same size.

IT REFUSES TO PASS on an empty or tiny population, and on a BINARY entry that
matches no tracked file (a stale entry protects nothing and reads like one that
does).  It never writes.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess  # nosec B404 - the tracked-file list comes from running git;
                   # there is no other source
import sys
import tempfile

#: Extensions of tracked files that are binary by nature, with the reason.
BINARY: dict[str, str] = {
    ".spv": "compiled SPIR-V words (little-endian u32, NULs are normal)",
    ".jpeg": "image",
}

#: Below this the listing is broken rather than the tree small. The tree holds
#: about 350 text files; this catches an empty or near-empty enumeration.
MINIMUM_TEXT_FILES = 100


def _git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess | None:
    """The one place git is run: fixed argv, shell=False. None if git is missing."""
    git = shutil.which("git")
    if git is None:
        print("FAIL: no `git` on PATH.", file=sys.stderr)
        return None
    # nosemgrep - argv[0] is the path resolved for the literal "git"; every other
    # element is a literal or this script's own repo root, and there is no shell.
    return subprocess.run(  # nosec B603 # nosemgrep
        [git, "-C", str(root), *args], capture_output=True, shell=False, check=False)


def _ls_files(root: pathlib.Path) -> list[str] | None:
    proc = _git(root, "ls-files", "-z")
    if proc is None:
        return None
    if proc.returncode != 0:
        print(f"FAIL: git ls-files: {proc.stderr.decode('utf-8', 'replace').strip()[:200]}",
              file=sys.stderr)
        return None
    return [n for n in proc.stdout.decode("utf-8", "replace").split(chr(0)) if n]


def _ext(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def _scan(root: pathlib.Path, text: list[str]) -> tuple[int, list[str]]:
    """(files with NUL, failure lines) over the text population."""
    nul, failures = 0, []
    needle = bytes([0])
    for n in text:
        try:
            data = (root / n).read_bytes()
        except OSError as err:
            failures.append(f"{n}: unreadable ({err})")
            continue
        count = data.count(needle)
        if count:
            nul += 1
            failures.append(f"{n}: {count} NUL byte(s), first at offset {data.index(needle)}")
    return nul, failures


def _population_failures(names: list[str], text: list[str]) -> list[str]:
    failures = [f"BINARY lists {e!r} but no tracked file has it (stale entry)"
                for e in BINARY if not any(_ext(n) == e for n in names)]
    if len(text) < MINIMUM_TEXT_FILES:
        failures.append(f"only {len(text)} text files (floor {MINIMUM_TEXT_FILES}): "
                        "the listing is broken, not the tree")
    return failures


def check(root: pathlib.Path) -> int:
    names = _ls_files(root)
    if names is None:
        return 1
    text = [n for n in names if _ext(n) not in BINARY]
    failures = _population_failures(names, text)
    nul, scan_failures = _scan(root, text)
    failures += scan_failures
    print(f"population: {len(names)} tracked, {len(text)} scanned as text, "
          f"{len(names) - len(text)} binary by extension ({', '.join(sorted(BINARY))}); "
          f"{nul} text file(s) with NUL")
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    return 1 if failures else 0


def self_test() -> int:
    """Positive controls: the gate must FAIL on each bad tree and PASS on the good one."""
    global MINIMUM_TEXT_FILES
    saved = MINIMUM_TEXT_FILES
    MINIMUM_TEXT_FILES = 2
    results: list[tuple[str, bool]] = []
    try:
        def tree(files: dict[str, bytes]) -> pathlib.Path:
            d = pathlib.Path(tempfile.mkdtemp(prefix="nul_gate_"))
            for name, content in files.items():
                (d / name).write_bytes(content)
            for args in (("init", "-q"), ("add", "-A")):
                proc = _git(d, *args)
                if proc is None or proc.returncode != 0:
                    raise RuntimeError(f"self-test setup failed: git {args[0]}")
            return d

        base = {"a.rs": b"fn main() {}\n", "b.md": b"hello\n", "k.spv": b"\x03\x02\x23\x07\x00\x00",
                "p.jpeg": b"\xff\xd8\x00"}
        cases = [
            ("clean tree passes (NULs in .spv are allowed)", base, 0),
            ("NUL in a text file fails", {**base, "c.rs": b"fn x() {}\x00\n"}, 1),
            ("UTF-16 text file fails", {**base, "d.md": "hi\n".encode("utf-16-le")}, 1),
            ("stale BINARY entry (no .jpeg tracked) fails",
             {"a.rs": b"x\n", "b.md": b"y\n", "k.spv": b"\x00"}, 1),
            ("too few text files fails", {"a.rs": b"x\n", "k.spv": b"\x00", "p.jpeg": b"\x00"}, 1),
        ]
        for label, files, want in cases:
            d = tree(files)
            try:
                got = check_quiet(d)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            results.append((label, got == want))
    finally:
        MINIMUM_TEXT_FILES = saved
    for label, ok in results:
        print(("ok   " if ok else "FAIL ") + label)
    return 0 if all(ok for _, ok in results) else 1


def check_quiet(root: pathlib.Path) -> int:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return check(root)


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    return check(pathlib.Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
