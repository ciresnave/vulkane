#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Does every version this repo cites FOR CONSUMPTION still resolve to what the
registry currently serves?

    a version cited for CONSUMPTION  ->  must accept the crate's max_stable
    a version cited to describe THIS TREE  ->  must be the tree's own

!! ONE STRING DOES BOTH JOBS ONLY WHILE THE TWO COINCIDE, and post-publish
bumping breaks that coincidence deliberately and permanently: a member's
declared version is normally NOT on the registry, so a doc that cites the
tree's version hands users something that does not exist.

!! THE FAILURE MODE IS SILENT, WHICH IS WHY THIS EXISTS. Measured on this repo
before it was fixed: seven sites said `vulkane = "0.4"` or `"0.10"` while
crates.io served 0.16.0. EVERY ONE OF THEM RESOLVED -- `cargo add` succeeded
and handed a stranger a years-old release with feature flags documented against
0.16. A requirement that fails to resolve at least tells the user something;
one that resolves to the wrong thing has no complainant.

So the test is NOT "does it resolve". It is "does it ACCEPT WHAT IS SERVED
NOW". `"0.4"` resolves perfectly and accepts nothing newer than 0.4.x.

!! CITATION vs HISTORY is decided STRUCTURALLY, not by a path list. A mention
inside a fenced `toml` block is an instruction to copy into a manifest. A
mention in prose -- `CHANGELOG.md` describing what a past release required --
is a record, and rewriting it would falsify the record rather than fix a
pointer. The fence is where the author already wrote down which one they meant.
An allowlist of "historical files" would name the CATEGORY; the fence names the
PROPERTY.

Exit codes: 0 clean, 1 with --gate if any citation is stale. Always 1 if the
scan found no citations at all -- a scan that matches nothing must FAIL, not
pass. `--self-test` checks this file's own guards offline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess  # nosec B404 - the member list and the file list both
                   # come from running a tool; there is no other source
import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "cited-versions-probe (+https://github.com/ciresnave/vulkane)"}
REGISTRY = "https://crates.io/api/v1/crates"

STALE = "STALE"
UNPARSED = "UNPARSED"
NEVER_PUBLISHED = "NEVER-PUBLISHED"


def tool_path(name: str) -> str:
    """The absolute path of the tool that will answer.

    !! A bare name is resolved by PATH at CALL TIME, so two runs that disagree
    cannot say whether they ran the same binary. This was applied to `cargo`
    and not to `git` in the first version of this file -- the same rule, in the
    same file, applied once.
    """
    import shutil

    found = shutil.which(name)
    if not found:
        raise SystemExit(
            "%s is not on PATH, so this scan cannot run. That is a missing "
            "tool, not a clean result." % name)
    return found


def workspace_crates(manifest_dir: str) -> set[str]:
    """Every workspace member's name. Citations of anything else are not ours."""
    out = subprocess.run(  # nosec B603 # nosemgrep
        [tool_path("cargo"), "metadata", "--no-deps", "--format-version", "1"],
        cwd=manifest_dir, capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit("cargo metadata failed in %s" % manifest_dir)
    return {p["name"] for p in json.loads(out.stdout)["packages"]}


def tracked_files(manifest_dir: str) -> list[str]:
    """Tracked `.md` and `.rs` files, from git rather than a directory walk.

    A walk would pick up `target/` and anything untracked; the question is what
    this REPOSITORY says, not what is lying around.
    """
    # nosemgrep - argv[0] is the absolute path resolved for the literal
    # "git"; every other element is a literal and there is no shell.
    out = subprocess.run(  # nosec B603 # nosemgrep
        [tool_path("git"), "ls-files", "*.md", "*.rs"],
        cwd=manifest_dir, capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit("git ls-files failed in %s" % manifest_dir)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def toml_fenced_lines(lines: list[str], is_rust: bool) -> set[int]:
    """1-based line numbers inside a ```toml (or bare ```) fence.

    !! OPEN/CLOSE state and LANGUAGE are two variables. Conflating them makes a
    ```rust block's CLOSING fence read as an OPENING one, flipping the state
    for the rest of the file -- which produces a plausible mixed answer rather
    than an obvious error. Measured while building this: three of four README
    citations were reported outside a fence they are plainly inside.
    """
    in_fence, lang, out = False, None, set()
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if is_rust:
            s = re.sub(r"^//[/!]\s?", "", s).strip()
        if s.startswith("```"):
            if in_fence:
                in_fence, lang = False, None
            else:
                in_fence, lang = True, s[3:].strip().lower()
            continue
        if in_fence and lang in ("toml", ""):
            out.add(i)
    return out


def citations(manifest_dir: str, crates: set[str]) -> list[tuple]:
    """(file, line, crate, requirement) for every consumption citation."""
    if not crates:
        raise SystemExit(
            "no workspace crates, so nothing could be cited -- refusing to "
            "report a clean scan over an empty subject")
    pattern = re.compile(
        r"(%s)\s*=\s*(?:\{[^}]*version\s*=\s*\"([^\"]+)\"|\"([^\"]+)\")"
        % "|".join(re.escape(c) for c in sorted(crates)))
    found = []
    for rel in tracked_files(manifest_dir):
        found.extend(citations_in_file(manifest_dir, rel, pattern))
    return found


def citations_in_file(manifest_dir: str, rel: str, pattern) -> list[tuple]:
    """Consumption citations in one file: those inside a fenced toml block."""
    path = os.path.join(manifest_dir, *rel.split("/"))
    try:
        raw = io.open(path, "rb").read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = raw.splitlines()
    fenced = toml_fenced_lines(lines, rel.endswith(".rs"))
    out = []
    for i, line in enumerate(lines, 1):
        if i not in fenced:
            continue
        m = pattern.search(line)
        if m:
            out.append((rel, i, m.group(1), m.group(2) or m.group(3)))
    return out


def accepts(req: str, version: str) -> bool | None:
    """Does a bare cargo requirement accept `version`? None if not understood.

    Bare means caret: `"0.16"` is `>=0.16.0, <0.17.0`; `"1.2"` is `>=1.2.0,
    <2.0.0`. Anything carrying an explicit operator, a wildcard or a comma is
    returned as None rather than guessed -- a matcher that silently mishandles
    a form it does not implement reports clean about a citation it never read.
    """
    parts = parse_version(req)
    v = parse_version(version)
    if parts is None or v is None:
        return None
    r = parts + [0] * (3 - len(parts))
    v = v + [0] * (3 - len(v))
    if v < r:
        return False
    return v < caret_upper(parts)


def parse_version(s: str) -> list[int] | None:
    """`"0.16"` -> [0, 16], or None if it is not a bare dotted number.

    None means NOT UNDERSTOOD and is propagated, never coerced -- a matcher
    that guesses at a form it does not implement reports clean about a
    citation it never read.
    """
    if not re.fullmatch(r"\d+(\.\d+){0,2}", s.strip()):
        return None
    return [int(x) for x in s.strip().split(".")]


def caret_upper(parts: list[int]) -> list[int]:
    """The exclusive upper bound of a bare cargo requirement.

    Set by the leftmost NON-ZERO component AS WRITTEN: `0.16` bounds at 0.17.0,
    `1.2` at 2.0.0, `0.0.3` at 0.0.4. Getting this wrong in the obvious
    direction -- bumping the major -- makes `0.4` accept `0.16.0`, which is
    precisely the defect this file exists to catch.
    """
    idx = next((i for i, x in enumerate(parts) if x != 0), len(parts) - 1)
    upper = parts + [0] * (3 - len(parts))
    upper[idx] += 1
    for j in range(idx + 1, 3):
        upper[j] = 0
    return upper


def max_stable(name: str) -> str | None:
    """The crate's current max_stable_version, or None if never published."""
    try:
        req = urllib.request.Request("%s/%s" % (REGISTRY, name), headers=UA)
        # nosemgrep - the prefix is the literal https REGISTRY constant, so the
        # value cannot select a scheme; only the crate name varies.
        with urllib.request.urlopen(req, timeout=60) as r:  # nosec B310 # nosemgrep
            return json.load(r)["crate"].get("max_stable_version")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def classify(req: str, served: str | None) -> tuple[str, str]:
    """(verdict, detail) for one citation."""
    if served is None:
        return NEVER_PUBLISHED, "the registry has never served this crate"
    ok = accepts(req, served)
    if ok is None:
        return UNPARSED, "requirement %r is not a bare caret form" % req
    if ok:
        return "ok", ""
    return STALE, "does not accept the served %s" % served


def report(rows: list[tuple], gate: bool) -> int:
    """Print the table and decide the exit code."""
    print("  %-34s %-18s %-9s %-9s %s"
          % ("site", "crate", "cited", "served", "verdict"))
    bad = []
    for site, crate, req, served, verdict, detail in rows:
        if verdict != "ok":
            bad.append((site, crate, req, detail))
        print("  %-34s %-18s %-9s %-9s %s"
              % (site, crate, req, served or "-", verdict))

    # A scan that matched nothing must FAIL, not pass. Zero citations and a
    # repo whose citations are all correct produce identical silence.
    if not rows:
        print("\n  !! no consumption citations found -- the scan did not run.")
        print("     A citation is a `<crate> = ...` line inside a fenced toml")
        print("     block. Finding none means the pattern, the fence tracker")
        print("     or the file list stopped working, not that the docs are")
        print("     clean.")
        return 1

    print()
    if bad:
        for site, crate, req, detail in bad:
            print("  %s cites %s = %r: %s" % (site, crate, req, detail))
        print()
        print("  A version cited for CONSUMPTION must accept what the registry")
        print("  serves NOW. `cargo add` following a stale one SUCCEEDS and")
        print("  hands the reader an old release -- it does not error, so")
        print("  nobody reports it.")
        if gate:
            return 1
    else:
        print("  %d citation(s), all accepting the currently served version."
              % len(rows))
    return 0


def self_test() -> int:
    """Offline arms. Every one was run against a broken version first."""
    failures = []

    def check(name, ok):
        print("  %-4s %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            failures.append(name)

    # -- the caret matcher, including the historical defect -----------------
    for req, ver, want, why in (
            ("0.4", "0.16.0", False, "THE DEFECT: 0.4 does not accept 0.16.0"),
            ("0.10", "0.16.0", False, "nor does 0.10"),
            ("0.16", "0.16.0", True, "0.16 accepts the served 0.16.0"),
            ("0.16", "0.16.5", True, "...and later patches"),
            ("0.16", "0.17.0", False, "but not the next minor"),
            ("1.2", "1.9.0", True, "1.x carets bound at the MAJOR"),
            ("1.2", "2.0.0", False, "...and stop there"),
            ("0.0.3", "0.0.4", False, "0.0.x bounds at the PATCH"),
            ("0.16", "0.15.0", False, "a served version BELOW the floor")):
        check("caret %-6s vs %-7s -> %-5s (%s)" % (req, ver, want, why),
              accepts(req, ver) is want)

    # ...and forms it must REFUSE TO GUESS rather than mishandle.
    for req in (">=0.16", "0.16.*", "^0.16", "0.16, <0.17", "", "latest"):
        check("requirement %-12r is UNPARSED, not guessed" % req,
              accepts(req, "0.16.0") is None)

    # -- the fence discriminator -------------------------------------------
    md = ["prose citing `vulkane = \"0.4\"` as history", "", "```rust",
          "let x = 1;", "```", "", "```toml", "vulkane = { version = \"0.16\" }",
          "```", "trailing prose"]
    fenced = toml_fenced_lines(md, False)
    check("prose outside any fence is not a citation", 1 not in fenced)
    # ...and the control: a ```rust block's CLOSE must not open a toml one.
    check("a rust block's closing fence does not open a toml one",
          4 not in fenced and 10 not in fenced)
    check("a line inside the toml fence IS a citation", 8 in fenced)

    rs = ["//! ```toml", "//! vulkane = { version = \"0.16\" }", "//! ```",
          "pub fn f() {}"]
    check("a toml fence inside rustdoc is found", 2 in toml_fenced_lines(rs, True))

    # -- exit codes ---------------------------------------------------------
    quiet = io.StringIO()

    def code(rows, gate):
        with contextlib.redirect_stdout(quiet):
            return report(rows, gate)

    clean = [("README.md:87", "vulkane", "0.16", "0.16.0", "ok", "")]
    stale = clean + [("README.md:9", "vulkane", "0.4", "0.16.0", STALE, "x")]
    check("an empty scan fails in report mode", code([], False) == 1)
    check("an empty scan fails in gate mode", code([], True) == 1)
    check("a clean non-empty scan passes when armed", code(clean, True) == 0)
    check("a stale citation reports without blocking when unarmed",
          code(stale, False) == 0)
    check("the same stale citation blocks when armed", code(stale, True) == 1)
    check("an UNPARSED requirement blocks too, rather than passing",
          code([("R:1", "vulkane", ">=1", "0.16.0", UNPARSED, "x")], True) == 1)

    if failures:
        print("\n  self-test FAILED: %s" % "; ".join(failures))
        return 1
    print("\n  self-test passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", default=".", help="workspace root")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 on a stale citation (default: report only)")
    ap.add_argument("--self-test", action="store_true",
                    help="check this file's own guards; needs no network")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    crates = workspace_crates(args.manifest_dir)
    served: dict[str, str | None] = {}
    rows = []
    for rel, line, crate, req in citations(args.manifest_dir, crates):
        if crate not in served:
            served[crate] = max_stable(crate)
        verdict, detail = classify(req, served[crate])
        rows.append(("%s:%d" % (rel, line), crate, req, served[crate],
                     verdict, detail))
    return report(rows, args.gate)


if __name__ == "__main__":
    sys.exit(main())
