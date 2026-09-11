#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every source file declares the workspace licence, and keeps declaring it.

    python3 .github/spdx_gate.py            # check
    python3 .github/spdx_gate.py --self-test

A SWEEP DOES NOT FIX A PROPERTY THAT REGROWS. `fuel` stamped 795 files on
2026-08-19 and had drifted to 823/833 by 2026-09-10 -- not because anything was
undone, but because ten files were added afterwards and nothing was watching.
The sweep is the one-off; this is the part that lasts.

WHAT THIS REFUSES TO DO, AND WHY EACH ONE IS DELIBERATE:

  * It never WRITES. A gate that repairs the thing it measures reports success
    forever and tells you nothing about whether anyone is maintaining it.

  * It never treats a DIFFERENT identifier as a failure to be corrected.
    `fuel-examples/src/bs1770.rs` is a verbatim Apache-2.0-ONLY third-party
    work, and fuel's blanket sweep stamped `MIT OR Apache-2.0` onto it --
    asserting a licence grant nobody made. Files whose licence is somebody
    else's decision belong in HOLDOUT below, by name, with a reason.

  * It refuses to pass on an EMPTY population. `0 of 0 files` is 100% compliant
    by arithmetic and means the glob matched nothing -- a renamed directory, a
    moved workspace, a typo in an extension. The comforting number and the
    broken query are the same number, and nobody audits good news.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell, no caller input
import sys

LICENCE = "MIT OR Apache-2.0"
# ⚠️ AN EXTENSION THIS TUPLE OMITS IS A POPULATION THE GATE NEVER COUNTED, and
# the omission shows up as a CLEANER number rather than a smaller one. The first
# version listed only .rs and .py and reported 120/120 at 100% while nine shader
# sources carried no header at all - not wrong, just never asked about.
#
# The shaders are safe to stamp, and that is measured rather than reasoned from
# the GLSL spec: all nine ALREADY opened with a `//` comment line before this
# gate existed, and they compile in the `shaderc` and naga jobs today.
EXTENSIONS = (".rs", ".py", ".wgsl", ".comp", ".vert", ".frag", ".glsl")

#: How far into a file the header may sit. Measured, not guessed: across 2,100
#: real .rs files in this portfolio (fuel, kiss-ref, lightbulb, synapse), 851
#: SPDX lines sit at line index 0, one at index 4, and NONE at index >= 10.
HEADER_WINDOW = 10

#: The gate's own floor. If the tree ever holds fewer source files than this,
#: the query is broken rather than the tree suddenly minimal. Set below the
#: real count on purpose: this catches a glob that matches nothing or almost
#: nothing, not ordinary deletion.
MINIMUM_FILES = 100

#: Paths this gate must NOT require a header on, each with the reason it is here.
#:
#: 🔴 THIS COMMENT USED TO SAY `git grep -l -i copyright` RETURNS ZERO HITS HERE.
#: IT RETURNS TWELVE. I wrote the unscoped query in the comment and ran a scoped
#: one -- the pre-flight's source-extension globs, which contain no `.xml`:
#:
#:     git grep -l -i copyright                     -> 12 files
#:       11 x LICENSE-APACHE / LICENSE-MIT             ours, expected
#:        1 x vulkane/vk.xml   Copyright 2015-2026 The Khronos Group Inc.
#:     git grep -l -i copyright -- '*.rs' '*.py'    ->  1 file
#:     control: the same grep for SPDX-License-Identifier -> 133 files
#:
#: ⚠️ THE EMPTY HOLDOUT IS STILL CORRECT, AND THAT IS THE DANGEROUS PART.
#: `.xml` is not in EXTENSIONS, so `vk.xml` is never scanned and never stamped -
#: its `SPDX-License-Identifier: Apache-2.0 OR MIT` on line 6 is KHRONOS'S OWN
#: upstream line, present since the file was bundled and untouched by any sweep.
#: The conclusion is right; the stated reason was false.
#:
#: ⚠️ A RIGHT ANSWER RESTING ON A FALSE REASON IS WORSE THAN A WRONG ONE,
#: because nothing will ever fail in a way that reveals it. The moment `.xml`
#: enters EXTENSIONS - plausible for a repo that bundles a registry XML - this
#: gate demands a header on `vk.xml`, and the old comment told whoever added it
#: that there was nothing here to protect.
#:
#: SO: IF `.xml` IS EVER ADDED TO EXTENSIONS, `vulkane/vk.xml` NEEDS AN ENTRY
#: HERE FIRST. It is a third-party registry file carrying Khronos's copyright,
#: and it is not ours to relicense.
#:
#: An entry that matches no file is an ERROR below -- a holdout that protects
#: nothing reads exactly like one with nothing to protect, right up until the
#: file it named is renamed and then stamped.
HOLDOUT: dict[str, str] = {}

MARKER = "SPDX-License-Identifier:"


def normalise(identifier: str) -> str:
    """Canonical form for COMPARISON only. `Apache-2.0 OR MIT` and
    `MIT OR Apache-2.0` are the same grant, and one repo in this portfolio
    spells it each way. A false conflict standing next to a true one trains
    the reader to dismiss both."""
    text = identifier.strip()
    for joiner in (" OR ", " or "):
        if joiner in text:
            return " OR ".join(sorted(p.strip() for p in text.split(joiner)))
    return text


def declared(text: str) -> str | None:
    """The identifier a file declares, or None.

    Split on the marker, never a regex: an earlier `[\\w.-]+(?: OR [\\w.-]+)?`
    truncated `MIT OR Apache-2.0` to `MIT OR Apache`, which made every
    correctly-stamped file look wrong and would have had a second pass append a
    duplicate header to all of them.
    """
    for line in text.splitlines()[:HEADER_WINDOW]:
        if MARKER in line:
            value = line.split(MARKER, 1)[1].strip()
            for terminator in ("*/", "-->", "*)"):
                if value.endswith(terminator):
                    value = value[: -len(terminator)].strip()
            return value or None
    return None


def tracked_sources(root: pathlib.Path) -> list[str]:
    # ⚠️ RESOLVED ABSOLUTE, NOT "git". A bare name is looked up through PATH at
    # call time, so what runs depends on the environment rather than on this
    # file. The argv is fixed, there is no shell, and nothing here comes from a
    # caller - `root` is this script's own parent directory.
    git = shutil.which("git")
    if git is None:
        print("FAIL: no `git` on PATH. This gate reads the tracked file list,",
              file=sys.stderr)
        print("      and cannot distinguish 'no files' from 'no git'.", file=sys.stderr)
        return []
    # ⚠️ `-z`, AND THAT IS NOT A STYLE CHOICE. Without it `git ls-files` QUOTES
    # any path containing a non-ASCII or special character, emitting
    # `"src/naÃ¯ve.rs"` - octal escapes, wrapped in literal quotes, and
    # THAT STRING IS NOT A PATH THAT EXISTS. The gate would report a perfectly
    # good file as UNREADABLE, or - before unreadable files were collected
    # rather than fatal - fail the whole run on one accented filename.
    #
    # Caught by a reviewer on kiss-ref#43. `tools/spdx.py` already used `-z`;
    # this did not. ⚠️ THE SWEEPER AND ITS GATE DISAGREED ABOUT HOW TO READ THE
    # SAME LIST, which is the divergence this project keeps finding in itself.
    proc = subprocess.run(  # noqa: S603 - fixed argv, shell=False
        [git, "-C", str(root), "ls-files", "-z", "--",
         *(f"*{e}" for e in EXTENSIONS)],
        capture_output=True, encoding=None, shell=False, check=False)
    if proc.returncode != 0:
        # ⚠️ stderr is reported, not discarded. `git ls-files` failing and
        # `git ls-files` finding nothing both yield an empty list.
        print(f"git ls-files failed: {(proc.stderr or '').strip()}", file=sys.stderr)
        return []
    text = proc.stdout.decode("utf-8", "replace")
    return [n for n in text.split(chr(0)) if n]


#: Files allowed to contain the word "copyright". ⚠️ A PATTERN, NOT A COUNT.
#: Licence texts contain it by definition; a changelog records licence changes;
#: this script discusses copyright in its own comments and so matches itself.
#: ⚠️ MATCHED ON THE BASENAME, NOT THE PATH. A substring test over the whole
#: path exempts anything beneath a directory whose name contains one of these -
#: `vendor/LICENSE-deps/foo.rs` would pass unexamined. Contrived here, where
#: nothing is vendored; not contrived in the three repos this gate also runs in,
#: two of which vendor third-party source.
COPYRIGHT_EXPECTED = ("LICENSE", "LICENCE", "COPYING", "CHANGELOG",
                      "spdx_gate.py", "NOTICE")


#: 🔴 THIRD-PARTY NOTICES THAT HAVE BEEN LOOKED AT AND RULED ON. Distinct from
#: `COPYRIGHT_EXPECTED`, which says "this file contains the word for a boring
#: reason". This says "this file carries SOMEBODY ELSE'S COPYRIGHT, we know, and
#: here is the decision" - a record, not an exemption from scanning.
#:
#: ⚠️ AND THE FIRST ENTRY IS THE ONE MY PROSE SAID DID NOT EXIST. The holdout
#: comment in #94 claimed `git grep -l -i copyright` returned zero hits here; it
#: returns twelve, one of them `vk.xml`. That comment was corrected, and then
#: this check found the file in its first run - which is the whole argument for
#: asserting a property over writing a sentence.
COPYRIGHT_ACKNOWLEDGED = {
    "vulkane/vk.xml":
        "the Vulkan registry, Copyright 2015-2026 The Khronos Group Inc., "
        "bundled verbatim so docs.rs can build. It carries Khronos's OWN "
        "`SPDX-License-Identifier: Apache-2.0 OR MIT` on line 6. `.xml` is not "
        "in EXTENSIONS so it is never stamped - and if `.xml` is ever added, "
        "this file needs a HOLDOUT entry FIRST. It is not ours to relicense.",
}


def _expected(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return any(tag in base for tag in COPYRIGHT_EXPECTED)


def survey_copyright(root: pathlib.Path) -> list[str]:
    """Tracked files carrying a copyright notice that are NOT expected to, or
    None if the survey COULD NOT RUN.

    ⚠️ `None` AND `[]` ARE DIFFERENT ANSWERS AND THE DIFFERENCE IS THE POINT.
    An empty list is the PASS condition here, so a survey that failed to run
    returning `[]` would be indistinguishable from a clean tree - and this
    gate's entire justification is that the holdout is CHECKED rather than
    asserted. A CHECK THAT SILENTLY NO-OPS ON FAILURE IS AN ASSERTION WITH A
    FUNCTION WRAPPED AROUND IT.

    ⚠️ THIS IS THE HOLDOUT'S JUSTIFICATION, MOVED OUT OF A COMMENT AND INTO CI.
    The comment used to state a COUNT: `-> 0` with a control of `-> 8`. That
    count drifted TWICE IN FOUR DAYS from two unrelated PRs, with neither author
    doing anything wrong - a changelog gained the word, then this script did.

    ⚠️ A COUNT CANNOT SURVIVE TREE GROWTH; A PROPERTY CAN. The thing the count
    was standing in for is "every -i copyright hit is a licence file or this
    script", and that is assertable. A COMMENT CANNOT GUARD - demonstrated twice
    inside this one file, first by the stale figure and then by the correction
    that vouched for it.

    ⚠️ AND IT FIRES EXACTLY WHERE THE HOLDOUT WOULD BE NEEDED. A new file
    carrying somebody else's copyright notice is precisely the case where a
    blanket sweep asserts a licence grant nobody made - `bs1770.rs` in `fuel`,
    Khronos's `vk.xml` in `vulkane`, Apple's kernels in `fuel`'s .metal files.
    """
    git = shutil.which("git")
    if git is None:
        print("FAIL: no `git` on PATH; the copyright survey could not run.",
              file=sys.stderr)
        return None
    proc = subprocess.run(  # noqa: S603 - fixed argv, shell=False
        [git, "-C", str(root), "grep", "-l", "-i", "-z", "copyright"],
        capture_output=True, encoding=None, shell=False, check=False)
    if proc.returncode not in (0, 1):
        # ⚠️ REPORTED AND REFUSED, NOT SWALLOWED. `git grep` exits 1 for "no
        # matches" and 128 for "not a repository", and an empty list cannot tell
        # them apart - see `tracked_sources` fifteen lines above, whose comment
        # says exactly this about `ls-files`. I wrote that comment and then
        # shipped this defect beneath it.
        print("FAIL: the copyright survey could not run: "
              + proc.stderr.decode("utf-8", "replace").strip()[:200],
              file=sys.stderr)
        return None
    names = [n for n in proc.stdout.decode("utf-8", "replace").split(chr(0)) if n]
    return sorted(n for n in names
                  if not _expected(n) and n not in COPYRIGHT_ACKNOWLEDGED)


def audit(root: pathlib.Path, files: list[str]):
    """(missing, wrong, unreadable) over `files`, skipping HOLDOUT entries."""
    expected = normalise(LICENCE)
    missing, wrong, unreadable = [], [], []
    for rel in files:
        if rel in HOLDOUT:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append((rel, str(exc)))
            continue
        found = declared(text)
        if found is None:
            missing.append(rel)
        elif normalise(found) != expected:
            wrong.append((rel, found))
    return missing, wrong, unreadable


def report(files: list, missing: list, wrong: list, unreadable: list,
           stale: list) -> None:
    """Print the findings. Separated from deciding them so that changing how
    this reads cannot change what it concluded."""
    clean = len(files) - len(missing) - len(wrong) - len(unreadable) - len(HOLDOUT)
    print(f"{clean}/{len(files)} tracked source files declare {LICENCE!r}"
          + (f"  ({len(HOLDOUT)} held out)" if HOLDOUT else ""))
    rows = ([("MISSING", rel, "") for rel in missing]
            + [("DIFFERENT", rel, f" declares {found!r}") for rel, found in wrong]
            + [("UNREADABLE", rel, f": {why}") for rel, why in unreadable]
            + [("STALE HOLDOUT", rel,
                " matches no tracked file - it protects NOTHING") for rel in stale])
    for label, rel, suffix in rows:
        print(f"  {label}  {rel}{suffix}")


def explain(missing: list, wrong: list) -> None:
    """What to do about each kind of finding. Separate from the finding itself,
    because the remedies differ in KIND: one is mechanical, one is a decision."""
    if missing:
        print()
        print("Add the header as the FIRST line, above any `//!` inner docs:")
        print(f"    // {MARKER} {LICENCE}")
        print("A shebang stays on line 1 and the header goes below it.")
    if wrong:
        print()
        print("A file declaring a DIFFERENT licence is NOT a formatting error.")
        print("Changing it asserts a grant its author may not have made. Either")
        print("the declaration is right and the file belongs in HOLDOUT with a")
        print("reason, or it is wrong and that is a decision for the owner.")


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    root = pathlib.Path(__file__).resolve().parent.parent
    files = tracked_sources(root)

    if len(files) < MINIMUM_FILES:
        print(f"FAIL: found {len(files)} source files, expected at least "
              f"{MINIMUM_FILES}.")
        print("      This gate reports 100% compliance on an empty set, so it")
        print("      fails here instead. The GLOB is broken, not the tree.")
        return 1

    missing, wrong, unreadable = audit(root, files)
    stale = sorted(set(HOLDOUT) - set(files))
    # ⚠️ THE HOLDOUT'S JUSTIFICATION, CHECKED RATHER THAN ASSERTED IN PROSE.
    surveyed = survey_copyright(root)
    if surveyed is None:
        # ⚠️ The survey could not run. Refusing is the only honest outcome: a
        # clean report here would be a claim nobody measured.
        return 1
    unexpected = [n for n in surveyed if n not in HOLDOUT]
    # ⚠️ AN ACKNOWLEDGEMENT THAT NAMES NO FILE IS A RECORD OF A DECISION ABOUT
    # SOMETHING THAT NO LONGER EXISTS - the same defect as a stale holdout, and
    # it fails the same way: silently, while looking like diligence.
    tracked_all = set()
    proc_git = shutil.which("git")
    if proc_git:
        out = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            [proc_git, "-C", str(root), "ls-files", "-z"],
            capture_output=True, encoding=None, shell=False, check=False)
        tracked_all = {n for n in out.stdout.decode("utf-8", "replace").split(chr(0)) if n}
    stale_ack = sorted(set(COPYRIGHT_ACKNOWLEDGED) - tracked_all) if tracked_all else []
    for rel in stale_ack:
        print(f"  STALE ACKNOWLEDGEMENT  {rel} matches no tracked file")
    if stale_ack:
        unexpected = unexpected + stale_ack
    report(files, missing, wrong, unreadable, stale)
    explain(missing, wrong)
    for rel in unexpected:
        print(f"  COPYRIGHT NOTICE  {rel} is not a licence file and is not "
              f"in HOLDOUT")
    if unexpected:
        print()
        print("A file carrying somebody else's copyright notice may not be")
        print("ours to license. Decide, then add it to HOLDOUT with the")
        print("reason, or to COPYRIGHT_EXPECTED if the match is incidental.")
    return 1 if (missing or wrong or stale or unreadable or unexpected) else 0


def self_test() -> int:
    """⚠️ The gate's own positive controls. A checker nobody has watched FAIL
    is a checker nobody has evidence works."""
    cases = [
        ("bare header", "// SPDX-License-Identifier: MIT OR Apache-2.0\n", "MIT OR Apache-2.0"),
        ("above inner docs", "// SPDX-License-Identifier: MIT OR Apache-2.0\n//! docs\n", "MIT OR Apache-2.0"),
        ("below a shebang", "#!/usr/bin/env python3\n# SPDX-License-Identifier: MIT OR Apache-2.0\n", "MIT OR Apache-2.0"),
        ("block comment", "/* SPDX-License-Identifier: MIT OR Apache-2.0 */\n", "MIT OR Apache-2.0"),
        ("nothing at all", "fn main() {}\n", None),
        # ⚠️ The truncation control. A regex-based reader returned `MIT OR
        # Apache` here and every correct file in the portfolio looked wrong.
        ("full dual identifier", "// SPDX-License-Identifier: MIT OR Apache-2.0\n", "MIT OR Apache-2.0"),
        # ⚠️ The self-counting control. This gate declares its own licence in
        # its own header AND names it in a string constant; a reader that
        # scanned the whole file would find the constant too.
        ("beyond the window", "\n" * 12 + "// SPDX-License-Identifier: MIT\n", None),
    ]
    failures = 0
    for name, text, expected in cases:
        got = declared(text)
        ok = got == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got!r}"
              + ("" if ok else f"  expected {expected!r}"))

    # ⚠️ CONTROLS FOR THE COPYRIGHT SURVEY'S CLASSIFIER. Needs no git: the part
    # that can silently rot is the PREDICATE, and `COPYRIGHT_EXPECTED` is a list
    # somebody will extend. A manual both-arms run proves the checker worked
    # that afternoon; nothing re-runs it when the list gains an entry.
    classifications = [
        ("LICENSE-MIT", True),
        ("crates/kiss-ref-core/LICENSE-APACHE", True),
        ("CHANGELOG.md", True),
        (".github/spdx_gate.py", True),
        ("src/lib.rs", False),
        ("crates/kiss-ops-vocab/src/lib.rs", False),
        # ⚠️ THE ONE THAT MATTERS, and it failed before basename anchoring: a
        # substring test over the whole PATH exempts everything beneath a
        # directory whose name contains a tag.
        ("vendor/LICENSE-deps/foo.rs", False),
        ("third_party/NOTICE-files/kernel.cu", False),
    ]
    for path, expected in classifications:
        got = _expected(path)
        ok = got == expected
        failures += not ok
        verb = "exempt" if expected else "examined"
        print(f"  {'ok  ' if ok else 'FAIL'}  {path} is {verb}")

    equivalences = [("MIT OR Apache-2.0", "Apache-2.0 OR MIT", True),
                    ("Apache-2.0", "MIT OR Apache-2.0", False),
                    ("MIT", "MIT OR Apache-2.0", False)]
    for a, b, same in equivalences:
        ok = (normalise(a) == normalise(b)) == same
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {a!r} {'==' if same else '!='} {b!r}")

    # ⚠️ SUMMED, NOT WRITTEN DOWN. This line said "10 controls" while 18 ran,
    # for one commit - a stale count inside the run whose entire purpose is to
    # kill stale counts. A COUNT CANNOT SURVIVE ITS OWN LIST GROWING.
    total = len(cases) + len(equivalences) + len(classifications)
    print(f"{chr(10)}{'PASS' if not failures else 'FAIL'}: {total} controls, "
          f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
