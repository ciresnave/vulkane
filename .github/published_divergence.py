#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Does a crate this workspace PUBLISHES sit on a version string the registry
already serves, while its content differs from what was served?

    version NOT on the registry  ->  no collision is possible          ->  OK
    version IS on the registry   ->  content MUST match the tarball    ->  DIVERGED

!! THE DEFECT IS A VERSION-STRING COLLISION, NOT DIVERGENCE.
A workspace member differing from the last published release is NORMAL -- it is
what development looks like. Reddening on that would be a detector firing on
healthy input, and a detector that fires on healthy input is worse than none.
The defect is sitting on a string the registry SERVES while differing from what
was served under it: a consumer resolving that string gets one artifact and this
workspace's own CI tests another, and both are green about different objects.

!! THE REMEDY IS A WORKFLOW, NOT A PUBLISH: bump immediately AFTER publishing,
not before. A member that moves to an unpublished version the moment it is
published can never collide, and the interval between publishes stops being a
period during which the tree makes a false claim.

!! THIS IS THE PRODUCER ARM AND IT IS THE ONLY ONE THAT FINDS THE DEFECT.
A lockfile tells you whether YOUR dependencies are pinned by checksum -- whether
you could be HURT by this class. It says nothing about whether YOU are one of
the two artifacts for somebody else. The candidate set here is every crate this
repo publishes, read from the workspace manifests; the lockfile is never opened.
A repo whose lockfile is entirely checksummed can still be a perpetrator.

Exit codes: 0 in report mode (the default) whatever it finds; with --gate, 1 if
anything DIVERGED. Always 1 if the scan itself found nothing to check -- a scan
that matches nothing must FAIL, not pass.

`--self-test` exercises the guards in this file against fabricated rows and
temporary directories; it needs no network and exits 1 if any arm fails.

!! Keep this docstring ASCII. argparse prints it for --help and a Windows
console is cp1252, so a warning glyph here makes --help crash with a
UnicodeEncodeError -- measured, on the first command anyone runs. The function
docstrings below are never printed and keep the usual markers.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import shutil
import subprocess  # nosec B404 - enumerating members means running cargo
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

UA = {"User-Agent": "published-divergence-probe (+https://github.com/ciresnave/vulkane)"}
REGISTRY = "https://crates.io/api/v1/crates"


def cargo_path() -> str:
    """The ABSOLUTE path of the cargo that will answer.

    !! Not merely a lint fix. A bare `cargo` is resolved by PATH at call time,
    so two runs that disagree cannot say whether they ran the same binary --
    and this file already insists elsewhere that a toolchain identify itself
    before its answer means anything. It was not applying that to the call it
    depends on.
    """
    found = shutil.which("cargo")
    if not found:
        raise SystemExit(
            "cargo is not on PATH, so the member list cannot be built. That is "
            "a missing toolchain, not an empty workspace -- do not read it as "
            "nothing to check.")
    return found


def publishes_to_crates_io(publish) -> bool:
    """Does this member make a claim on a CRATES.IO version string?

    `cargo metadata` spells the field three ways and only one of them is the
    absence of a restriction:

        None                  publish anywhere -- the default          YES
        []                    `publish = false`                        no
        ["crates-io"]         explicitly allowed here                  YES
        ["some-private-reg"]  allowed SOMEWHERE ELSE                   no

    !! The last row is why `publish != []` is not the test. A member restricted
    to a private registry publishes nothing to crates.io, so comparing it
    against a crates.io artifact compares it against a STRANGER'S crate that
    merely shares its name -- a confident DIVERGED about two unrelated things.
    """
    return publish is None or "crates-io" in publish


def members(manifest_dir: str) -> list[tuple[str, str, str]]:
    """Every member that claims a crates.io version string: (name, version, dir).

    From `cargo metadata`, never from Cargo.lock.
    """
    # nosemgrep - argv[0] is NOT a static string ON PURPOSE: it is the
    # absolute path `shutil.which` resolved for the literal name "cargo",
    # which is strictly more determinate than the bare name this rule would
    # accept. The tail is literal and there is no shell.
    out = subprocess.run(  # nosec B603 # nosemgrep
        [cargo_path(), "metadata", "--no-deps", "--format-version", "1"],
        cwd=manifest_dir, capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        # stderr is kept deliberately: a swallowed failure here yields an empty
        # list, which reads as "nothing to check" rather than "cargo failed".
        sys.stderr.write(out.stderr)
        raise SystemExit("cargo metadata failed in %s" % manifest_dir)
    meta = json.loads(out.stdout)
    return [
        (p["name"], p["version"], os.path.dirname(p["manifest_path"]))
        for p in sorted(meta["packages"], key=lambda p: p["name"])
        if publishes_to_crates_io(p.get("publish"))
    ]


def registry_get(url: str, timeout: int):
    """Open a URL that must be on the registry, asserting the scheme.

    !! `urllib` honours `file://` and any custom scheme its openers know, so
    "it is an https constant plus a crate name" is a fact about crate-name
    syntax rather than a property of this call. The prefix is checked instead,
    which makes the reachable surface a statement in the code.
    """
    if not url.startswith(REGISTRY + "/"):
        raise SystemExit("refusing a URL outside the registry: %r" % url)
    req = urllib.request.Request(url, headers=UA)
    # nosec B310 - the registry prefix is asserted immediately above, so the
    # scheme cannot be file:// or any other opener the value might name.
    # nosemgrep - the dynamic part cannot select a scheme: the assertion
    # above rejects anything not prefixed with the literal https REGISTRY,
    # so file:// and every other opener are unreachable. Verified both
    # ways: file:///etc/passwd and https://evil.example refused, the real
    # registry URL allowed.
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310 # nosemgrep


def served(name: str) -> set[str] | None:
    """Versions the registry serves. None means the crate was never published."""
    try:
        with registry_get("%s/%s" % (REGISTRY, name), 60) as r:
            return {v["num"] for v in json.load(r)["versions"]}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def reject_unsafe_members(t: tarfile.TarFile) -> None:
    """Refuse any member that could write outside the extraction directory.

    !! Runs on EVERY Python version, not only where `filter=` is missing. A
    guard placed only in the fallback branch is present on the machine you
    tested it on and absent on the older runner that takes the other branch --
    which is the one configuration nobody looks at.
    """
    for m in t.getmembers():
        name = m.name.replace(chr(92), "/")
        if name.startswith("/") or os.path.isabs(name) or ".." in name.split("/"):
            raise SystemExit(
                "refusing a tarball member whose path escapes the extraction "
                "directory: %r" % m.name)
        if m.issym() or m.islnk():
            raise SystemExit(
                "refusing a link member in a registry tarball: %r" % m.name)


def fetch(name: str, version: str, into: str) -> str:
    url = "%s/%s/%s/download" % (REGISTRY, name, version)
    with registry_get(url, 180) as r:
        blob = r.read()
    path = os.path.join(into, "%s-%s.crate" % (name, version))
    io.open(path, "wb").write(blob)
    with tarfile.open(path, "r:gz") as t:
        reject_unsafe_members(t)
        # `filter="data"` is the 3.14 default and a hard error to omit there;
        # setting it explicitly keeps one behaviour across versions.
        # nosec B202 - `reject_unsafe_members` above has already refused any
        # escaping path or link member, on every Python version; `filter="data"`
        # is a second layer where the interpreter provides it.
        try:
            t.extractall(into, filter="data")  # nosec B202
        except TypeError:  # Python < 3.12 has no `filter=`
            t.extractall(into)  # nosec B202
    return os.path.join(into, "%s-%s" % (name, version))


def lines(path: str) -> list[bytes]:
    """CRLF-normalized, and BYTES all the way down.

    Without the normalization every file differs on a Windows checkout, and a
    comparison that always fires is a comparison nobody reads.

    !! Without the bytes it under-reports instead. Decoding with
    `errors="replace"` maps every invalid sequence onto U+FFFD, so two files
    that differ in their actual bytes compare EQUAL -- a divergence probe
    silently blind to exactly the corruption it should be loudest about. There
    is no reason to decode: nothing here reads the text, only compares it.
    """
    return io.open(path, "rb").read().replace(b"\r\n", b"\n").split(b"\n")


def edit_size(a: list[bytes], b: list[bytes]) -> tuple[int, int, int]:
    """(hunks, removed, added) from a real diff algorithm.

    ⚠️ NEVER a line-by-line inequality count. A positional walk marks every line
    after an insertion as differing: one file scored 689 that way and 15 by this
    one. First-divergence under-reports to a single token; positional comparison
    over-reports by everything downstream. Both are wrong about the change.
    """
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    return (len(ops),
            sum(o[2] - o[1] for o in ops),
            sum(o[4] - o[3] for o in ops))


ABSENT_FROM_TREE = -1
ABSENT_FROM_ARTIFACT = -2


# Regenerated or rewritten by cargo during packaging, so they differ on every
# crate ever published and would make the comparison fire on everything.
GENERATED = frozenset({
    "Cargo.toml",            # cargo-normalized from Cargo.toml.orig
    "Cargo.toml.orig",       # a copy, by construction
    "Cargo.lock",            # resolved at package time
    ".cargo_vcs_info.json",  # stamped with the commit
})


def shipped_files(root: str) -> set[str]:
    """Every AUTHORED file in an unpacked crate, forward-slashed and relative.

    !! Was `src/` only, which on kiss-vulkan-vocab compared 1 of 14 packaged
    files -- skipping `manifest/vulkan-vocabulary.json`, the normative artifact
    that crate exists to publish, plus the emitter and every test.
    """
    out = set()
    for base, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(base, f), root).replace(os.sep, "/")
            if rel not in GENERATED:
                out.add(rel)
    return out


def packaged_files(crate_dir: str) -> set[str]:
    """What `cargo package` WOULD ship from the tree right now.

    !! Not a directory walk. The published side lists what shipped, so the tree
    side has to answer the same question or the two are not comparable -- a raw
    walk flags every file cargo deliberately excludes, and `target/` alone would
    bury the finding.
    """
    # !! `--allow-dirty` because the SUBJECT IS THE WORKING TREE. Without it
    # cargo refuses outright when anything is uncommitted -- and the gate would
    # then fail for every local user with work in progress, which is exactly
    # who needs it before a publish. CI never sees this: it checks out clean,
    # so the check's environment differs from the one that matters.
    #
    # !! No `-p <name>`: running in the crate's own directory selects it, so
    # NOTHING VARIABLE REACHES argv. An earlier version validated the name
    # against cargo's identifier rules instead, which is strictly weaker -- a
    # checked argument is still an argument. Verified the two forms agree
    # before switching (identical listings for kiss-vulkan-vocab and
    # vulkane_derive).
    # nosemgrep - same disposition as the other two call sites: argv[0] is the
    # absolute path resolved for the literal "cargo", which is why it is not a
    # static string, and every other element IS one.
    out = subprocess.run(  # nosec B603 # nosemgrep
        [cargo_path(), "package", "--quiet", "--list", "--allow-dirty"],
        cwd=crate_dir, capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit("cargo package --list failed in %s" % crate_dir)
    return {ln.strip().replace(os.sep, "/") for ln in out.stdout.splitlines()
            if ln.strip() and ln.strip() not in GENERATED}


def find_in_tree(tree_dir: str, rel: str, fallback_root: str | None) -> str | None:
    """Where a packaged file actually lives, or None if nowhere readable.

    !! A path in `cargo package --list` is not necessarily a path under the
    crate directory. `vulkane` declares `readme = "README.md"` and has none of
    its own -- cargo packages the WORKSPACE readme under that name. Assuming
    otherwise made this crash on any SERVED member with an inherited file,
    which is the one condition the gate exists to detect.
    """
    direct = os.path.join(tree_dir, *rel.split("/"))
    if os.path.exists(direct):
        return direct
    if fallback_root:
        inherited = os.path.join(fallback_root, *rel.split("/"))
        if os.path.exists(inherited):
            return inherited
    return None


def compare(published_dir: str, tree_dir: str,
            tree_listing: set[str] | None = None,
            fallback_root: str | None = None) -> tuple[list[tuple], int]:
    """Authored files only, walked SYMMETRICALLY.

    !! The union of both sides, not the tarball's side. Walking only what was
    published means a file ADDED to the workspace is never examined and the
    crate reports `ok` while differing -- the probe would be blind to the
    commonest way a tree moves ahead of a release.

    The tarball's Cargo.toml is cargo-normalized and its .cargo_vcs_info.json is
    generated, so both differ on every crate ever published. Including them would
    flag everything — the same always-fires failure as ignoring CRLF.
    """
    findings = []
    published = shipped_files(published_dir)
    tree = tree_listing if tree_listing is not None else shipped_files(tree_dir)
    for rel in sorted(published | tree):
        p = os.path.join(published_dir, *rel.split("/"))
        t = find_in_tree(tree_dir, rel, fallback_root)
        if rel not in tree or t is None:
            findings.append((rel, ABSENT_FROM_TREE, 0, 0))
        elif rel not in published:
            findings.append((rel, ABSENT_FROM_ARTIFACT, 0, 0))
        else:
            a, b = lines(p), lines(t)
            if a != b:
                findings.append((rel,) + edit_size(a, b))
    return findings, len(published | tree)


def print_rows(rows: list[tuple]) -> tuple[list[str], list[str]]:
    """Print the table. Returns (diverged, unscanned) by crate name."""
    print("  %-22s %-10s %-16s %6s  %s"
          % ("crate", "version", "string", "files", "verdict"))
    diverged, unscanned = [], []
    for name, version, state, checked, findings in rows:
        # !! A crate on a SERVED string compared over ZERO files is not clean,
        # it is UNSCANNED. This is the same rule as the empty member list in
        # `report` below, one level down -- and I had guarded the empty MEMBER
        # LIST while leaving the empty FILE SET reading as agreement. Found by
        # review on #85, which is the second time this file's own anti-vacuous
        # rule was applied at one level and not the other.
        blind = state == "SERVED" and checked == 0
        verdict = "UNSCANNED" if blind else ("DIVERGED" if findings else "ok")
        if blind:
            unscanned.append(name)
        elif findings:
            diverged.append(name)
        print("  %-22s %-10s %-16s %6d  %s"
              % (name, version, state, checked, verdict))
        print_findings(findings)
    return diverged, unscanned


def print_findings(findings: list[tuple]) -> None:
    for rel, hunks, rem, add in findings:
        if hunks == ABSENT_FROM_TREE:
            print("  %52s %s  (absent from the tree)" % ("", rel))
        elif hunks == ABSENT_FROM_ARTIFACT:
            print("  %52s %s  (absent from the published artifact)" % ("", rel))
        else:
            print("  %52s %s  %d hunks, -%d/+%d" % ("", rel, hunks, rem, add))


def print_control(rows: list[tuple]) -> None:
    """Say whether an all-ok result could be distinguished from a broken one."""
    controls = [r for r in rows if r[2] in ("UNPUBLISHED", "NEVER-PUBLISHED")]
    print()
    if controls:
        print("  control: %s on an unpublished version -> ok, so a DIVERGED row is a"
              % controls[0][0])
        print("           finding rather than a comparator artifact.")
    else:
        print("  !! NO in-tree control: every member sits on a served version, so an")
        print("     all-ok result cannot be distinguished from a broken comparator.")
        print("     Bump one member post-publish to get a known-green row.")


def report(rows: list[tuple], gate: bool) -> int:
    """Print the table and decide the exit code.

    Split out of `main` so the exit-code rules can be exercised with fabricated
    rows -- no network, no cargo, no registry. The anti-vacuous guard below is
    the one line that decides whether this file is a gate or a decoration, and
    a guard whose only proof lived in a throwaway directory is a guard nobody
    can re-check later.
    """
    diverged, unscanned = print_rows(rows)

    # A scan that matched nothing must FAIL, not pass. An empty member list and
    # a clean workspace produce identical silence otherwise, and the empty one
    # is the dangerous reading: it says "checked, all fine" about zero crates.
    if not rows:
        print("\n  !! no publishable members found -- the scan did not run.")
        return 1

    print_control(rows)

    if unscanned:
        print()
        print("  %d crate(s) sit on a SERVED version and were compared over ZERO"
              % len(unscanned))
        print("  files: %s" % ", ".join(unscanned))
        print("  Nothing was examined, so this is not a clean result. Either the")
        print("  published tarball carries no `src/`, or the member keeps its")
        print("  sources somewhere this probe does not look.")

    if diverged:
        print()
        print("  %d crate(s) sit on a SERVED version string with different content: %s"
              % (len(diverged), ", ".join(diverged)))
        print("  Remedy: bump to an unpublished version. Publishing is only needed if")
        print("  the difference is one consumers should receive -- measure that, do not")
        print("  assume it: identical sources are not required for identical behaviour,")
        print("  and differing sources do not imply differing output.")

    if (diverged or unscanned) and gate:
        return 1
    return 0


# --------------------------------------------------------------------------
# self-test
#
# Every arm below was run against a deliberately broken version of the code it
# checks, and failed there, before being kept. The drill is re-run from scratch
# after any refactor: arms that were red against the old code prove nothing
# about the new one, and this file has already had one gate stop being able to
# fail while looking unchanged.
#
# Grouped into functions because they check unrelated things, and because a
# single 100-line self-test hides which group a failure came from.
# --------------------------------------------------------------------------

def exit_code_arms(check) -> None:
    """The exit-code rules, over fabricated rows. No network, no cargo.

    Every arm that expects a ZERO is downstream of the vacuous guard, so
    breaking that one guard reddens several arms at once; that is correct and
    not a redundancy to trim.
    """
    quiet = io.StringIO()

    def code(rows, gate):
        with contextlib.redirect_stdout(quiet):
            return report(rows, gate)

    clean = [("a", "1.0.0", "SERVED", 12, []),
             ("b", "2.0.0", "UNPUBLISHED", 0, [])]
    dirty = clean + [("c", "3.0.0", "SERVED", 4, [("src/lib.rs", 2, 5, 5)])]
    blind = [("a", "1.0.0", "SERVED", 0, [])]

    # Report mode is where this lands first, so a vacuous scan has to fail
    # there too; otherwise the unarmed period is one in which the check cannot
    # report its own absence.
    check("an empty scan fails in report mode", code([], False) == 1)
    check("an empty scan fails in gate mode", code([], True) == 1)
    # ...and its control: without this the guard could be an unconditional
    # `return 1` and both arms above would still read as passes.
    check("a clean non-empty scan passes when armed", code(clean, True) == 0)

    check("a divergence reports without blocking when unarmed",
          code(dirty, False) == 0)
    check("the same divergence blocks when armed", code(dirty, True) == 1)

    check("a SERVED crate compared over zero files does not read clean",
          code(blind, True) == 1)
    check("...and it reports without blocking when unarmed",
          code(blind, False) == 0)
    # An UNPUBLISHED crate compares zero files quite legitimately -- it has no
    # artifact to compare against -- so a guard firing on ANY zero would red
    # every correctly-bumped member, which is the remedy this gate encourages.
    check("an UNPUBLISHED crate at zero files is still clean",
          code([("b", "2.0.0", "UNPUBLISHED", 0, [])], True) == 0)


def publish_filter_arms(check) -> None:
    """Which `publish` spellings claim a crates.io string."""
    for publish, want, why in ((None, True, "the default: publish anywhere"),
                               ([], False, "`publish = false`"),
                               (["crates-io"], True, "explicitly allowed here"),
                               (["a-private-reg"], False, "allowed ELSEWHERE")):
        check("publish=%-17r -> %-5s (%s)" % (publish, want, why),
              publishes_to_crates_io(publish) == want)


def comparison_arms(check, tmp: str) -> None:
    """Line endings, byte fidelity, and the diff instrument."""
    crlf = os.path.join(tmp, "crlf.rs")
    lf = os.path.join(tmp, "lf.rs")
    other = os.path.join(tmp, "other.rs")
    io.open(crlf, "wb").write(b"fn a() {}\r\nfn b() {}\r\n")
    io.open(lf, "wb").write(b"fn a() {}\nfn b() {}\n")
    io.open(other, "wb").write(b"fn a() {}\nfn c() {}\n")

    # !! The fixture must be verified before the arm that uses it means
    # anything. A CRLF control whose "CRLF" file carries no CR passes for the
    # wrong reason -- the normalizer is never exercised and the arm reports
    # success. Read as bytes: this box has at least one CR detector that
    # answers identically on pure-LF and pure-CRLF input, and WHICH detectors
    # are blind turns out to differ between sessions on one machine.
    raw_crlf = io.open(crlf, "rb").read()
    raw_lf = io.open(lf, "rb").read()
    check("the CRLF fixture actually carries CR and the LF one does not",
          raw_crlf.count(b"\r\n") == 2 and raw_lf.count(b"\r") == 0)

    # A comparison that fires on every file is one nobody reads, and on a
    # Windows checkout line endings alone produce exactly that.
    check("line endings alone are not a difference", lines(crlf) == lines(lf))
    # ...and its control, or the normalizer could be returning a constant.
    check("a real difference survives normalization", lines(lf) != lines(other))

    # `edit_size` must not be a positional walk. Every line after the insertion
    # shifts, so a zip-and-count says 3; the answer is 1 hunk of 1 added line.
    # Both numbers are computed here, so the docstring's claim about the two
    # instruments is checked rather than asserted.
    a = ["one", "two", "three", "four"]
    b = ["one", "INSERTED", "two", "three", "four"]
    hunks, rem, add = edit_size(a, b)
    positional = sum(1 for x, y in zip(a, b) if x != y)
    check("an insertion is one hunk, not everything downstream",
          (hunks, rem, add) == (1, 0, 1) and positional == 3)

    # Two DIFFERENT invalid UTF-8 sequences decode to the same U+FFFD, so a
    # decoding comparator calls these files equal.
    bad_a, bad_b = os.path.join(tmp, "ba.rs"), os.path.join(tmp, "bb.rs")
    io.open(bad_a, "wb").write(b"x = \xff\n")
    io.open(bad_b, "wb").write(b"x = \xfe\n")
    check("distinct invalid UTF-8 bytes are not collapsed together",
          lines(bad_a) != lines(bad_b))


def walk_arms(check, tmp: str) -> None:
    """The file walk, in both directions."""
    pub, tree = os.path.join(tmp, "pub"), os.path.join(tmp, "tree")
    os.makedirs(os.path.join(pub, "src"))
    os.makedirs(os.path.join(tree, "src"))
    io.open(os.path.join(pub, "src", "lib.rs"), "wb").write(b"same\n")
    io.open(os.path.join(tree, "src", "lib.rs"), "wb").write(b"same\n")
    found, checked = compare(pub, tree)
    check("identical sources yield no findings, over a nonzero file count",
          found == [] and checked == 1)

    io.open(os.path.join(pub, "src", "gone.rs"), "wb").write(b"x\n")
    found, checked = compare(pub, tree)
    check("a file the tarball has and the tree lacks is flagged",
          len(found) == 1 and found[0][1] == ABSENT_FROM_TREE and checked == 2)

    # ...and the other direction, which the tarball-only walk could not see: a
    # file ADDED to the workspace is the commonest way a tree moves ahead of
    # its release, and it was reported as `ok`.
    io.open(os.path.join(tree, "src", "added.rs"), "wb").write(b"y\n")
    found, checked = compare(pub, tree)
    kinds = sorted(f[1] for f in found)
    check("a file the tree has and the tarball lacks is flagged",
          kinds == [ABSENT_FROM_ARTIFACT, ABSENT_FROM_TREE] and checked == 3)

    # -- A PACKAGED PATH NEED NOT LIVE UNDER THE CRATE DIRECTORY ------------
    # `vulkane` declares `readme = "README.md"` and has none of its own, so
    # cargo packages the WORKSPACE readme under that name. Reading it from the
    # crate directory raised FileNotFoundError -- and only for a SERVED member,
    # since an UNPUBLISHED one is never compared. The gate crashed on the one
    # condition it exists to detect.
    ws = os.path.join(tmp, "ws")
    os.makedirs(ws)
    io.open(os.path.join(pub, "README.md"), "wb").write(b"shared\n")
    io.open(os.path.join(ws, "README.md"), "wb").write(b"shared\n")
    found, checked = compare(pub, tree, {"src/lib.rs", "README.md"},
                             fallback_root=ws)
    # Assert about README specifically rather than a total: these fixtures
    # accumulate files across arms, so an exact count would encode the order
    # the arms happen to run in rather than the property under test.
    check("a packaged file inherited from the workspace root is compared",
          not any(f[0] == "README.md" for f in found) and checked >= 2)

    # ...and when it is nowhere, it is REPORTED rather than raising.
    io.open(os.path.join(pub, "LICENSE-MIT"), "wb").write(b"x\n")
    found, checked = compare(pub, tree,
                             {"src/lib.rs", "README.md", "LICENSE-MIT"},
                             fallback_root=ws)
    check("a packaged file readable nowhere is flagged, not raised",
          any(f[0] == "LICENSE-MIT" and f[1] == ABSENT_FROM_TREE for f in found))
    os.remove(os.path.join(pub, "LICENSE-MIT"))
    os.remove(os.path.join(pub, "README.md"))

    # -- THE DEFECT THIS WALK WIDTH EXISTS FOR ------------------------------
    # A `src/`-only walk compared 1 of 14 packaged files on kiss-vulkan-vocab,
    # skipping manifest/vulkan-vocabulary.json -- the normative artifact that
    # crate exists to publish. A divergence there reported `ok`, and the
    # `files` column read `1` on every run with no denominator beside it.
    os.makedirs(os.path.join(pub, "manifest"))
    os.makedirs(os.path.join(tree, "manifest"))
    io.open(os.path.join(pub, "manifest", "v.json"), "wb").write(b'{"a":1}\n')
    io.open(os.path.join(tree, "manifest", "v.json"), "wb").write(b'{"a":2}\n')
    found, checked = compare(pub, tree)
    check("a difference OUTSIDE src/ is compared at all",
          any(f[0] == "manifest/v.json" and f[1] > 0 for f in found))

    # ...and the exclusion, or widening the walk would fire on every crate ever
    # published: cargo rewrites these during packaging, so they differ always.
    io.open(os.path.join(pub, "Cargo.toml"), "wb").write(b'[package]\n')
    io.open(os.path.join(tree, "Cargo.toml"), "wb").write(b'[package]\nDIFFERENT\n')
    io.open(os.path.join(pub, ".cargo_vcs_info.json"), "wb").write(b'{"sha1":"a"}\n')
    io.open(os.path.join(tree, ".cargo_vcs_info.json"), "wb").write(b'{"sha1":"b"}\n')
    found2, checked2 = compare(pub, tree)
    check("cargo-rewritten files are never compared",
          sorted(f[0] for f in found2) == sorted(f[0] for f in found)
          and checked2 == checked)


def tarball_arms(check, tmp: str) -> None:
    """The extraction guard, which runs on every Python version."""
    payload = os.path.join(tmp, "payload")
    io.open(payload, "wb").write(b"x")

    def build(path, arcname):
        with tarfile.open(path, "w:gz") as w:
            info = w.gettarinfo(payload, arcname=arcname)
            with io.open(payload, "rb") as fh:
                w.addfile(info, fh)
        return path

    def refused(archive):
        try:
            with tarfile.open(archive, "r:gz") as r:
                reject_unsafe_members(r)
            return False
        except SystemExit:
            return True

    evil = build(os.path.join(tmp, "evil.tar.gz"), "../escaped.txt")
    safe = build(os.path.join(tmp, "safe.tar.gz"), "crate-1.0.0/src/lib.rs")
    check("a tarball member escaping the directory is refused", refused(evil))
    # ...and the control, or the guard could be refusing everything.
    check("an ordinary tarball member is not refused", not refused(safe))


def cargo_arm(check, tmp: str) -> None:
    """The one arm that needs a toolchain.

    `cargo metadata --no-deps` does not resolve dependencies, so this stays
    offline. It answers first: a result from a cargo that never identified
    itself says nothing about which cargo produced it.
    """
    # A missing cargo must say so in its own words. Left bare it raises
    # FileNotFoundError and the arm reds with a traceback, which reads as a
    # defect in this file rather than as a runner without a toolchain -- and
    # this job installs none, relying on the image providing one.
    found = shutil.which("cargo")
    if not found:
        answer, rc = "NOTHING (cargo is not on PATH)", 1
    else:
        # nosemgrep - same as above: `found` is the absolute path resolved
        # for the literal name "cargo", and the arm PRINTS it so a reader
        # can see which binary answered.
        ver = subprocess.run(  # nosec B603 # nosemgrep
            [found, "--version"], capture_output=True, text=True,
            encoding="utf-8")
        answer, rc = (ver.stdout or ver.stderr).strip(), ver.returncode
    # Print WHICH cargo, not only what it said: two runs that do not name the
    # binary they used cannot be compared.
    print("  --   cargo answers: %s" % (answer or "NOTHING"))
    print("  --   from: %s" % (found or "<not found>"))
    if rc != 0:
        print("       ^ this arm needs a toolchain on the runner; add one to")
        print("         the job rather than reading the failure as a code defect.")

    ws = os.path.join(tmp, "ws")
    os.makedirs(os.path.join(ws, "src"))
    io.open(os.path.join(ws, "Cargo.toml"), "w", encoding="utf-8").write(
        '[package]\nname = "unpublishable"\nversion = "0.1.0"\n'
        'edition = "2021"\npublish = false\n\n[workspace]\n')
    io.open(os.path.join(ws, "src", "main.rs"), "w",
            encoding="utf-8").write("fn main() {}\n")
    check("a `publish = false` member is not a candidate",
          rc == 0 and members(ws) == [])


def self_test() -> int:
    """Run every arm. Offline except the last, which says so."""
    failures = []

    def check(name, ok):
        print("  %-4s %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            failures.append(name)

    exit_code_arms(check)
    publish_filter_arms(check)
    with tempfile.TemporaryDirectory() as tmp:
        comparison_arms(check, tmp)
        walk_arms(check, tmp)
        tarball_arms(check, tmp)
        cargo_arm(check, tmp)

    if failures:
        print("\n  self-test FAILED: %s" % "; ".join(failures))
        return 1
    print("\n  self-test passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", default=".", help="workspace root")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 on divergence (default: report only)")
    ap.add_argument("--self-test", action="store_true",
                    help="check this file's own guards; needs no network")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, version, tree_dir in members(args.manifest_dir):
            pub = served(name)
            if pub is None:
                rows.append((name, version, "NEVER-PUBLISHED", 0, []))
            elif version not in pub:
                rows.append((name, version, "UNPUBLISHED", 0, []))
            else:
                d = fetch(name, version, tmp)
                findings, checked = compare(
                    d, tree_dir, packaged_files(tree_dir),
                    fallback_root=args.manifest_dir)
                rows.append((name, version, "SERVED", checked, findings))

    return report(rows, args.gate)


if __name__ == "__main__":
    sys.exit(main())
