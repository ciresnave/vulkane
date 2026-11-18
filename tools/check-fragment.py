#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check that a pull request carries exactly one CHANGELOG fragment, or bumps the version itself.

    python tools/check-fragment.py <base-ref>        e.g. origin/main

Compares HEAD with the merge base of <base-ref>. A PR must do exactly one of:

  * add one `changelog.d/<topic>.md` (not README.md) and leave every version alone; or
  * change a crate or package.json version itself (a release-line backport, or a PR opened before the repo moved to fragments) and add
    no fragment.

Doing both, adding two fragments, or doing neither fails, with the reason. A repo whose stamp covers only some crates lists the
manifests the rest bump themselves in `changelog.d/self-versioned.txt` (one path per line); those are not counted as "bumps a
version". Standard library only.
"""
import re
import shutil
import subprocess  # nosec B404 - runs `git` with fixed arguments in a developer or CI checkout
import sys


def git(*args):
    exe = shutil.which("git")
    if not exe:
        raise SystemExit("check-fragment: git is not on PATH")
    return subprocess.run([exe, *args], check=True, capture_output=True, text=True).stdout  # nosec B603 - fixed arguments


def version_lines(diff):
    toml = [
        l for l in diff.splitlines()
        if re.match(r'^[+-]\s*version\s*=\s*"\d+\.\d+\.\d+[^"]*"', l)
        and not re.search(r"path\s*=|git\s*=|registry\s*=", l)
    ]
    # a package.json's own `"version": "X.Y.Z"` (its dependencies are `"name": "^1.2.3"`)
    npm = [l for l in diff.splitlines() if re.match(r'^[+-]\s*"version"\s*:\s*"\d+\.\d+\.\d+[^"]*"', l)]
    return toml + npm


def is_manifest(path):
    return path.endswith(("Cargo.toml", "package.json")) and not path.endswith("-package.json")


def self_versioned():
    """Manifests a repo lists in `changelog.d/self-versioned.txt` (one path per line, `#` comments): crates that bump their own
    version by rule (the stamp covers only the project-line crates), so a PR that bumps one of them AND adds a fragment is fine.
    Read from HEAD, so the list is reviewed like any other change; a missing file means none."""
    try:
        text = git("show", "HEAD:changelog.d/self-versioned.txt")
    except Exception:  # no such file: no exemption
        return set()
    lines = (l.split("#", 1)[0].strip() for l in text.replace("\r\n", "\n").splitlines())
    return {l for l in lines if l}


def judge(added_fragments, version_changed):
    if added_fragments and version_changed:
        return "this PR both bumps a version and adds a fragment: do one (fragments, once the PM stamps the version)"
    if version_changed or len(added_fragments) == 1:
        return None
    if not added_fragments:
        return "this PR has no CHANGELOG fragment: add changelog.d/<topic>.md (see changelog.d/README.md)"
    return f"this PR adds {len(added_fragments)} fragments ({', '.join(added_fragments)}): exactly one"


def fragment_problem(path):
    """Why a fragment cannot be stamped (empty, no `kind:` line, no `---`, empty body), else None."""
    try:
        text = git("show", "HEAD:" + path).replace("\r\n", "\n")
    except Exception:  # a failed read must fail the check, not pass an empty file
        return f"{path} could not be read"
    head, sep, body = text.partition("\n---\n")
    if not text.strip():
        return f"{path} is empty"
    if not sep or not head.startswith("kind:") or not head[len("kind:"):].strip():
        return f"{path} must start `kind: <text>`, then a line `---`, then the entry"
    if not body.strip():
        return f"{path} has no entry text after the `---` line"
    return None


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    base = git("merge-base", sys.argv[1], "HEAD").strip()
    names = git("diff", "--name-status", "--diff-filter=A", base, "HEAD").splitlines()
    added = [n.split("\t", 1)[1] for n in names if re.match(r"^A\tchangelog\.d/(?!README\.md$)[^/]+\.md$", n)]
    # Only manifests the PR MODIFIES can bump a version; a crate the PR adds carries its first `version =` line (the stamp sets it later).
    manifests = [p for p in git("diff", "--name-only", "--diff-filter=M", base, "HEAD").splitlines() if is_manifest(p)]
    own = self_versioned()
    manifests = [p for p in manifests if p not in own]
    changed = any(version_lines(git("diff", "-U0", base, "HEAD", "--", p)) for p in manifests)
    why = judge(added, changed)
    if why:
        print("check-fragment: " + why)
        return 1
    for frag in added:
        bad = fragment_problem(frag)
        if bad:
            print("check-fragment: " + bad)
            return 1
    print("check-fragment: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
