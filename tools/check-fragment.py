#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check that a pull request carries exactly one CHANGELOG fragment, or bumps the version itself.

    python tools/check-fragment.py <base-ref>        e.g. origin/main

Compares HEAD with the merge base of <base-ref>. A PR must do exactly one of:

  * add one `changelog.d/<topic>.md` (not README.md) and leave every version alone; or
  * change a crate version itself (a release-line backport, or a PR opened before the repo moved to fragments) and add
    no fragment.

Doing both, adding two fragments, or doing neither fails, with the reason. Standard library only.
"""
import re
import subprocess
import sys


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def version_lines(diff):
    return [
        l for l in diff.splitlines()
        if re.match(r'^[+-]\s*version\s*=\s*"\d+\.\d+\.\d+[^"]*"', l)
        and not re.search(r"path\s*=|git\s*=|registry\s*=", l)
    ]


def judge(added_fragments, version_changed):
    if added_fragments and version_changed:
        return "this PR both bumps a version and adds a fragment: do one (fragments, once the PM stamps the version)"
    if version_changed or len(added_fragments) == 1:
        return None
    if not added_fragments:
        return "this PR has no CHANGELOG fragment: add changelog.d/<topic>.md (see changelog.d/README.md)"
    return f"this PR adds {len(added_fragments)} fragments ({', '.join(added_fragments)}): exactly one"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    base = git("merge-base", sys.argv[1], "HEAD").strip()
    names = git("diff", "--name-status", "--diff-filter=A", base, "HEAD").splitlines()
    added = [n.split("\t", 1)[1] for n in names if re.match(r"^A\tchangelog\.d/(?!README\.md$)[^/]+\.md$", n)]
    manifests = [p for p in git("diff", "--name-only", base, "HEAD").splitlines() if p.endswith("Cargo.toml")]
    changed = any(version_lines(git("diff", "-U0", base, "HEAD", "--", p)) for p in manifests)
    why = judge(added, changed)
    if why:
        print("check-fragment: " + why)
        return 1
    print("check-fragment: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
