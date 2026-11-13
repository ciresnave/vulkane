#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Stamp a version onto a change (portable kit, PM 2026-11-12; derived from Unpopped's tools/stamp-version.py).

    python tools/stamp-version.py <version> [--root DIR] [--dry-run]

A PR branch adds only its own files plus a CHANGELOG fragment (`changelog.d/<topic>.md`, format in `changelog.d/README.md`)
and never edits `CHANGELOG.md`, a `Cargo.toml` version or `Cargo.lock`. When the PM allocates the number the gate runs this
script, which

 1. finds every `Cargo.toml` tracked by git whose `[package]` version is the project's current version (one version per
    project) and sets the new one, including `path`+`version` dependency lines and the matching `Cargo.lock` entries;
 2. sets `[workspace.package] version` when the root uses it;
 3. turns the fragments into one entry above the newest CHANGELOG entry (heading style copied from the file);
 4. deletes the fragments it used.

An optional `tools/stamp-current.txt` (one line: a Cargo.toml path) names the manifest whose version is "current" when the
repo has crates on independent versions; only crates at that version are stamped.

Extra files that pin the version as `=X.Y.Z` (README install lines) are listed one path per line in
`tools/stamp-extra.txt` (optional). Standard library only. Line endings of each file are kept. Files are enumerated with
`git ls-files`, never a disk walk.
"""
import argparse
import os
import re
import subprocess
import sys

EM = "—"


def read(path):
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="surrogateescape")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def write(path, text, crlf):
    with open(path, "wb") as f:
        f.write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def die(msg):
    print(f"stamp-version: {msg}", file=sys.stderr)
    sys.exit(2)


def tracked(root, pattern):
    out = subprocess.run(["git", "-C", root, "ls-files", pattern], capture_output=True, text=True)
    if out.returncode != 0:
        die("git ls-files failed: " + out.stderr.strip())
    return [l for l in out.stdout.splitlines() if l]


def package_version(text):
    """Version in the [package] table, or None; 'workspace' when `version.workspace = true`."""
    m = re.search(r"(?ms)^\[package\]\s*$(.*?)(?=^\[|\Z)", text)
    if not m:
        return None
    body = m.group(1)
    v = re.search(r'(?m)^\s*version\s*=\s*"(\d+\.\d+\.\d+[^"]*)"', body)
    if v:
        return v.group(1)
    if re.search(r"(?m)^\s*version\.workspace\s*=\s*true", body):
        return "workspace"
    return None


def workspace_version(text):
    m = re.search(r"(?ms)^\[workspace\.package\]\s*$(.*?)(?=^\[|\Z)", text)
    if not m:
        return None
    v = re.search(r'(?m)^\s*version\s*=\s*"(\d+\.\d+\.\d+[^"]*)"', m.group(1))
    return v.group(1) if v else None


def current_version(root):
    pin = os.path.join(root, "tools", "stamp-current.txt")
    if os.path.exists(pin):
        rel = read(pin)[0].strip().splitlines()[0].strip()
        text, _ = read(os.path.join(root, rel))
        v = package_version(text) or workspace_version(text)
        if not v or v == "workspace":
            die(f"tools/stamp-current.txt names {rel}, which has no literal version")
        return v
    top, _ = read(os.path.join(root, "Cargo.toml"))
    wv = workspace_version(top)
    if wv:
        return wv
    pv = package_version(top)
    if pv and pv != "workspace":
        return pv
    for p in tracked(root, "*Cargo.toml"):
        v = package_version(read(os.path.join(root, p))[0])
        if v and v != "workspace":
            return v
    die("no crate version found")


def set_table_version(text, table, old, new):
    """Replace `version = "old"` inside one [table]; returns (text, changed)."""
    pat = re.compile(r"(?ms)^(\[" + re.escape(table) + r"\]\s*$)(.*?)(?=^\[|\Z)")
    m = pat.search(text)
    if not m:
        return text, False
    body, n = re.subn(r'(?m)^(\s*version\s*=\s*")' + re.escape(old) + '"', lambda x: x.group(1) + new + '"', m.group(2))
    return (text[: m.start(2)] + body + text[m.end(2):], n > 0)


def version_changes(root, old, new):
    q = re.escape(old)
    out, names = [], []
    top_text, _ = read(os.path.join(root, "Cargo.toml"))
    ws_old = workspace_version(top_text) == old
    for rel in tracked(root, "*Cargo.toml"):
        p = os.path.join(root, rel)
        t, crlf = read(p)
        orig = t
        pv = package_version(t)
        if pv == "workspace" and ws_old:
            nm = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', t)
            if nm:
                names.append(nm.group(1))
        if pv == old:
            t, _ = set_table_version(t, "package", old, new)
            nm = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', t)
            if nm:
                names.append(nm.group(1))
        if workspace_version(t) == old:
            t, _ = set_table_version(t, "workspace.package", old, new)
        # path + version dependency lines (inline tables), exact or =pinned
        t = re.sub(
            r'(?m)^(\s*[\w.-]+\s*=\s*\{[^}\n]*path\s*=\s*"[^"]+"[^}\n]*version\s*=\s*"=?)' + q + '"',
            lambda m: m.group(1) + new + '"',
            t,
        )
        # version before path: `x = { version = "old", path = ".." }`
        t = re.sub(
            r'(?m)^(\s*[\w.-]+\s*=\s*\{[^}\n]*version\s*=\s*"=?)' + q + r'("[^}\n]*path\s*=)',
            lambda m: m.group(1) + new + m.group(2),
            t,
        )
        if t != orig:
            out.append((p, t, crlf))
    lock = os.path.join(root, "Cargo.lock")
    if os.path.exists(lock) and not names:
        die("Cargo.lock exists but no crate was identified as carrying the version")
    if os.path.exists(lock) and names:
        t, crlf = read(lock)
        orig = t
        for nm in names:
            t = re.sub(r'(name = "' + re.escape(nm) + r'"\nversion = ")' + q + '"', lambda m: m.group(1) + new + '"', t)
        if t == orig:
            die("Cargo.lock: no workspace crate entry carries the old version")
        out.append((lock, t, crlf))
    extra = os.path.join(root, "tools", "stamp-extra.txt")
    if os.path.exists(extra):
        for line in read(extra)[0].splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = os.path.join(root, line)
            t, crlf = read(p)
            t2, n = re.subn(r"=" + q + r"\b", "=" + new, t)
            if n == 0:
                die(f"{line}: no =old pin")
            out.append((p, t2, crlf))
    if not out:
        die("nothing carries the old version")
    return out


def fragments(directory):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, n) for n in sorted(os.listdir(directory)) if n.endswith(".md") and n.lower() != "readme.md"]


def parse_fragment(path):
    text, _ = read(path)
    head, sep, body = text.partition("\n---\n")
    if not sep or not head.startswith("kind:"):
        die(f"{path}: expected a first line `kind: ...`, a line `---`, then the entry")
    return head[len("kind:"):].strip(), body.strip("\n")


def changelog_change(root, version, frags):
    if not frags:
        die("no changelog.d/*.md fragment: every pushed change needs a CHANGELOG entry")
    kinds, bodies = [], []
    for f in frags:
        k, b = parse_fragment(f)
        kinds.append(k)
        bodies.append(b)
    kind = next((k for k in kinds if "breaking" in k.lower()), kinds[0])
    p = os.path.join(root, "CHANGELOG.md")
    t, crlf = read(p)
    m = re.search(r"(?m)^## (\[?)`?\d+\.\d+\.\d+", t)
    if not m:
        die("CHANGELOG.md has no version heading to put the entry above")
    if m.group(1) == "[":
        heading = f"## [{version}] {EM} {kind}"
    elif t[m.start():].startswith("## `"):
        heading = f"## `{version}` {EM} {kind}"
    else:
        heading = f"## {version} {EM} {kind}"
    entry = heading + "\n\n" + "\n\n".join(bodies) + "\n\n"
    if re.search(r"(?m)^---\s*$", t[m.start(): m.start() + 4000]):
        entry += "---\n\n"
    return (p, t[: m.start()] + entry + t[m.start():], crlf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", a.version):
        die(f"`{a.version}` is not MAJOR.MINOR.PATCH")
    root = os.path.abspath(a.root)
    old = current_version(root)
    if old == a.version:
        die(f"the crates already say {old}")
    frags = fragments(os.path.join(root, "changelog.d"))
    changes = version_changes(root, old, a.version)
    changes.append(changelog_change(root, a.version, frags))
    for path, _, _ in changes:
        print(("would change " if a.dry_run else "changed ") + os.path.relpath(path, root).replace("\\", "/"))
    if a.dry_run:
        print(f"{old} -> {a.version}; {len(frags)} fragment(s)")
        return
    for path, text, crlf in changes:
        write(path, text, crlf)
    for f in frags:
        os.remove(f)
    print(f"stamped {old} -> {a.version}; removed {len(frags)} fragment(s)")


if __name__ == "__main__":
    main()
