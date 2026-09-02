#!/usr/bin/env python3
"""Emit the MSRV job's matrix by reading each crate's declared floor.

The declared `rust-version` is a promise to consumers, and this script exists so
that the promise and the thing that checks it cannot be different numbers. A
workflow with `1.88` written into it is a SECOND COPY of the promise: bump a
manifest and the gate keeps testing the old floor, reporting green for a version
nobody compiles any more. So the floor is read from `cargo metadata` at run time
and never restated here.

Crates may legitimately declare DIFFERENT floors -- a leaf crate with no
Vulkan loader in its graph can support an older compiler than the crate that
pulls one in -- so this emits one matrix leg per crate rather than one number
for the workspace.

Prints the matrix as JSON on stdout; everything else goes to stderr.
"""

import json
import subprocess
import sys

meta = json.loads(
    subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
)

# `publish` is None when the crate is publishable and a (possibly empty) list
# when it is restricted. Only a crate that can reach a consumer is making an
# MSRV promise worth exercising.
publishable = [p for p in meta["packages"] if p.get("publish") is None]
declared = [p for p in publishable if p.get("rust_version")]
silent = [p["name"] for p in publishable if not p.get("rust_version")]

for name in sorted(silent):
    print(
        f"note: {name} is published but declares no rust-version "
        f"(no promise made, so there is nothing to exercise)",
        file=sys.stderr,
    )

# A matrix of zero legs is green in exactly the way a passing one is. If the
# filter above ever stops matching -- a renamed metadata field, a workspace that
# stops publishing, a `cargo metadata` schema change -- this job must go red
# rather than quietly certify nothing. An unexercised floor is the defect this
# job was added to find, and a job that silently tests no crates has become an
# instance of it.
if not declared:
    print(
        "error: no publishable crate declares a rust-version; refusing to "
        "report success on an empty matrix",
        file=sys.stderr,
    )
    sys.exit(1)

# --- minimality ------------------------------------------------------------
#
# The matrix above proves each floor is SUFFICIENT: the crate builds at it.
# Nothing proved it NECESSARY. A floor set too high is a false promise in the
# other direction -- it turns away consumers who could have compiled -- and it
# fails silently, because everything is green and the people excluded never
# appear.
#
# The obvious check is unsound, twice, and both were measured before this was
# written:
#
#   cargo +1.87 build -p vulkan_gen
#     -> error: rustc 1.87.0 is not supported by the following package
#
# That is cargo enforcing the DECLARED rust-version. The check would "prove" the
# floor necessary by observing that cargo obeys the floor: the number causes the
# failure that justifies it. A check whose subject enforces its own precondition
# cannot return the interesting answer -- it must pass, on every crate, whether
# or not the floor is real.
#
#   cargo +1.84 build -p kiss-vulkan-vocab
#     -> error: failed to load manifest for workspace member ...
#
# `edition = "2024"` requires 1.85, so any cargo below the edition floor dies
# reading the workspace and never reaches the code. That is a failure at the
# wrong layer, reported as a finding about the right one.
#
# So: `--ignore-rust-version` to get past the first, and emit a leg only where
# `floor - 1` is still at or above the edition floor to avoid the second. Where
# it is not, the floor IS the edition floor and is minimal by construction --
# there is no lower version to fail at, which is an answer rather than a gap.

# Editions and the Rust version that stabilised each. This is a restated
# external fact, which this file otherwise avoids -- but it is Rust's own
# history and cannot change retroactively, unlike anything read from this
# workspace. cargo itself refuses to load a manifest below these.
EDITION_FLOOR = {"2015": (1, 0), "2018": (1, 31), "2021": (1, 56), "2024": (1, 85)}


def _parse(v):
    """`"1.88"` or `"1.88.0"` -> (1, 88). Returns None if it is not that shape."""
    parts = v.split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts[:2]):
        return None
    return (int(parts[0]), int(parts[1]))


def minimality_verdict(built_below, below, dep_floor):
    """What a minimality leg's build result MEANS. One implementation, testable.

    `built_below` is whether the crate compiled one minor under its floor.

      "necessary-code"  it did not compile: the code requires the floor
      "necessary-dep"   it compiled, but a dependency declares above `below`,
                        so cargo refuses a consumer there anyway
      "too-high"        it compiled and nothing else holds the floor up

    Lifted out of the job's shell because "necessary-dep" is only reachable when
    a build SUCCEEDS below the floor, which no crate here currently does -- a
    branch a green run never enters, so a green run cannot be its evidence. The
    same reason `toolchain_verdict` exists.
    """
    if not built_below:
        return "necessary-code"
    b, d = _parse(below), _parse(dep_floor or "")
    if d and b and d > b:
        return "necessary-dep"
    return "too-high"


def _reachable_non_dev(root, resolve):
    """Package ids reachable from `root` without crossing a dev-only edge.

    Split out of `dependency_floors` because "what does this crate pull in" and
    "which of those declares the highest floor" are two questions, and because a
    graph walk can be tested against a hand-built graph while the whole function
    needs a real `cargo metadata`. The dev exclusion in particular had no test
    until it was reachable on its own.
    """
    seen, stack = set(), [root]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for d in resolve[cur]["deps"]:
            kinds = {k.get("kind") for k in d.get("dep_kinds", [])}
            # `kind` is None for a normal dependency, so an edge is dev-ONLY
            # when every kind on it is "dev". An edge that is both dev and
            # normal still reaches a consumer and must be followed.
            if kinds and kinds <= {"dev"}:
                continue
            stack.append(d["pkg"])
    return seen


def _highest_declared(ids, skip, packages, workspace):
    """The highest floor declared among `ids`, and which package declares it.

    Attribution is deterministic. Several packages can share the highest floor
    and dict order would let the named one change between runs -- a message
    varying without the fact varying. External packages win ties because
    "libloading 0.9 declares 1.88" is actionable, while naming a workspace
    sibling that declares 1.88 for its own reasons just moves the question.
    """
    candidates = []
    for i in ids:
        if i == skip:
            continue
        v = _parse(packages[i].get("rust_version") or "")
        if v:
            candidates.append((v, i in workspace,
                               packages[i]["name"], packages[i]["version"]))
    if not candidates:
        return None, None
    v, _ws, name, version = sorted(
        candidates, key=lambda c: (c[0], not c[1], c[2]), reverse=True
    )[0]
    return v, name + " " + version


def dependency_floors():
    """Highest floor DECLARED by each workspace crate's non-dev dependencies.

    Deliberately a declaration and not a measurement, and that is the right
    instrument here rather than the cheap one. For a crate's OWN floor,
    "declared" and "required" are different questions -- which is why the
    minimality job builds instead of reading a number. For a DEPENDENCY they are
    the same question: cargo refuses a consumer whose toolchain is below any
    dependency's declared `rust-version`, whether or not that dependency's code
    truly needs it. The declaration IS the constraint the consumer meets.

    Dev-dependencies are excluded because a consumer never compiles them, which
    is the same reason the sufficiency legs run `build` rather than `test`.
    """
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1"],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    pk = {p["id"]: p for p in meta["packages"]}
    res = {n["id"]: n for n in meta["resolve"]["nodes"]}
    ws = set(meta["workspace_members"])
    return {
        pk[pid]["name"]: _highest_declared(_reachable_non_dev(pid, res), pid, pk, ws)
        for pid in ws
    }


def minimality_legs(packages):
    """One leg per crate whose floor can be shown necessary by a build.

    Emits `below` = floor minus one minor, and the highest floor DECLARED by the
    crate's non-dev dependencies. The job asks the crate to FAIL to compile at
    `below`; if it succeeds, the code does not need the floor -- but the floor
    may still be necessary, because `--ignore-rust-version` ignores dependency
    floors too and a consumer at `below` would be refused by cargo. So a
    successful build is only "too high" when no dependency reaches above `below`
    either. Without this, a crate whose floor came SOLELY from a dependency
    would red on a correct floor, which is the failure direction that gets
    guards switched off.
    """
    dep = dependency_floors()
    legs = []
    for p in packages:
        floor = _parse(p["rust_version"])
        edition = EDITION_FLOOR.get(str(p.get("edition", "")))
        if floor is None or edition is None:
            print(
                f"note: {p['name']} declares rust-version {p['rust_version']!r} "
                f"and edition {p.get('edition')!r}; not a shape this can step "
                f"below, so no minimality leg",
                file=sys.stderr,
            )
            continue
        below = (floor[0], floor[1] - 1)
        if below < edition:
            print(
                f"note: {p['name']} floor {p['rust_version']} is the edition "
                f"{p['edition']} floor, so it is minimal by construction -- "
                f"there is no lower version to fail at",
                file=sys.stderr,
            )
            continue
        dfloor, dwho = dep.get(p["name"], (None, None))
        if dfloor:
            print(
                f"note: {p['name']} highest dependency floor is "
                f"{dfloor[0]}.{dfloor[1]} ({dwho})",
                file=sys.stderr,
            )
        legs.append({"name": p["name"],
                     "msrv": p["rust_version"],
                     "below": f"{below[0]}.{below[1]}",
                     "dep_floor": f"{dfloor[0]}.{dfloor[1]}" if dfloor else "",
                     "dep_by": dwho or ""})
    return sorted(legs, key=lambda e: e["name"])


matrix = sorted(
    ({"name": p["name"], "msrv": p["rust_version"]} for p in declared),
    key=lambda e: e["name"],
)
for entry in matrix:
    print(f"note: {entry['name']} declares {entry['msrv']}", file=sys.stderr)

if "--verdict" in sys.argv:
    # `--verdict <built_below:0|1> <below> <dep_floor>` -- the CI job asks
    # rather than reimplementing the rule in shell, so there is one copy.
    a = sys.argv[sys.argv.index("--verdict") + 1:]
    if len(a) < 2:
        sys.exit("usage: --verdict <built_below:0|1> <below> [dep_floor]")
    print(minimality_verdict(a[0] == "1", a[1], a[2] if len(a) > 2 else ""))
elif "--minimality" in sys.argv:
    legs = minimality_legs(declared)
    # Unlike the sufficiency matrix, ZERO legs is a legitimate answer: every
    # floor could be an edition floor. But zero legs for a workspace that has a
    # floor above its edition is the silent-empty-matrix defect, so say which.
    print(f"note: {len(legs)} minimality leg(s) of {len(declared)} declared "
          f"floor(s)", file=sys.stderr)
    print(json.dumps(legs, separators=(",", ":")))
else:
    print(json.dumps(matrix, separators=(",", ":")))
