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

matrix = sorted(
    ({"name": p["name"], "msrv": p["rust_version"]} for p in declared),
    key=lambda e: e["name"],
)
for entry in matrix:
    print(f"note: {entry['name']} declares {entry['msrv']}", file=sys.stderr)

print(json.dumps(matrix, separators=(",", ":")))
