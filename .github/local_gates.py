#!/usr/bin/env python3
"""Run CI's gates locally, extracted from the workflow rather than listed here.

Written because a *list* of local checks is a note, and notes decay silently.
Mine did: `fmt`, `clippy` and two test runs were green, I reported "all exit 0",
and `Build docs` -- the fifth thing CI runs -- went red on the very next push.
The four commands were each honest; the summary of them was not, and nothing
about the four could have revealed the omission. **A local verification set is a
claim about CI's job list, and it decays every time CI gains a job.**

So this holds no list of commands. It reads every `run:` step out of
`.github/workflows/ci.yml` and decides what to do with each. What it *does* hold
is a table of POLICIES, and that table is self-policing: **a command matching no
policy is a hard failure, not a skip.** A new CI job therefore forces a decision
here instead of being quietly absent -- which is the exact defect this replaces.

# Why this is not a faithful replay of CI, deliberately

Copying CI's environment verbatim would be wrong for this repo, and wrong in a
way that looks right. The workflow sets `GPU_RUN_UNGUARDED` at workflow level --
correct there, because a CI runner is ephemeral and single-tenant, so there is no
shared GPU to serialize against. **Setting it here would switch off the
machine-wide serialization guard on a developer box, which is the one environment
it exists for.** Device-touching commands run under `gpu-run.ps1` instead. That
divergence is the point, so it is stated rather than silently applied.

The env that IS replicated is the part whose absence weakens the check:
`RUSTDOCFLAGS=-D warnings` is what turns an intra-doc link to a private item into
a failure. A local `cargo doc` without it passes. **A harness that runs a weaker
configuration than CI reports green on a configuration nobody ships.**

# What this executes, and the trust boundary

This runs the commands written in `.github/workflows/ci.yml`. That is the point,
and it is also the whole security story: a static-analysis pass will flag the
`subprocess.run` here as a call without a literal argument, correctly, because
the argument comes from a file.

The file is the workflow CI already executes with more privilege than you have.
Anyone who can change it can already run code in CI, so running it here is not an
escalation ON A BRANCH YOU TRUST. **On a branch you do not trust it is arbitrary
code execution on your machine** -- but so is `cargo test`, which runs that
branch's `build.rs`. The rule is the same one Rust already asks of you: do not
run a build, a test, or this, on a branch you would not review first.

Stated rather than suppressed, because a security finding waved away as a false
positive is indistinguishable from one nobody read.

Never pipe this. It is a gate, and a pipe turns its exit status into the exit
status of whatever you piped into.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
GPU_RUN_ENV = "VULKANE_GPU_RUN"
# Where the machine-wide GPU serialization wrapper lives. The default is where
# it sits on the machine this was written on; anywhere else, set the env var.
# A hard-coded absolute path would make the harness behave differently per
# machine, which is the same drift this file exists to remove -- by machine
# instead of by time.
GPU_RUN = Path(os.environ.get(GPU_RUN_ENV, "C:/Projects/fuel/scripts/gpu-run.ps1"))

RUN = "run"
GPU = "gpu"
SKIP = "skip"
MSRV = "msrv"

# `gpu-run.ps1` passes the child's exit code through, EXCEPT for this one:
# EX_TEMPFAIL, meaning it could not take the machine-wide gpu-run mutex.
# That mutex is shared across every project on this box, so a busy
# neighbour is a normal outcome and NOT a statement about this repo.
# Counting it as a gate failure would be the same conflation as calling a
# line-ending bug a captured toolchain: a real non-zero wearing the wrong
# cause's name.
EX_TEMPFAIL = 75

# (substring, policy, reason). First match wins; order matters.
#
# This table CAN drift from the workflow -- a reworded command stops matching.
# That is why an unmatched command is fatal rather than skipped: drift here is
# loud, and loud drift is a different thing from silent absence.
POLICIES = [
    ("pip install", SKIP,
     "runner provisioning; a gate that installs things makes a local check depend"
     " on the network"),
    ("apt-get", SKIP,
     "provisions a Linux runner; this box is Windows with the SDK already installed"),
    ("$GITHUB_OUTPUT", SKIP,
     "CI plumbing, not a gate; the emitter it wraps is exercised by the msrv legs"),
    ("rustup toolchain install", SKIP,
     "the msrv policy installs what it needs before invoking it"),
    ("${{ matrix.", MSRV,
     "expanded from .github/msrv_matrix.py, one invocation per declared floor"),
    ("rustup show active-toolchain", RUN,
     "asserts the pin is the mechanism; works identically here"),
    ("local_gates.py --self-test", RUN,
     "this harness checking its own guards -- it is what fails in CI when a job"
     " gains no policy"),
    ("cargo fmt", RUN, ""),
    ("cargo clippy", RUN, ""),
    ("cargo doc", RUN, ""),
    ("--test shaderc_test", SKIP,
     "needs a system libshaderc or a source build of glslang"),
    ("--test slang_test", SKIP, "needs the Slang toolchain"),
    ("cargo test", GPU, ""),
    ("cargo run", GPU, ""),
    ("cargo build", RUN, ""),
]


def steps():
    """Every `run:` step in the workflow, with the env CI gives it."""
    try:
        import yaml
    except ImportError:
        # The person who hits this is the person the tool is for: CI installs
        # pyyaml explicitly, so only a local run can lack it. A traceback here
        # costs most exactly where it helps least.
        sys.exit("error: this needs pyyaml to read the workflow. Install it with:"
                 "\n  " + sys.executable + " -m pip install pyyaml")

    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    top = {k: str(v) for k, v in (doc.get("env") or {}).items()}
    found = []
    for job_id, job in doc["jobs"].items():
        for step in job.get("steps", []):
            if "run" in step:
                env = dict(top)
                env.update({k: str(v) for k, v in (step.get("env") or {}).items()})
                found.append((job_id, step.get("name", ""), step["run"], env))
    return found


def classify(command):
    for needle, policy, reason in POLICIES:
        if needle in command:
            return policy, reason
    return None, None


def ci_env(step_env):
    """CI's environment, minus the one variable that must not cross over.

    `GPU_RUN_UNGUARDED` is the reason-bearing escape hatch the serialization
    guard honours. It belongs on an ephemeral single-tenant runner and nowhere
    else. An unexpanded `${{ ... }}` is a CI expression, not a value, so it is
    dropped rather than exported literally.
    """
    env = dict(os.environ)
    # POP, not merely decline-to-add. Declining only stops it arriving from the
    # step; it does nothing about an ambient one. CI sets this at workflow level,
    # so on a runner it is already in `os.environ` and an inherit-then-filter
    # would pass it straight through -- and any developer with it exported would
    # silently lose the guard this whole harness routes around. The self-test
    # caught exactly that, in CI, which is the only place the variable is set.
    env.pop("GPU_RUN_UNGUARDED", None)
    for key, value in step_env.items():
        if key == "GPU_RUN_UNGUARDED" or "${{" in value:
            continue
        env[key] = value
    return env


def msrv_invocations(command):
    """Expand a `${{ matrix.crate.* }}` command over the declared floors."""
    raw = subprocess.run(
        [sys.executable, str(REPO / ".github" / "msrv_matrix.py")],
        capture_output=True, text=True, cwd=REPO,
    )
    if raw.returncode != 0:
        raise SystemExit(
            "msrv_matrix.py failed, so the msrv legs cannot be expanded:\n" + raw.stderr
        )
    expanded = []
    for leg in json.loads(raw.stdout):
        cmd = command.replace("${{ matrix.crate.msrv }}", leg["msrv"])
        cmd = cmd.replace("${{ matrix.crate.name }}", leg["name"])
        expanded.append((cmd, leg))
    return expanded


def shell(command, env):
    """Run one workflow command through bash, as `shell: bash` steps do.

    `command` is not a literal, and a security scan is right to say so. It comes
    from `.github/workflows/ci.yml` -- the workflow CI already executes with more
    privilege than the developer running this. On a branch you trust that is not
    an escalation; on a branch you do not, it is arbitrary code execution, and so
    is `cargo test` running that branch's `build.rs`.

    Suppressed WITH THE REASON AT THE SITE, which is this repository's convention
    for a reason-bearing exemption (`GPU_RUN_UNGUARDED` in the workflow,
    `GPU-LOCK-DIRECT:` in the test suite). An exemption whose justification lives
    somewhere else is one the next reader has to take on trust; the full argument
    is in the module docstring under "the trust boundary".
    """
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    return subprocess.run(["bash", "-lc", command], cwd=REPO, env=env).returncode


def main():
    collected = steps()

    # An empty extraction is indistinguishable from a passing run. If the parse
    # breaks or the workflow is restructured, this must go red rather than
    # report success over nothing.
    if not collected:
        sys.exit("error: no `run:` steps found in the workflow -- refusing to "
                 "report success over an empty gate list")

    unclassified = [(j, c) for j, _, c, _ in collected if classify(c)[0] is None]
    if unclassified:
        print("error: CI runs commands this harness has no policy for.\n", file=sys.stderr)
        for job, command in unclassified:
            print("  [" + job + "] " + " ".join(command.split())[:100], file=sys.stderr)
        sys.exit("\nAdd a policy in POLICIES saying whether it runs locally and, if not,"
                 " why.\nThis is fatal on purpose: a CI job nobody decided about is"
                 " exactly the gap\nthis script replaces.")

    failures, unavailable, ran, skipped = [], [], 0, 0
    for job, name, command, step_env in collected:
        policy, reason = classify(command)
        label = "[" + job + "] " + (name or " ".join(command.split())[:60])

        if policy == SKIP:
            print("  SKIP  " + label + "\n          " + reason)
            skipped += 1
            continue

        env = ci_env(step_env)

        if policy == MSRV:
            for expanded, leg in msrv_invocations(command):
                # Read the install result. Ignoring it lets the build below run on
                # whatever toolchain happened to be present and report a pass or a
                # fail about a version nobody chose -- an exit code produced and
                # never read, which is the same defect as treating gpu-run's
                # EX_TEMPFAIL as a gate result, one layer over.
                install = subprocess.run(
                    ["rustup", "toolchain", "install", leg["msrv"], "--profile", "minimal"],
                    capture_output=True, text=True,
                )
                if install.returncode != 0:
                    failures.append(label + " (" + leg["name"] + " @ " + leg["msrv"]
                                    + ") toolchain install failed: "
                                    + install.stderr.strip()[:160])
                    ran += 1
                    continue
                print("  RUN   [" + job + "] " + leg["name"] + " @ " + leg["msrv"])
                code = shell(expanded, env)
                if code != 0:
                    failures.append(label + " (" + leg["name"] + " @ "
                                    + leg["msrv"] + ") exit=" + str(code))
                ran += 1
            continue

        if policy == GPU:
            if not GPU_RUN.exists():
                sys.exit("error: " + str(GPU_RUN) + " is missing, and running a "
                         "device-touching command without it would bypass the "
                         "serialization guard. Refusing.")
            command = 'pwsh "' + str(GPU_RUN) + '" -Project vulkane -- ' + command

        print("  RUN   " + label)
        code = shell(command, env)
        if code == EX_TEMPFAIL and policy == GPU:
            unavailable.append(label)
        elif code != 0:
            failures.append(label + "  exit=" + str(code))
        ran += 1

    # `ran` can exceed `extracted - skipped`: one msrv step expands to one
    # invocation per declared floor, exactly as CI fans it out over the matrix.
    print("\n  %d workflow steps -> %d invocations, %d skipped, %d failed"
          % (len(collected), ran, skipped, len(failures)))
    if unavailable:
        print("\n  GPU lock unavailable -- these did not run, and did not fail:")
        for u in unavailable:
            print("    " + u)
        print("    another project holds the machine-wide lock; retry later")
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        sys.exit(1)
    print("  all gates that run on this box passed")


def self_test():
    """Exercise this harness's own guards.

    The guards are the reason to trust the rest of it, and a guard nobody runs
    is the defect this whole file exists to remove. Each case here is one I
    checked by hand once; running them by hand once is exactly how the previous
    verification set decayed.
    """
    failures = []

    def check(name, ok):
        print(("  ok   " if ok else "  FAIL ") + name)
        if not ok:
            failures.append(name)

    # 1. every command CI runs today has a policy
    collected = steps()
    unclassified = [c for _, _, c, _ in collected if classify(c)[0] is None]
    check("every current CI command is classified", not unclassified)
    check("the workflow parse found commands at all", len(collected) > 0)

    # 2. a command with no policy is detected (the whole point)
    check("an unknown command is unclassified, not defaulted",
          classify("some-new-tool --check --strict")[0] is None)

    # 3. the escape hatch must not cross over, and the strict flags must
    expr = "${{ runner.os == 'Linux' }}"
    # Set it AMBIENTLY for the duration, because that is the case that failed:
    # on a runner the workflow-level env is already in `os.environ`, and a test
    # that only passes it via step_env exercises the easy half.
    os.environ["GPU_RUN_UNGUARDED"] = "ambient, as a runner would have it"
    try:
        env = ci_env({"GPU_RUN_UNGUARDED": "x", "RUSTDOCFLAGS": "-D warnings",
                      "VULKANE_REQUIRE_DEVICE": expr})
    finally:
        os.environ.pop("GPU_RUN_UNGUARDED", None)
    check("GPU_RUN_UNGUARDED does not reach a local run",
          "GPU_RUN_UNGUARDED" not in env)
    check("RUSTDOCFLAGS=-D warnings does reach a local run",
          env.get("RUSTDOCFLAGS") == "-D warnings")
    check("an unexpanded CI expression is dropped, not exported literally",
          "VULKANE_REQUIRE_DEVICE" not in env)

    # 4. device-touching policies really are marked GPU, not RUN
    check("cargo test is routed through the serialization lock",
          classify("cargo test --workspace")[0] == GPU)
    check("cargo build is not needlessly locked",
          classify("cargo build -p vulkane")[0] == RUN)

    if failures:
        sys.exit("self-test FAILED: " + "; ".join(failures))
    print("  self-test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
