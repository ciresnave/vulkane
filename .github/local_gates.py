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
The MSRV legs add one more: `probe_toolchain` runs `cargo +<floor> --version` to
make a toolchain identify itself. Bandit flags it twice -- B603 for a subprocess
call and B607 for a partial executable path -- and both are stated here rather
than suppressed, for the same reason as above.

B607 is asking for an absolute path to `cargo`. That would defeat the check. The
rustup SHIM is the mechanism under test: `+<floor>` is meaningful only to it, and
resolving past it would probe a different binary from the one the MSRV legs use.
The whole point is to confirm the shim hands back the floor that was asked for.

B603's untrusted input is the floor string, which comes from the `rust-version`
field of this workspace's own manifests via `.github/msrv_matrix.py` -- the same
tree whose `build.rs` a plain `cargo build` already runs.

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
# Codacy flags implicit string concatenation inside a list, and the rule is
# right in general: a dropped comma between two string literals joins them
# silently instead of erroring. It fired here on PR #42 and was carried, so the
# argument is recorded at the call site rather than in a review thread -- the
# annotation is reachable only through the check-runs API, and nobody re-reading
# this table will go there.
#
# What was actually measured, on that PR head:
#
#   * a dropped comma BETWEEN rows is not silent at all. `(...)(...)` is a call,
#     so the module raises TypeError on import and the harness never runs.
#   * a dropped comma INSIDE a row joins two reason fragments and leaves arity
#     at 3, so an arity check cannot see it. That is the real hole, and it is
#     the one the rule is pointing at.
#   * the consequence here is bounded: the joined field is the REASON, which is
#     prose. Pattern and action decide behaviour and sit ahead of it, where a
#     dropped comma is a syntax error.
#
# So the fix is not to ban the pattern but to remove the ambiguity: every
# continuation below is joined with an explicit `+`. The rule stops firing, the
# intent stops being inferable, and a future multi-line reason does not reopen
# it. A known-red that recurs trains readers to skip the check, which costs more
# than the character it saves.
POLICIES = [
    ("pip install", SKIP,
     "runner provisioning; a gate that installs things makes a local check depend" +
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
     "this harness checking its own guards -- it is what fails in CI when a job" +
     " gains no policy"),
    ("cargo fmt", RUN, ""),
    ("cargo clippy", RUN, ""),
    ("cargo doc", RUN, ""),
    ("--test shaderc_test", SKIP,
     "needs a system libshaderc or a source build of glslang"),
    ("--test slang_test", SKIP, "needs the Slang toolchain"),
    ("cargo package --list", RUN,
     "lists what WOULD be published without building or touching a device; the" +
     " bundled-vk.xml assertion it guards is worth running locally too"),
    ("cargo test", GPU, ""),
    ("cargo run", GPU, ""),
    ("cargo build", RUN, ""),
]


def probe_toolchain(name):
    """Ask a toolchain to identify itself. Returns the CompletedProcess.

    ONE call site rather than four, so `--self-test` exercises the function the
    MSRV legs actually call instead of a copy of it. A duplicated probe can drift
    from the real one and keep passing, which is the failure this file exists to
    avoid.

    `cargo` is resolved from PATH deliberately. The rustup shim IS the mechanism
    under test -- `+name` means nothing to anything else -- so an absolute path
    would be testing something other than what the MSRV legs use. See the trust
    boundary note at the top of this file.
    """
    return subprocess.run(["cargo", "+" + name, "--version"],
                          capture_output=True, text=True)


def toolchain_verdict(install_code, probe_code):
    """Is an MSRV floor usable? Two exit codes, and they answer DIFFERENT things.

    `install_code` is rustup's, and rustup attempts a self-update during install.
    On a shared box that fails ("could not remove 'rustup-bin'") whenever another
    session holds rustup.exe, one line after reporting the toolchain installed and
    unchanged -- so a nonzero install code routinely means nothing about the
    toolchain. `probe_code` is `cargo +<floor> --version`: the toolchain being
    asked to identify itself, which is the property a build depends on.

    Lifted out of the loop because the interesting case only occurs when rustup
    happens to be locked. A branch reachable only on a bad day is one a green run
    never exercises, so the run cannot be the evidence -- this can.
    """
    if probe_code != 0:
        return "unusable"
    if install_code != 0:
        return "note"
    return "ok"


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
    """Expand a `${{ matrix.crate.* }}` command over the right matrix.

    There are TWO matrices now and they are not interchangeable. The sufficiency
    legs carry `msrv`; the minimality legs also carry `below`, and cover only the
    crates whose floor can be stepped beneath. Expanding a minimality command
    over the sufficiency matrix would substitute nothing for `below`, leaving a
    literal `${{ ... }}` in the command line and running SOMETHING -- which is
    the shape where a gate reports about a thing nobody chose.

    Selected by what the command actually references rather than by job name, so
    a renamed job does not silently pick the wrong matrix.
    """
    minimality = "matrix.crate.below" in command
    argv = [sys.executable, str(REPO / ".github" / "msrv_matrix.py")]
    if minimality:
        argv.append("--minimality")
    raw = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    if raw.returncode != 0:
        raise SystemExit(
            "msrv_matrix.py failed, so the msrv legs cannot be expanded:\n" + raw.stderr
        )
    expanded = []
    for leg in json.loads(raw.stdout):
        cmd = command.replace("${{ matrix.crate.msrv }}", leg["msrv"])
        cmd = cmd.replace("${{ matrix.crate.name }}", leg["name"])
        if "below" in leg:
            cmd = cmd.replace("${{ matrix.crate.below }}", leg["below"])
        # A leftover placeholder means the matrix did not carry a field the
        # command asked for. Running it anyway would execute a command nobody
        # wrote, so refuse instead.
        if "${{" in cmd:
            raise SystemExit(
                "an unexpanded matrix placeholder survived expansion, so the "
                "command would not be the one CI runs:\n  " + cmd.strip()
            )
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
    # NOT suppressed, because it cannot be: `# nosemgrep` was tried both on the
    # preceding line and on the matched line, and Sourcery honours neither. A
    # directive that does nothing is worse than none -- it reads as though the
    # finding were handled. So the finding stands, red, correct, and answered in
    # the docstring above and in the PR thread.
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
                # A minimality leg runs at `below`, not at the declared floor.
                # Installing and probing `msrv` here would verify 1.88 and then
                # run a 1.87 command -- a guard confirming a different toolchain
                # from the one under test, which is worse than no guard.
                want = leg.get("below", leg["msrv"])
                # Read the install result. Ignoring it lets the build below run on
                # whatever toolchain happened to be present and report a pass or a
                # fail about a version nobody chose -- an exit code produced and
                # never read, which is the same defect as treating gpu-run's
                # EX_TEMPFAIL as a gate result, one layer over.
                install = subprocess.run(
                    ["rustup", "toolchain", "install", want, "--profile", "minimal"],
                    capture_output=True, text=True,
                )
                # `install.returncode` answers "did rustup exit cleanly", which
                # is NOT the question the build below depends on. rustup attempts
                # a SELF-UPDATE during install, and on a shared box that fails
                # with "could not remove 'rustup-bin' ... Access is denied"
                # whenever another session holds rustup.exe -- one line after
                # reporting the toolchain installed and unchanged. Reading the
                # exit code turned three already-present toolchains into three
                # red MSRV legs, which is a false red, and a harness that cries
                # wolf locally is one people stop running.
                #
                # Ask the toolchain to identify itself instead. That is the
                # property the build actually needs, and it is false exactly when
                # the install really did fail.
                probe = probe_toolchain(want)
                verdict = toolchain_verdict(install.returncode, probe.returncode)
                if verdict == "unusable":
                    failures.append(label + " (" + leg["name"] + " @ " + leg["msrv"]
                                    + " @ " + want + ") toolchain is not usable (rustup exit="
                                    + str(install.returncode) + "): "
                                    + (probe.stderr.strip()
                                       or install.stderr.strip())[:160])
                    ran += 1
                    continue
                if verdict == "note":
                    print("  note  rustup exited " + str(install.returncode)
                          + " installing " + want
                          + ", but the toolchain answers: " + probe.stdout.strip())
                print("  RUN   [" + job + "] " + leg["name"] + " @ " + want
                      + (" (must FAIL)" if "below" in leg else ""))
                code = shell(expanded, env)
                if code != 0:
                    # `want`, not `leg["msrv"]`: a minimality leg runs at
                    # `below`, and a failure line naming the declared floor
                    # sends the reader to a version that was never executed.
                    failures.append(label + " (" + leg["name"] + " @ " + want
                                    + ") exit=" + str(code))
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

    # 1b. every POLICIES row is shaped (pattern, action, reason).
    #
    # Deliberately NOT sold as covering the concatenation footgun in the header
    # comment above -- it does not, and saying so would be worse than leaving it
    # unchecked. A joined reason keeps arity at 3 and passes this untouched.
    # What it does catch is a row edited by hand into a shape `classify` would
    # then read wrongly: a lost reason, an action that is not one of the four,
    # a non-string reason, and an EMPTY pattern -- which is the dangerous one,
    # because `"" in command` is true for every command, so one empty pattern
    # silently classifies the entire workflow as whatever that row says.
    shapes = [(i, r) for i, r in enumerate(POLICIES)
              if not (isinstance(r, tuple) and len(r) == 3
                      and isinstance(r[0], str) and r[0]
                      and r[1] in (RUN, GPU, SKIP, MSRV)
                      and isinstance(r[2], str))]
    check("every POLICIES row is (pattern, action, reason)", not shapes)

    # The probe is only a check if it can come back false, so drive it both ways
    # through the SAME function the MSRV legs call. The subject is the pinned
    # channel read out of rust-toolchain.toml -- a file read rather than another
    # subprocess, and a fact this repo already owns rather than "whatever happens
    # to be active", which varies per shell.
    pinned = ""
    toolchain_file = REPO / "rust-toolchain.toml"
    if toolchain_file.exists():
        for line in toolchain_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("channel"):
                pinned = line.split("=", 1)[1].strip().strip("\"'")
                break
    check("the pinned channel is readable, so the control has a subject",
          bool(pinned))
    if pinned:
        check("the toolchain probe answers for an installed toolchain",
              probe_toolchain(pinned).returncode == 0)
    check("the toolchain probe FAILS for a toolchain that does not exist",
          probe_toolchain("not-a-real-toolchain-xyzzy").returncode != 0)

    # 1d. the four (install, probe) combinations, including the one a green run
    # never reaches. The gates passing does NOT show that a locked rustup is
    # tolerated -- on a run where rustup exits 0, that branch is never entered.
    verdicts = {(0, 0): "ok",        # ordinary
                (1, 0): "note",      # rustup could not self-update; toolchain fine
                (1, 1): "unusable",  # install genuinely failed
                (0, 1): "unusable"}  # installer "succeeded", toolchain absent
    wrong = [(k, toolchain_verdict(*k)) for k, want in verdicts.items()
             if toolchain_verdict(*k) != want]
    check("a locked rustup with a working toolchain is a note, not a failure",
          toolchain_verdict(1, 0) == "note")
    check("a toolchain that cannot answer is unusable however rustup exited",
          not wrong)

    # 1e. the two MSRV matrices are different populations, and a command is
    # routed by what it references. Getting this wrong would expand a
    # minimality command over the sufficiency matrix and run a version nobody
    # chose, which is the defect the MSRV job exists to prevent.
    suff = msrv_invocations("X ${{ matrix.crate.name }} ${{ matrix.crate.msrv }}")
    mini = msrv_invocations("X ${{ matrix.crate.name }} ${{ matrix.crate.below }}")
    check("the sufficiency matrix has legs", len(suff) > 0)
    check("every minimality leg carries a `below`",
          all("below" in leg for _, leg in mini))
    check("no minimality leg is expanded from the sufficiency matrix",
          all("below" not in leg for _, leg in suff))
    check("expansion leaves no unsubstituted placeholder",
          not any("${{" in cmd for cmd, _ in suff + mini))
    check("a minimality leg steps BELOW its declared floor",
          all(leg["below"] != leg["msrv"] for _, leg in mini))

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
    # `cargo package --list` neither builds nor enumerates a device, so locking
    # it behind the GPU mutex would serialise a check that needs no hardware.
    check("cargo package --list runs locally without the GPU lock",
          classify("list=$(cargo package --list -p vulkane --allow-dirty)")[0] == RUN)
    # The no-source step moves vk.xml aside and asserts the build FAILS. It must
    # classify off its `cargo build`, not fall through to unclassified -- an
    # unclassified step is fatal, which is how this harness caught it being added.
    check("the no-source negative step classifies as a plain local run",
          classify('stash="${RUNNER_TEMP:-/tmp}/vk.xml.stashed" mv vulkane/vk.xml '
                   '"$stash" out=$(cargo build -p vulkane 2>&1)')[0] == RUN)

    if failures:
        sys.exit("self-test FAILED: " + "; ".join(failures))
    print("  self-test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
