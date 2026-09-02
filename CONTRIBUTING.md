# Contributing to Vulkane

Thank you for your interest in contributing to Vulkane! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Install development dependencies:
   - Rust **via rustup**. `rust-toolchain.toml` pins the toolchain this repo is
     *verified with*, and rustup installs it on your first `cargo` command, so
     there is nothing to choose. That is a different claim from the per-crate
     `rust-version` floors below, which are what a **consumer** may compile
     with. Neither number implies the other and CI exercises both.
   - **A `cargo` that did not come from rustup ignores the pin entirely** — a
     distro package, a vendored toolchain, anything without rustup's shim. The
     file is not an error there; it is simply not read, so you silently build
     with whatever that toolchain is. If that is you, the workspace needs 1.88 —
     the highest floor any member declares. Each crate's `rust-version` is the
     authority and CI exercises every one: `vulkane` and `vulkan_gen` need 1.88
     (let-chains, and `libloading 0.9` declares 1.88 itself); `kiss-vulkan-vocab`
     and `vulkane_derive` still build on 1.85.
   - Vulkan SDK 1.4.316 or later
   - CMake 3.20 or later and a C++ toolchain — only needed for the `shaderc`
     and `slang` features, which build their compilers from source. The
     examples themselves are pure Rust and need neither.

2. Clone and build:

   ```bash
   git clone https://github.com/ciresnave/vulkane.git
   cd vulkane
   cargo build
   ```

3. Run tests:

   ```bash
   cargo test                          # core suite
   cargo test --features naga          # GLSL front-end
   cargo test --features kiss-target   # KISS `vulkan:` token derivation
   ```

   `cargo test --all-features` also works but pulls in `shaderc` and `slang`,
   so it needs the C++ toolchain above and takes considerably longer.

   Note that much of the suite enumerates and runs work on a **real Vulkan
   device** rather than mocking one. Expect it to fail on a machine with no
   Vulkan-capable GPU or no installed ICD, and avoid running two GPU-touching
   suites concurrently.

## Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` with the project's configuration
- Run `clippy` with all lints enabled
- Maintain comprehensive documentation

## Documentation Requirements

1. Every public API must have:
   - Clear purpose and usage examples
   - Parameter descriptions
   - Safety section for unsafe functions
   - Error conditions and handling

2. Include doc tests demonstrating usage:

```rust
/// Allocate `size` bytes of device memory from the given memory type.
///
/// # Examples
///
/// ```no_run
/// use vulkane::safe::{Device, DeviceMemory};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let device: Device = unimplemented!();
/// # let memory_type_index = 0;
/// let memory = DeviceMemory::allocate(&device, 1024, memory_type_index)?;
/// # Ok(())
/// # }
/// ```
```

Pick the type index with `PhysicalDevice::find_memory_type`, and prefer
`DeviceMemory::allocate_with` when you need to pass a `MemoryAllocateInfo`
(for a dedicated allocation or an exportable handle, say).

## Testing Requirements

1. Unit Tests:
   - Test each public API
   - Cover error cases
   - Mock external dependencies

2. Integration Tests:
   - Test realistic usage patterns
   - Verify component interactions
   - Test platform-specific features

3. Performance Tests:
   - Benchmark critical operations
   - Compare against baselines
   - Document performance characteristics

## Adding a gate

Every gate in this repository is demonstrated FAILING before it is trusted. Not
as ceremony: a check that has only ever been run against working code has shown
that it runs, which is a different claim from showing that it catches anything.

    cargo package --list must contain vk.xml
      -> proven by adding `exclude = ["vk.xml"]` and watching it red

    the crate must not build one minor below its floor
      -> proven by raising a floor nothing requires and watching it red

    the changelog must name the version in Cargo.toml
      -> proven at 64001e7, the commit where that was actually false

The last one is the strongest form: born-red against **the defect that really
happened, at the commit where it happened**. A synthetic reproduction shows the
check runs; the real one shows it catches *this*.

### A check that cannot find its subject must FAIL, not pass

Put the positive control first, before the assertion it protects. If the shape
being parsed ever changes, the real assertion compares against nothing — and
comparing against nothing passes.

    total="$(grep -c '^## \[' CHANGELOG.md || true)"
    if [ "$total" -lt 2 ]; then exit 1; fi   # <- before the real checks

The same rule applies to a null result. "I searched and found nothing" is not a
finding until the same query is shown finding something you know is there:
*0 hits for X; control: the same query finds Y, which I know is present.*

### How checks in this repository have actually passed without checking

Nine shapes, each with the instance it came from here. **Eight of the nine
produced a REASSURING result** — which is why each needed a control rather than
a closer look. A worrying result invites scrutiny for free; a reassuring one is
self-certifying, and re-reading it only re-confirms it.

1. **A filter between a check and its exit code.**
   `git check-ignore -v path | sed ... || echo ABSENT` — `sed` succeeds on empty
   input, so the `||` never fires and an absence prints nothing, which reads as
   confirmation. Keep the exit code; never pipe the thing you are testing.
2. **A query whose shape cannot match.** A single-line grep for let-chains found
   **0 sites in all four crates**, which would have "proved" the documented MSRV
   reason false. The query was wrong, not the answer; multiline found 17.
3. **Matching the label instead of the outcome.** `grep -c FAIL` matched the
   *name* of a passing assertion, "the probe FAILS for a toolchain that does not
   exist". Caught only because the exit code disagreed with the grep.
4. **Two tools disagreeing about a path.** Windows-Python's `/tmp` is `C:\tmp`,
   not Git-Bash's. A file written by one and read by the other is silently
   absent, so a comparison reports "differ" and a fixture reads as empty.
5. **A fixture that cannot produce the defect.** `printf` turned `\n` into a
   real newline, so the "escaped form" test input was not escaped. A born-red
   whose broken input is not actually broken is a control that cannot fire.
6. **The right control in the wrong environment.** A guard's born-red passed in
   a shell with no `CARGO_TERM_COLOR`; CI sets it to `always`, cargo prefixes
   diagnostics with ANSI escapes, and an anchored `^error\[E` then matches
   nothing. Run controls under `.github/local_gates.py`, which replays CI's
   workflow-level env, rather than in a bare shell.
7. **A different question that agrees most of the time.**
   `git diff main <branch>` for "is it merged" reports differences in *both*
   directions, so a merged branch that main has moved past looks unmerged. Use
   `gh pr list --head <branch>`. Likewise `git rev-parse --abbrev-ref HEAD`
   answers "where is the worktree pointed", not "what contains this commit" --
   that is `git branch --contains`.
8. **The mode you were not looking at.** An edit broke `msrv_matrix.py`'s
   default output while every targeted check exercised only `--minimality`.
   Green everywhere it was looked at. Run the gates, not the check you wrote.
9. **Repairing with the mechanism that caused it.** A quoting failure fixed by
   rewriting the same literal in the same syntax fails the same way. Change the
   mechanism -- a quoted heredoc, a file, a different tool -- not the value.

### Individually green is not the same claim as green

Several pull requests each passing alone have not been run together. After a
run of merges, run `python .github/local_gates.py` on `main` itself: that is a
separate measurement, and it is the one that describes what people get.

### Gate what the change can break

A documentation-only change to a file that is not compiled, not packaged and
read by no test cannot be informed by the GPU gates. Do not run them for
appearances -- they consume a machine-wide lock and add a green row that a later
reader will mistake for evidence.

**State the omission with its control**, in the pull request, in this form:

    grep -rl '<the file>' --include=*.rs --include=*.yml --include=*.py
    grep -rl '<something you know is referenced>' --include=... # the control

"0 hits, and the same query finds N for a string I know is there" is a
measurement. "It is only documentation" is an assertion.

**Run it; do not copy a number out of here.** This paragraph originally quoted
"0 references for `CONTRIBUTING.md`" as its example, measured minutes before
being written. It was 1 by the time it was checked -- because the changelog gate
added in #50 prints *"see the Release Process in CONTRIBUTING.md, step 3"*, so
the pull request that motivated this section changed the number the section
cited. **A count in prose is a measurement pinned to a moment, in a place
nothing re-measures.**

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes:
   - Follow code style guidelines
   - Add tests
   - Update documentation
4. Submit a pull request:
   - Describe the changes
   - Link related issues
   - Add test results

## Development Workflow

1. Check existing issues and PRs
2. Create an issue for new features
3. Write failing tests first
4. Implement the feature
5. Document thoroughly
6. Submit PR for review

## Safety Guidelines

1. Mark unsafe functions appropriately:

   ```rust
   /// Create a buffer from a raw handle
   ///
   /// # Safety
   ///
   /// The caller must ensure:
   /// - The handle is valid
   /// - The handle was created by the same device
   /// - The handle is not used elsewhere
   pub unsafe fn from_raw(
       device: &Arc<Device>,
       handle: VkBuffer,
   ) -> Buffer {
       // Implementation
   }
   ```

2. Document all safety requirements thoroughly
3. Validate inputs wherever possible
4. Use safe abstractions by default

## Performance Guidelines

1. Profile before optimizing
2. Document performance characteristics
3. Add benchmarks for changes
4. Compare against baselines

## Moving an MSRV floor

A crate's `rust-version` is a promise to **consumers**: this is the oldest
compiler that can use the published crate. It is not the toolchain you develop
with -- see Development Setup, where the three numbers that all get called "the
Rust version" are separated.

**The floor is wrong in two directions and only one of them is loud.** Too low
and CI goes red, because the crate does not build there. Too high and everything
is green forever, while consumers who could have compiled are turned away and
never appear to tell you. CI now checks both:

| job | asks | failure means |
|---|---|---|
| `MSRV <floor> - <crate>` | does it build at the floor? | the floor is too LOW |
| `MSRV <floor> is minimal - <crate>` | does it fail one minor below? | the floor is too HIGH |

### When a floor may move

Only when the code or a dependency genuinely requires it. "Newer is tidier" is
not a reason: every bump excludes someone, permanently, for every published
version carrying it.

Check what is actually forcing it before writing a number:

    cargo metadata --format-version 1     # dependencies declare floors too

`vulkane`'s 1.88 has two independent drivers -- let-chains in its own source,
and `libloading 0.9` declaring `rust-version = 1.88.0`. Removing one would not
lower the floor. Knowing which drivers exist is the difference between lowering
a floor and guessing at it.

### What moving one obliges

1. **A CHANGELOG entry under a breaking kind.** Raising a floor is a breaking
   change for consumers even though nothing in the API moved.
2. **Updating the comment beside `rust-version`.** The reason lives at the
   floor, not in this file, because someone bumping a floor edits the manifest
   and never opens this document. State the KIND of reason and no counts: "7
   let-chain sites" is a measurement pinned to a moment in a place nothing
   re-measures.
3. **Expect new clippy lints.** Raising a floor switches on MSRV-gated lints for
   APIs that have fallen under it. 1.85 to 1.88 cost 35 sites across 13 files.
   This is a stream, not a one-off: every clippy release adds more.

### Reading a red minimality leg

**It means the floor is now higher than the code needs.** The repair is to
**lower the declared floor**, never to raise the version the job steps to --
that would silently ratify an unmeasured floor rather than measure it.

Both 1.88 floors were measured minimal by hand before the job existed, so a red
there is a regression and not a check finding its feet.

### Why only two crates have a minimality leg

`kiss-vulkan-vocab` and `vulkane_derive` declare 1.85, which is **edition 2024's
own floor** -- cargo below it cannot read the manifest at all. There is no lower
version to fail at, so they are minimal by construction and `msrv_matrix.py`
emits no leg for them. That is an answer, not a gap; adding a leg there would
measure "cargo 1.84 cannot read edition 2024" and report it as a fact about the
floor.

### Why the job passes `--ignore-rust-version`

Without it, cargo refuses on the declared `rust-version` before compiling
anything, and the job would prove the floor necessary by observing that cargo
obeys the floor -- the number causing the failure that justifies it. Measured:

    cargo +1.87 build -p vulkan_gen
      error: rustc 1.87.0 is not supported by the following package

    cargo +1.87 build --ignore-rust-version -p vulkan_gen
      error[E0658]: `let` expressions in this position are unstable

Only the second is about the code. The job also requires the failure to carry a
compiler diagnostic, because a missing toolchain and an unreadable manifest also
exit non-zero and would otherwise read as "the floor is necessary".

**One limitation.** `--ignore-rust-version` ignores every `rust-version`,
including dependencies'. The leg therefore measures whether the CODE needs the
floor. A crate whose floor came solely from a dependency would be reported too
high; that half is a `cargo metadata` question and is deliberately not folded
into this job.

## Release Process

The six-line version of this list was accurate and still let a release ship
wrong: `vulkane 0.14.0` reached crates.io and was tagged while every entry it
carried was still filed under `## [Unreleased]`, because "update changelog" does
not say what updating one means. Each step below says what to type and what to
check, because a release is done under time pressure by someone who will not
re-derive it.

**crates.io is immutable.** A version cannot be replaced, only yanked and
superseded. A wrong publish costs a version number, so the checks come first.

### 1. Land everything first

`README.md`, `build.rs`, `src/`, `docs/` and `vk.xml` all ship **inside** the
published crate. A doc fix merged after the publish is not in the artifact
people download, and `docs.rs` renders the published copy, not `main`.

    cargo package --list -p vulkane

### 2. Set versions, and know which crates move

    cargo metadata --no-deps --format-version 1

**`vulkane` publishes LAST.** It depends on the other three, and cargo resolves
every dependency from the registry at publish time, so a floor that is not
published yet fails the publish with `exit 101`. Among `kiss-vulkan-vocab`,
`vulkan_gen` and `vulkane_derive` there is no ordering -- they depend on nothing
in this workspace.

Derive that rather than trusting this paragraph; a list of crate names here goes
stale the first time one is added:

    cargo metadata --no-deps --format-version 1 | python -c "import sys,json; \
    m=json.load(sys.stdin); p={x['name'] for x in m['packages']}; \
    print({x['name']: sorted(d['name'] for d in x['dependencies'] if d['name'] in p) \
    for x in m['packages']})"

Note that a dependency already satisfied by a PUBLISHED version does not force a
republish. `vulkane` requires `vulkane_derive = "0.1"`, which `0.1.0` satisfied;
`0.1.1` still had to go first if it was going at all, but it was not blocking.

### 3. Stamp the changelog

Not "update" -- **stamp**. Add a dated release header beneath an empty
`## [Unreleased]`, leaving the entries where they are:

    ## [Unreleased]

    ## [0.14.0] — 2026-09-02

Match the existing headers byte for byte, em dash included. This is the step
0.14.0 skipped: the version was on the registry and tagged while the repository
said it did not exist.

### 4. Run the gates

    python .github/local_gates.py

Runs what CI runs, read out of `ci.yml` rather than from a list that can drift.

### 5. Publish, in the order from step 2

    cargo publish -p <crate> --dry-run
    cargo publish -p <crate>

### 6. Verify the REGISTRY, not cargo's output

`cargo publish` printing "Published" is its own report of its own action. Ask
crates.io, and ask it something that answers the question -- dumping the JSON
shows a `versions` array of numeric IDs, which runs fine and tells you nothing:

    for c in vulkane definitely-not-a-real-crate-xyzzy; do
      curl -s -H "User-Agent: you <your@email>" \
        "https://crates.io/api/v1/crates/$c" | python -c "
    import sys, json
    raw = sys.stdin.read()
    if not raw.strip():
        print('EMPTY BODY -- you did not send a User-Agent'); raise SystemExit(1)
    d = json.loads(raw)
    if 'errors' in d:
        print('absent'); raise SystemExit(0)
    print([v['num'] for v in d['versions'] if not v['yanked']][:3])
    "
    done

    vulkane: ['0.14.0', '0.13.0', '0.10.1']
    definitely-not-a-real-crate-xyzzy: absent

**Send a User-Agent.** Without one crates.io returns an EMPTY BODY and curl
exits 0, so a parser reports nothing found -- indistinguishable from a publish
that did not happen, and the shape that leads to publishing twice.

**The second name is not decoration.** It is the negative control: a crate that
cannot exist must come back `absent`. If it does not, the instrument is
answering rather than the registry, and the first line means nothing either.

### 7. Verify the ARTIFACT, then tag the commit it names

"`main` has the fix" and "the crate people download has the fix" are different
claims. Only the second one reaches anybody:

    curl -sL -H "User-Agent: you <your@email>" -o v.crate \
      https://static.crates.io/crates/vulkane/vulkane-<version>.crate
    tar xzf v.crate
    cat vulkane-<version>/.cargo_vcs_info.json     # -> {"git":{"sha1":"..."}}
    grep -c "<something the release changed>" vulkane-<version>/src/lib.rs

**Tag the sha1 from `.cargo_vcs_info.json`, not `HEAD`.** They usually agree;
when they do not, `HEAD` is wrong and nothing says so. Tags here are annotated:

    git tag -a v<version> <sha1-from-the-artifact> -F -
    git push origin v<version>

### 8. Tag only after the registry confirms

The tag asserts that a published version exists. Tagging first makes it a
promise instead.

## Getting Help

- Open an issue at <https://github.com/ciresnave/vulkane/issues>
- Check existing issues and discussions first — Vulkan questions recur
- Consult the API docs at <https://docs.rs/vulkane>

## License

By contributing, you agree to license your code under either:

- Apache License, Version 2.0
- MIT License

at your option.
