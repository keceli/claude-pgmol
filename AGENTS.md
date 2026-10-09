# Instructions for a molecular point-group Python package

## Objective

Create a **new, installable Python package** that accepts a molecular XYZ file
and determines its Schoenflies point group. Develop against the public
acceptance fixtures in `core_xyz/`. Each filename follows
`<formula>_<pointgroup>.xyz`; for example, `H2O_C2v.xyz` expects `C2v`.
Coordinates are in ångströms. Infinite groups use the ASCII labels `Cinfv`
and `Dinfh`, and a single atom uses `Kh`.

**Success criterion:** correctly identify the point group of **every** XYZ
file in `core_xyz/` through the package's public API, using only its XYZ
contents at inference time. The folder includes multiple examples for some
groups. Do not hard-code filenames, formulas, atom counts, or fixture-specific
coordinates. Generalize to different valid XYZ files and atom orderings.

## Package contract

- Provide an importable public function, for example
  `find_point_group(path: str | Path, tolerance: float = 0.1) -> str`, that
  reads an XYZ file and returns a canonical Schoenflies label. Make the
  distance tolerance's meaning and units explicit in the docstring. Also
  provide a coordinate-based entry point so callers need not write a file.
- Provide a command-line interface that accepts an XYZ path and prints the
  point-group label; exit nonzero with a useful error for invalid input.
- Read ordinary XYZ files correctly: atom count, comment line, element
  symbols, and three finite Cartesian coordinates per atom. Reject truncated
  or inconsistent input rather than silently classifying a different
  molecule. Document which optional XYZ conventions are supported.
- Apply the tolerance consistently to geometric matching. Account for
  equivalent atoms only when their chemical identities agree. Handle
  linear molecules and a single atom separately from finite nonlinear
  groups. Detect proper rotations, reflections, inversion, and improper
  rotations as needed to distinguish the labels in `core_xyz/`.
- Results should not depend on rigid translation, arbitrary rotation,
  reflection of the input, or the order of XYZ atom lines. For near-symmetric
  geometries, prefer a documented tolerance rule and deterministic behavior
  over ad hoc rounding.

## Engineering standards

- Use `pyproject.toml` with standard build metadata and a modern `src/`
  package layout. Declare supported Python versions and runtime dependencies
  accurately. Keep installation and import free of side effects.
- Separate XYZ parsing, geometry/symmetry logic, and the CLI. Keep the public
  API small, typed, and documented. Prefer clear, testable functions over
  monolithic routines; do not hide broad exceptions or mutate caller-owned
  coordinates unexpectedly.
- Add a README with installation instructions, short Python and CLI examples,
  tolerance semantics, supported labels, expected XYZ format, limitations,
  and a guide to running the tests. Include concise API docstrings and an
  example XYZ file.
- Write automated tests for every public behavior: each public XYZ fixture,
  rigid motions, atom permutations, linear and one-atom cases, distinct
  species, noise/tolerance boundaries, malformed input, and CLI success/error
  paths. Use deterministic random seeds. Derive fixture expectations from
  filenames in the tests, **not in the implementation**. Aim for at least
  90% meaningful line coverage of the new package, with branch coverage
  where practical; a coverage number alone is not a substitute for edge
  cases and independent expected results.
- Configure CI (such as GitHub Actions) to install the package from a clean
  environment, run the test suite with coverage, and run lint/format checks
  on multiple supported Python versions. Keep dependency versions and
  development commands reproducible. A failing test or quality check should
  fail CI; do not mask missing cases with blanket skips or expected failures.
- Profile slow cases before adding special-case optimizations. Document
  complexity or performance limits. Avoid modifying acceptance XYZ files
  merely to make tests pass.

## Git workflow

- Check `git status` before editing. If the package does not yet have a Git
  repository, ask before initializing one or creating a nested repository.
  Keep `.venv/`, caches, build artifacts, generated coverage reports, and
  credentials out of version control with `.gitignore`. Do not commit secrets.
- Work on a topic branch when the project uses branches. Make small, coherent
  commits with descriptive messages that explain the change; keep unrelated
  changes separate. Review `git diff` and staged files before each commit.
  Do not commit failing tests as if the work were complete.
- Do not discard, reset, amend, or overwrite someone else's work. Avoid
  force-pushes and history rewrites of shared branches. Reconcile conflicts
  explicitly, rerun the relevant tests, and request approval for destructive
  operations. Do not push or open a pull request without authorization.
- For a proposed merge, provide a concise change summary, test and coverage
  results, known limitations, and a link to the relevant commits or diff.
  Require passing CI and resolved review findings before merging.

## Independent code review

- Before declaring completion, request **at least two independent, read-only
  reviews** of the same proposed diff, using different model families or
  providers when available. Look up the actual available models rather than
  inventing model IDs. Give reviewers the requirements and diff separately;
  do not show either reviewer the other's findings until both have submitted
  their reports. If distinct models or independent agents are unavailable,
  state the limitation and seek human review instead of claiming this check
  passed.
- Ask one reviewer to focus on scientific correctness, symmetry edge cases,
  tolerance behavior, and whether the implementation generalizes beyond the
  fixtures. Ask the other to focus on Python packaging, XYZ validation,
  security, maintainability, tests, coverage, documentation, and CI. Both
  should attempt to find counterexamples, not merely approve the code.
- Reviewers must not edit files or run destructive commands. They should
  report actionable findings with severity, affected file and line, a short
  explanation, and a reproducible example or test when possible. The author
  triages every finding, fixes or explains it, reruns tests and CI, and asks
  for a follow-up review of substantive fixes. Keep review summaries and
  unresolved issues visible in the merge request or final handoff.
- If reviewers need separate workspaces, use isolated read-only checkouts or
  worktrees of the same commit. Never let parallel agents write to the same
  working tree or merge their changes without an explicit reconciliation.

## Acceptance check

Run the package's own full suite and a separate parametrized test over
`core_xyz/*.xyz` that calls the **public API** for each file and compares its
return value with the final `_pointgroup` segment of the filename. Report
the number of passing fixtures, test coverage, supported Python versions,
and any remaining limitations. Completion requires every public fixture to
pass without bypassing the package's parser or detection logic.
