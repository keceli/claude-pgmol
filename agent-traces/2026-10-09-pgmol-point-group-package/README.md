# pgmol — building a molecular point-group package

An interactive replay of the AI coding session that produced this
repository, from an empty directory to a reviewed, tested package with a
deployed web interface.

**[▶ Watch the replay](replay.html)**

| | |
| --- | --- |
| Date | 2026-10-09 → 2026-10-12 (UTC) |
| Model | `claude-opus-5` (Claude Code) |
| Turns | 25 user prompts, 814 assistant messages |
| Tokens | 22,500 input · 1,177,882 output · 324,504,877 cache read |
| Wall clock | ~66 h elapsed across the session |

## What the session covers

1. **The package.** XYZ parsing, symmetry detection and Schoenflies
   classification, a typed public API, a CLI, docs and CI — built against
   the 36 acceptance fixtures in `core_xyz/`.
2. **Generalisation testing.** A property sweep over 51 point groups built
   from textbook generators independently of the implementation, which found
   three defects the fixtures could not: chiral polyhedral axes that were
   never proposed, near-identity rotations accepted as `C112`, and
   near-duplicate matrices inflating group orders past 120.
3. **Three rounds of review.** Roughly 38 defects found and fixed, each
   reproduced before the fix and covered by a regression test afterwards —
   including results that depended on XYZ line order, improper rotations
   named by matrix order rather than axis order, groups reported with zero
   operations, and two `O(N²)` memory blowups.
4. **A browser interface.** A static page running the real package under
   Pyodide rather than a JavaScript reimplementation, deployed to GitHub
   Pages.
5. **A 3D viewer** that draws the detected rotation axes, mirror planes and
   inversion centre over the structure.

The replay preserves the failed attempts and debugging as well as the
results — the blank WebGL screenshots that turned out to be a headless
compositing limitation, the `pkill` patterns that matched their own command
line, and the test premises that were wrong rather than the code.

## Commits

Merges: `30dd93f`, `521e6d4`, `72703f4`.

<details><summary>18 individual commits</summary>

```
eab23b8 Add a 3D viewer that draws the detected symmetry elements
4ab8f3d Add a browser interface, deployed to GitHub Pages
1c9e158 Address the three findings from the pull-request review
7fe7752 Correct what the axis reconstruction is actually needed for
87661b4 Add the challenge acceptance fixtures
957e3fd Fix the secondary issues the review flagged but could not detail
500e893 Fix fifteen defects found by independent review of the branch
1d00b02 Move CI off actions pinned to the deprecated Node 20 runtime
44742c8 Use NumPy's free reduction functions so type checking is version-stable
c12e6b4 Fix four defects found reviewing the branch
b19c4f4 Correct the measured timings in the README
df3da1a Type the array API precisely and check it under mypy --strict
bcc816a Add README and CI
29553ae Add test suite with independent reference point groups
9fd39f0 Add public API and command-line interface
2add37c Add symmetry detection and point-group classification
a3353f5 Add XYZ reading with strict validation
6a0b417 Add packaging metadata, licence and ignore rules
```
</details>

## Provenance and limitations

- `session.jsonl` is the sanitised transcript; `replay.html` is generated
  from it and is self-contained (no external requests).
- **Sanitisation**: home directory paths rewritten to `~`, one personal email
  address redacted, and one unrelated user's path — which reached the
  transcript when a scratch filename collided in a shared `/tmp` — replaced
  with a placeholder. No credentials, tokens or keys were present; the
  transcript was scanned for them explicitly.
- **Snapshot, not the whole session.** The trace was captured while the
  session was still running, so it ends just before the archiving step
  itself. The source transcript had grown by ten records by the time the
  copy was verified.
- **Subagent traces are not included.** A `/code-review` invocation fanned
  out into ten parallel reviewers, whose eleven transcripts total 5.9 MB.
  Their findings appear in the main trace, but their internal reasoning does
  not. They are available in the Claude Code project directory if wanted.
