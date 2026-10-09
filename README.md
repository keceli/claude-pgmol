# pgmol

Determine the Schoenflies point group of a molecule from an XYZ geometry.

`pgmol` reads ordinary XYZ coordinates, finds every symmetry operation that
maps the molecule onto itself within a distance tolerance, closes that set
under composition, and names the resulting group.

```console
$ pgmol examples/water.xyz
C2v
```

## Installation

Requires Python 3.10+ and NumPy.

```console
$ python -m venv .venv && source .venv/bin/activate
$ pip install .
```

For development, including the test and lint tooling:

```console
$ pip install -e ".[dev]"
```

## Python API

```python
>>> from pgmol import find_point_group, point_group_from_coordinates
>>> find_point_group("examples/water.xyz")
'C2v'
>>> point_group_from_coordinates(["H", "H"], [[0, 0, 0.37], [0, 0, -0.37]])
'Dinfh'
```

`analyze` returns the full picture rather than just the label:

```python
>>> from pgmol import analyze, read_xyz
>>> result = analyze(read_xyz("core_xyz/CH4_Td.xyz"))
>>> result.label, result.order
('Td', 24)
>>> result.operation_counts()
{'C2': 3, 'C3': 8, 'E': 1, 'S4': 6, 'sigma': 6}
>>> result.consistent
True
```

The public API is:

| Name | Purpose |
| --- | --- |
| `find_point_group(path, tolerance=0.1)` | Label for an XYZ file |
| `point_group_from_coordinates(symbols, coordinates, tolerance=0.1)` | Label for in-memory coordinates |
| `analyze(molecule, tolerance=0.1)` | Full `SymmetryAnalysis` record |
| `read_xyz(path)` / `parse_xyz(text)` | Parse XYZ into a `Molecule` |
| `molecule_from_coordinates(symbols, coordinates)` | Build a `Molecule` directly |
| `Molecule` | Immutable symbols + coordinates |
| `PGMolError`, `XYZFormatError`, `SymmetryError` | Exceptions |

Coordinates passed in are copied, never modified. Importing the package has
no side effects beyond importing NumPy.

## Command line

```console
$ pgmol core_xyz/C6H6_D6h.xyz
D6h
$ pgmol --tolerance 0.05 core_xyz/C2H6_D3d.xyz
D3d
$ pgmol --verbose examples/water.xyz
C2v
atoms: 3
tolerance: 0.1 A
order: 4
operations: 1xC2, 1xE, 2xsigma
```

The label goes to stdout; diagnostics and errors go to stderr. Exit status is
`0` on success, `1` for unreadable, malformed or unanalysable input, and `2`
for command-line misuse (argparse's convention).

## Tolerance semantics

`tolerance` is a **distance in ångströms**: the largest displacement allowed
between an atom's symmetry-transformed position and the position of the atom
it is matched with. An operation is accepted when, element by element, every
transformed atom lands within `tolerance` of a *distinct* atom of the same
element — that is, when the operation induces a permutation of each element's
atoms. The default is `0.1` Å.

The rule is monotone and deterministic: a larger tolerance can only add
operations, so it can promote a slightly distorted geometry to a
higher-symmetry label, and the same input always produces the same answer.
There is no rounding or snapping of coordinates anywhere.

Three things make the result well defined rather than tolerance-dependent
guesswork:

* **Ambiguity cap.** Once twice the tolerance reaches the shortest distance
  between two atoms of the same element, a transformed atom could be within
  tolerance of two different partners and the correspondence stops being
  defined. The tolerance used for matching is therefore capped just below
  half that distance. `SymmetryAnalysis.tolerance` reports what you asked
  for and `SymmetryAnalysis.effective_tolerance` what was used; for ordinary
  molecules at the default tolerance the cap never binds.
* **Canonical operations.** A candidate matrix is replaced by the
  least-squares orthogonal matrix for the atom permutation it induces, so
  every candidate that implies the same permutation collapses to the same
  operation. Operations are then identified exactly by `(permutation,
  determinant sign)`, with no matrix comparison tolerance involved.
* **Finite order.** Accidental near-coincidences are rejected because a
  genuine point-group operation has finite order, and a `Cn` axis needs an
  orbit of `n` atoms of one element — so `n` cannot exceed the largest
  element count.

`SymmetryAnalysis.consistent` is `False` when the operations found do not
close under composition or their count does not match the order the label
implies. That means the tolerance is too loose for the geometry: tighten it.
The command line prints a warning in that case.

Collinearity is tested against the tolerance as a plain geometric threshold
(no matching is involved), so a molecule bent by less than the tolerance is
reported as linear.

## Supported labels

* `Kh` — a single atom.
* `Cinfv`, `Dinfh` — linear molecules.
* `C1`, `Cs`, `Ci`.
* Axial families for any `n`: `Cn`, `Cnv`, `Cnh`, `Dn`, `Dnh`, `Dnd`, and
  `S2n` (even-index improper-rotation groups).
* Polyhedral: `T`, `Th`, `Td`, `O`, `Oh`, `I`, `Ih`.

Infinite groups use the ASCII spellings `Cinfv` and `Dinfh`.

## XYZ format

One frame per file:

```
<natoms>
<comment>
<symbol> <x> <y> <z>
...
```

Supported conventions:

* Blank lines before the count line and after the atom block are ignored.
* The comment line may be empty.
* Element symbols are case-insensitive (`cl`, `CL`, `Cl`), and a bare atomic
  number (`17`) is accepted; both are canonicalised to the IUPAC symbol.
* Coordinates are in ångströms, in any float syntax Python accepts.

Rejected with a message naming the file and line, rather than silently
reinterpreted: a wrong atom count, a truncated or blank-interrupted atom
block, anything other than exactly four fields per atom line (so extended-XYZ
columns are an error, not data), unknown element symbols, non-finite
coordinates, coincident atoms, and trailing content such as a second
trajectory frame.

## Limitations

* Single-frame XYZ only; trajectories, extra per-atom columns and
  extended-XYZ comment metadata are rejected rather than ignored.
* Isotopes and charges are not modelled — atoms are equivalent when their
  element symbols match.
* The geometry is treated as rigid. Non-rigid (permutation-inversion)
  symmetry, such as a freely rotating methyl group, is not considered.
* The reported group is the symmetry of the coordinates as given, at the
  tolerance given. Deciding whether a near-symmetric geometry "is" the
  idealised structure is the caller's choice of tolerance, not a property of
  the file.

## Performance

Cost is dominated by the number of candidate axes, which grows as O(N²) in
the atom count, times the cost of testing one candidate, which is O(N²) in
the worst case for matching atoms. A cheap probe over a few atoms rejects
most candidates before the full match, and group closure is pure index
arithmetic on permutations rather than matrix algebra.

Measured on one core (Python 3.10, NumPy 2.2):

| Molecule | Atoms | Group | Time |
| --- | --- | --- | --- |
| H2O | 3 | `C2v` | ~3 ms |
| C6H6 | 12 | `D6h` | ~11 ms |
| B12H12 | 24 | `Ih` | ~85 ms |
| C60 | 60 | `Ih` | ~1.9 s |

All 36 acceptance fixtures together take about 0.3 s. Molecules of a few
hundred equivalent atoms are not a design target.

## Tests

```console
$ pip install -e ".[dev]"
$ pytest                                   # full suite
$ pytest --cov=pgmol --cov-report=term-missing
$ ruff check src tests && ruff format --check src tests
$ mypy --strict src/pgmol
```

The package ships `py.typed`, and the annotations are checked under
`mypy --strict` in CI.

The suite covers:

* every file in `core_xyz/`, with the expected label derived from the file
  name **in the tests**, never in the package;
* all 51 reference point groups, built in `tests/reference_groups.py` by
  applying textbook generator matrices to seeds in general position — so the
  expected answers come from group theory, not from this implementation;
* invariance under rotation, translation, reflection, uniform scaling and
  atom reordering;
* linear molecules, single atoms, noise and tolerance boundaries, malformed
  input, and CLI success and error paths.

An independent check runs on every fixture and every generated group: the
number of operations found must equal the order the Schoenflies label
implies, and the set must be closed under composition.

## How it works

1. Translate to the unweighted centroid of the atoms. Every symmetry
   operation permutes atoms of the same element, so it fixes the centroid —
   which is why no atomic-mass table is needed.
2. Handle a single atom (`Kh`) and collinear atoms (`Cinfv`/`Dinfh`)
   separately.
3. Enumerate candidate axes from the geometry: atom directions, and for each
   same-element pair the bisector, the difference and the common normal, plus
   the principal axes of the gyration tensor. Axes that pass through no atom
   — the `C3` axes of a chiral tetrahedral group, the `C5` axes of a chiral
   icosahedral one — are reconstructed from the displacements a rotation must
   produce: `u` is perpendicular to `a - Ra` for every atom `a`.
4. For each axis, read the candidate rotation angles straight off the
   geometry instead of guessing orders: a symmetry operation must carry a
   chosen reference atom onto an atom of the same element at the same height
   along the axis and the same distance from it, and each such atom fixes one
   angle exactly.
5. Test each candidate, canonicalise the survivors, and close the set under
   composition.
6. Name the group with the standard flow chart: several high-order axes mean
   a polyhedral group; otherwise the principal axis fixes `n`, and `n`
   perpendicular `C2` axes, a horizontal mirror, `n` vertical mirrors and
   `S2n` select among the `D` and `C` families.

## License

MIT — see `LICENSE`.
