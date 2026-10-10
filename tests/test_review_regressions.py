"""Regressions for defects found reviewing the branch.

One test per finding, each reproducing the original failure.
"""

from __future__ import annotations

import math
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from conftest import acceptance_files
from pgmol import (
    Molecule,
    XYZFormatError,
    analyze,
    molecule_from_coordinates,
    parse_xyz,
    point_group_from_coordinates,
    read_xyz,
)
from pgmol.geometry import (
    StructureMatcher,
    element_blocks,
    improper_rotation_matrix,
    minimum_equivalent_separation,
)
from pgmol.symmetry import identify_operation

Z = [0.0, 0.0, 1.0]


# --- the answer must not depend on the order of the XYZ lines -------------

# Two oxygens and two hydrogens, slightly distorted, so that both elements
# have the same number of off-axis atoms and the reference-element choice
# used to fall through to whichever appeared first in the file.
_TIE = {
    "O": ([0.830237, -0.031712, 0.019867], [-0.730948, 0.035383, -0.000964]),
    "H": ([0.918750, 0.722161, 0.523879], [-0.952326, -0.706672, 0.574055]),
}


@pytest.mark.parametrize(
    "order",
    [
        ("O", "O", "H", "H"),
        ("H", "H", "O", "O"),
        ("O", "H", "O", "H"),
        ("H", "O", "H", "O"),
    ],
)
def test_tied_element_counts_do_not_make_the_result_order_dependent(order) -> None:
    taken = {"O": 0, "H": 0}
    symbols, coordinates = [], []
    for element in order:
        coordinates.append(_TIE[element][taken[element]])
        taken[element] += 1
        symbols.append(element)
    assert point_group_from_coordinates(symbols, coordinates, tolerance=0.08) == "C2"


# --- Schoenflies index, not matrix order, names an improper rotation ------


@pytest.mark.parametrize(("index", "expected"), [(3, "S3"), (4, "S4"), (5, "S5"), (6, "S6")])
def test_improper_rotation_is_named_by_its_axis_order(index: int, expected: str) -> None:
    """S_n for odd n has matrix order 2n, so the two must not be conflated."""
    operation = identify_operation(improper_rotation_matrix(Z, 2 * math.pi / index))
    assert operation.label == expected
    assert operation.order == (index if index % 2 == 0 else 2 * index)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("BF3_D3h", {"S3": 2}),
        ("Cl5P_D3h", {"S3": 2}),
        ("C10FeH10_D5h", {"S5": 4}),
        # D6h holds two S3 and two S6; they share matrix order 6 but are
        # genuinely different operations.
        ("C6H6_D6h", {"S3": 2, "S6": 2}),
        ("CH4_Td", {"S4": 6}),
        ("C2H6_D3d", {"S6": 2}),
    ],
)
def test_improper_counts_match_the_character_tables(stem: str, expected: dict) -> None:
    counts = analyze(read_xyz(Path("core_xyz") / f"{stem}.xyz")).operation_counts()
    assert {k: v for k, v in counts.items() if k.startswith("S")} == expected


# --- the linear short-circuit must agree with the rest of the pipeline ----


def test_a_bent_molecule_is_not_called_linear_at_the_default_tolerance() -> None:
    """The two H are 0.1 A apart, so matching is capped well below the bend."""
    result = analyze(
        molecule_from_coordinates(
            ["O", "H", "H"], [[0, 0, 0.6], [0.05, 0, -0.3], [-0.05, 0, -0.3]]
        )
    )
    assert result.label == "C2v"
    assert result.effective_tolerance < 0.05


@pytest.mark.parametrize(
    ("stem", "tolerance", "expected"),
    [("CH4_Td", 1.0, "Td"), ("C6H6_D6h", 5.0, "D6h")],
)
def test_a_large_tolerance_does_not_collapse_a_molecule_to_a_line(
    stem: str, tolerance: float, expected: str
) -> None:
    molecule = read_xyz(Path("core_xyz") / f"{stem}.xyz")
    assert analyze(molecule, tolerance=tolerance).label == expected


# --- a point group always contains the identity ---------------------------


@pytest.mark.parametrize("tolerance", [1e-6, 1e-8, 1e-10, 1e-14])
def test_a_tiny_tolerance_never_yields_an_empty_group(tolerance: float) -> None:
    """Squared-distance cancellation once rejected even the identity."""
    methane = molecule_from_coordinates(
        ["C", "H", "H", "H", "H"],
        [
            [0, 0, 0],
            [0.629, 0.629, 0.629],
            [0.629, -0.629, -0.629],
            [-0.629, 0.629, -0.629],
            [-0.629, -0.629, 0.629],
        ],
    )
    result = analyze(methane, tolerance=tolerance)
    assert result.order == 24
    assert result.label == "Td"


@pytest.mark.parametrize("path", acceptance_files(), ids=lambda p: p.stem)
def test_every_fixture_keeps_its_label_at_a_very_tight_tolerance(path: Path) -> None:
    assert analyze(read_xyz(path), tolerance=1e-9).order != 0


# --- value semantics on every public type ---------------------------------


def test_symmetry_analysis_compares_and_hashes(water_xyz: Path) -> None:
    first = analyze(read_xyz(water_xyz))
    second = analyze(read_xyz(water_xyz))
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert first != analyze(read_xyz(water_xyz), tolerance=0.2)
    assert first.__eq__(17) is NotImplemented


def test_symmetry_operations_compare_and_hash(water_xyz: Path) -> None:
    operations = analyze(read_xyz(water_xyz)).operations
    assert operations[0] == operations[0]
    assert operations[0] != operations[1]
    assert len(set(operations)) == len(operations)
    assert operations[0].__eq__("E") is NotImplemented


def test_structure_matcher_is_comparable_by_identity() -> None:
    coordinates = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    matcher = StructureMatcher(coordinates, element_blocks(["H", "H"]), 0.1)
    assert matcher == matcher
    assert matcher != StructureMatcher(coordinates, element_blocks(["H", "H"]), 0.1)


def test_reflecting_a_planar_molecule_preserves_both_equality_and_hash(
    water_xyz: Path,
) -> None:
    """Negative zero must not split a molecule from its own mirror image."""
    molecule = read_xyz(water_xyz)
    flipped = Molecule(molecule.symbols, molecule.coordinates * np.array([1.0, -1.0, 1.0]))
    assert flipped == molecule
    assert hash(flipped) == hash(molecule)
    assert len({molecule, flipped}) == 1
    assert {molecule: "C2v"}[flipped] == "C2v"


# --- every construction path validates ------------------------------------


@pytest.mark.parametrize(
    ("coordinates", "fragment"),
    [
        ([[0, 0, 0], [1, 2]], "must be numeric"),
        ([["a", "b", "c"], [1, 2, 3]], "must be numeric"),
        ([[0, 0, 0], [1j, 2, 3]], "must be numeric"),
    ],
)
def test_unusable_coordinate_input_raises_the_documented_error(
    coordinates, fragment: str
) -> None:
    with pytest.raises(XYZFormatError, match=fragment):
        point_group_from_coordinates(["H", "H"], coordinates)


def test_molecule_rejects_coincident_atoms_too() -> None:
    """Molecule is public, and used to be the one path that skipped the check."""
    with pytest.raises(ValueError, match="coincident"):
        Molecule(("H", "H", "O"), [[0, 0, 0], [0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("magnitude", [1e307, 1e200, 1e150])
def test_absurdly_large_coordinates_are_rejected_cleanly(magnitude: float) -> None:
    """These are finite, so they used to reach the grid and LAPACK and crash."""
    with pytest.raises(XYZFormatError, match="magnitude"):
        point_group_from_coordinates(["H", "H"], [[magnitude, 0, 0], [0, 0, 1.0]])
    with pytest.raises(XYZFormatError, match="magnitude"):
        parse_xyz(f"2\nbig\nH {magnitude:g} 0.0 0.0\nH 0.0 0.0 1.0\n")


def test_large_but_usable_coordinates_still_work() -> None:
    far = 1e8
    coordinates = np.array([[0, 0, 0], [0.757, 0, 0.587], [-0.757, 0, 0.587]]) + far
    assert point_group_from_coordinates(["O", "H", "H"], coordinates) == "C2v"


@pytest.mark.parametrize("token", ["²", "³", "٣", "②"])
def test_non_ascii_digits_are_not_element_symbols(token: str) -> None:
    """str.isdigit() accepts these; int() rejects some and misreads others."""
    with pytest.raises(XYZFormatError, match="unknown element symbol"):
        parse_xyz(f"1\nc\n{token} 0 0 0\n")


def test_structure_matcher_requires_blocks_to_cover_every_atom() -> None:
    """A partial block map used to return uninitialised memory as indices."""
    coordinates = np.array([[0, 0, 1.0], [0, 0, -1.0], [5, 5, 5.0], [1, 2, 3.0]])
    with pytest.raises(ValueError, match="every atom"):
        StructureMatcher(coordinates, {"H": np.array([0, 1])}, 0.1)


# --- the ambiguity cap must stay linear -----------------------------------


def test_minimum_separation_is_bounded_and_linear() -> None:
    rng = np.random.default_rng(0)
    coordinates = rng.normal(size=(8000, 3)) * 50.0
    blocks = element_blocks(["C"] * 8000)
    tracemalloc.start()
    try:
        value = minimum_equivalent_separation(coordinates, blocks, limit=0.2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert math.isfinite(value)
    # The all-pairs form would need 24 * 8000**2 bytes, about 1.5 GB.
    assert peak < 30e6


def test_minimum_separation_is_exact_for_small_blocks() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 3.0, 0.0]])
    blocks = element_blocks(["H", "H", "O"])
    assert minimum_equivalent_separation(coordinates, blocks) == pytest.approx(1.5)
    assert minimum_equivalent_separation(coordinates, blocks, limit=0.1) == pytest.approx(1.5)


def test_analysis_of_a_large_molecule_does_not_exhaust_memory() -> None:
    """The cap lookup used to allocate an natoms-by-natoms-by-3 tensor."""
    rng = np.random.default_rng(3)
    coordinates = rng.normal(size=(4000, 3)) * 60.0
    tracemalloc.start()
    try:
        molecule = molecule_from_coordinates(["C"] * 4000, coordinates)
        from pgmol.api import _effective_tolerance

        _effective_tolerance(molecule.coordinates, element_blocks(molecule.symbols), 0.1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 30e6


def test_minimum_separation_grid_picks_the_closest_of_several_neighbours() -> None:
    """Three atoms in one cell: the grid must keep the smallest distance."""
    rng = np.random.default_rng(11)
    coordinates = rng.normal(size=(300, 3)) * 80.0
    coordinates[1] = coordinates[0] + np.array([0.09, 0.0, 0.0])
    coordinates[2] = coordinates[0] + np.array([0.0, 0.04, 0.0])
    coordinates[3] = coordinates[0] + np.array([0.0, 0.0, 0.14])
    blocks = element_blocks(["C"] * 300)
    assert minimum_equivalent_separation(coordinates, blocks, limit=0.2) == pytest.approx(0.04)
