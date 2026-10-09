"""Tolerance semantics: noise, boundaries and the ambiguity cap."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import acceptance_files, expected_label
from pgmol import analyze, molecule_from_coordinates, point_group_from_coordinates, read_xyz


def acceptance_files_by_stem() -> dict:
    return {path.stem: path for path in acceptance_files()}


SEED = 424242
WATER = (
    ["O", "H", "H"],
    np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]]),
)


@pytest.mark.parametrize("path", acceptance_files(), ids=lambda p: p.stem)
def test_noise_well_below_the_tolerance_preserves_the_label(path) -> None:
    molecule = read_xyz(path)
    rng = np.random.default_rng(SEED)
    jittered = molecule.coordinates + rng.normal(scale=0.004, size=molecule.coordinates.shape)
    assert point_group_from_coordinates(molecule.symbols, jittered) == expected_label(path)


def test_a_displacement_larger_than_the_tolerance_lowers_the_symmetry() -> None:
    """Stretching one O-H breaks C2 but keeps the molecular plane: C2v -> Cs."""
    symbols, base = WATER
    distorted = base.copy()
    distorted[1] = distorted[1] * 1.4
    assert point_group_from_coordinates(symbols, distorted, tolerance=0.1) == "Cs"


def test_the_tolerance_decides_at_the_boundary() -> None:
    """One documented knob, one monotone effect, no rounding anywhere.

    The same geometry is C2v for a tolerance comfortably above the
    distortion and Cs for one comfortably below it.
    """
    symbols, base = WATER
    displacement = 0.05
    distorted = base.copy()
    distorted[1, 0] += displacement

    assert point_group_from_coordinates(symbols, distorted, tolerance=0.2) == "C2v"
    assert point_group_from_coordinates(symbols, distorted, tolerance=0.005) == "Cs"


def test_tightening_the_tolerance_never_raises_the_symmetry() -> None:
    symbols, base = WATER
    rng = np.random.default_rng(SEED)
    distorted = base + rng.normal(scale=0.05, size=base.shape)
    orders = [
        analyze(molecule_from_coordinates(symbols, distorted), tolerance=tol).order
        for tol in (0.3, 0.1, 0.03, 0.01, 0.003)
    ]
    assert orders == sorted(orders, reverse=True)


def test_effective_tolerance_is_capped_when_matching_would_be_ambiguous() -> None:
    """Two same-element atoms 0.12 A apart make a 0.5 A tolerance meaningless."""
    symbols = ["H", "H", "H", "O"]
    coordinates = np.array(
        [[1.0, 0.0, 0.0], [1.06, 0.104, 0.0], [-1.0, 0.0, 0.4], [0.0, 0.0, -0.3]]
    )
    result = analyze(molecule_from_coordinates(symbols, coordinates), tolerance=0.5)
    shortest = np.linalg.norm(coordinates[0] - coordinates[1])
    assert result.effective_tolerance < 0.5
    assert result.effective_tolerance < 0.5 * shortest
    assert result.tolerance == 0.5


def test_effective_tolerance_is_untouched_for_ordinary_molecules() -> None:
    symbols, base = WATER
    result = analyze(molecule_from_coordinates(symbols, base), tolerance=0.1)
    assert result.effective_tolerance == 0.1


def test_no_cap_when_no_element_repeats() -> None:
    result = analyze(
        molecule_from_coordinates(["C", "N", "O"], [[0, 0, 1.2], [0, 0, 0], [0.9, 0, -0.8]]),
        tolerance=0.4,
    )
    assert result.effective_tolerance == 0.4


def test_a_merely_approximate_geometry_is_flagged() -> None:
    """A tolerance too loose for the geometry yields a non-group, and says so."""
    rng = np.random.default_rng(1234)
    shell = rng.normal(size=(40, 3))
    shell = 1.2 * shell / np.linalg.norm(shell, axis=1)[:, None]
    result = analyze(molecule_from_coordinates(["C"] * 40, shell), tolerance=0.35)
    assert not result.consistent or result.label == "C1"


def test_a_loose_tolerance_can_promote_a_distorted_geometry() -> None:
    """Documented behaviour: more slack means more symmetry, deterministically.

    Methane jittered by 0.02 A is strictly ``C1``, but a tolerance that
    exceeds the distortion restores the idealised ``Td``.
    """
    molecule = read_xyz(acceptance_files_by_stem()["CH4_Td"])
    rng = np.random.default_rng(5)
    jittered = molecule.coordinates + rng.normal(scale=0.02, size=molecule.coordinates.shape)
    assert point_group_from_coordinates(molecule.symbols, jittered, tolerance=0.003) == "C1"
    assert point_group_from_coordinates(molecule.symbols, jittered, tolerance=0.1) == "Td"


def test_an_out_of_plane_nudge_barely_changes_bond_lengths() -> None:
    """Why the tolerance is a distance: geometry, not coordinate differences.

    Lifting one hydrogen of water 0.08 A out of the molecular plane changes
    the O-H length by only 0.003 A, so the molecule stays ``C2v`` for any
    tolerance above that -- the operation matching is what the tolerance
    applies to, not the raw coordinates.
    """
    symbols, base = WATER
    distorted = base.copy()
    distorted[1, 1] += 0.08
    lengths = [float(np.linalg.norm(distorted[i] - distorted[0])) for i in (1, 2)]
    assert abs(lengths[0] - lengths[1]) == pytest.approx(0.00333, abs=1e-4)
    assert point_group_from_coordinates(symbols, distorted, tolerance=0.005) == "C2v"
    assert point_group_from_coordinates(symbols, distorted, tolerance=0.001) == "Cs"


def test_the_same_input_always_gives_the_same_answer() -> None:
    symbols, base = WATER
    rng = np.random.default_rng(SEED)
    distorted = base + rng.normal(scale=0.03, size=base.shape)
    answers = {
        point_group_from_coordinates(symbols, distorted, tolerance=0.07) for _ in range(5)
    }
    assert len(answers) == 1
