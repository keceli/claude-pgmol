"""The answer must not depend on how the molecule is placed or listed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import acceptance_files, expected_label
from pgmol import find_point_group, point_group_from_coordinates, read_xyz

FIXTURES = acceptance_files()
SEED = 20251009


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation, via QR of a Gaussian matrix."""
    matrix, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = matrix * np.sign(np.diag(upper))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0
    return matrix


def test_random_rotation_helper_is_a_proper_rotation() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        matrix = random_rotation(rng)
        assert matrix.T @ matrix == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_invariant_under_rotation_translation_and_relabelling(path: Path) -> None:
    molecule = read_xyz(path)
    expected = expected_label(path)
    rng = np.random.default_rng(SEED)

    for _ in range(3):
        rotation = random_rotation(rng)
        shift = rng.uniform(-20.0, 20.0, size=3)
        order = rng.permutation(molecule.natoms)
        coordinates = molecule.coordinates @ rotation.T + shift
        symbols = [molecule.symbols[index] for index in order]
        assert point_group_from_coordinates(symbols, coordinates[order]) == expected


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_invariant_under_reflection_of_the_input(path: Path) -> None:
    """A mirror image belongs to the same point group, chiral or not."""
    molecule = read_xyz(path)
    mirrored = molecule.coordinates * np.array([1.0, 1.0, -1.0])
    assert point_group_from_coordinates(molecule.symbols, mirrored) == expected_label(path)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_invariant_under_uniform_scaling(path: Path) -> None:
    """Scaling up relaxes the angular slack, so the label must not change.

    The tolerance is scaled with the molecule so the comparison is fair.
    """
    molecule = read_xyz(path)
    scaled = molecule.coordinates * 2.5
    assert point_group_from_coordinates(
        molecule.symbols, scaled, tolerance=0.25
    ) == expected_label(path)


@pytest.mark.parametrize("path", FIXTURES[:8], ids=lambda p: p.stem)
def test_reversing_the_atom_order_changes_nothing(path: Path) -> None:
    molecule = read_xyz(path)
    assert point_group_from_coordinates(
        molecule.symbols[::-1], molecule.coordinates[::-1]
    ) == expected_label(path)


def test_file_and_memory_entry_points_agree_after_a_rigid_motion(
    tmp_path: Path, water_xyz: Path
) -> None:
    molecule = read_xyz(water_xyz)
    rng = np.random.default_rng(SEED)
    moved = molecule.coordinates @ random_rotation(rng).T + np.array([3.0, -1.0, 2.0])
    relocated = tmp_path / "moved.xyz"
    relocated.write_text(
        type(molecule)(molecule.symbols, moved, "moved").to_xyz(), encoding="utf-8"
    )
    assert find_point_group(relocated) == "C2v"
