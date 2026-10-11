"""Generalisation beyond the fixtures: all 51 reference point groups.

Molecules are built by applying textbook generator matrices (see
``reference_groups``) to seeds in general position, so the expected answer
comes from group theory rather than from the package.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference_groups as reference
from pgmol import analyze, molecule_from_coordinates
from pgmol.classify import expected_order


def test_polyhedral_seeds_are_in_general_position() -> None:
    reference.check_general_position()


@pytest.mark.parametrize("label", reference.ALL_LABELS)
def test_reference_group_has_the_textbook_order(label: str) -> None:
    """Guard the test's own generators before trusting them."""
    assert len(reference.operations(label)) == expected_order(label)


def _tolerance_for(symbols, coordinates) -> float:
    """A tolerance that is physically sensible for the generated shell.

    These synthetic molecules pack many identical atoms onto one sphere, far
    more tightly than a real molecule, so the default 0.1 A would genuinely
    be ambiguous for the largest of them.
    """
    shortest = reference.minimum_equivalent_distance(symbols, coordinates)
    return min(0.1, 0.1 * shortest) if np.isfinite(shortest) else 0.1


@pytest.mark.parametrize("label", reference.ALL_LABELS)
def test_generated_molecule_is_classified_exactly(label: str) -> None:
    symbols, coordinates = reference.build_molecule(label)
    result = analyze(
        molecule_from_coordinates(symbols, coordinates),
        tolerance=_tolerance_for(symbols, coordinates),
    )
    assert result.label == label
    assert result.order == expected_order(label)
    assert result.consistent


@pytest.mark.parametrize("label", ["C3", "S4", "D2d", "D5h", "D4d", "Td", "Th", "O"])
def test_generated_molecule_survives_a_rigid_motion(label: str) -> None:
    """``Ih`` is covered by the B12H12 fixture in ``test_invariance``."""
    symbols, coordinates = reference.build_molecule(label)
    rng = np.random.default_rng(7)
    basis, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    basis = basis * np.sign(np.diag(upper))
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1.0
    order = rng.permutation(len(symbols))
    moved = coordinates @ basis.T + np.array([2.0, -3.0, 0.5])
    result = analyze(
        molecule_from_coordinates([symbols[i] for i in order], moved[order]),
        tolerance=_tolerance_for(symbols, coordinates),
    )
    assert result.label == label


def test_chiral_groups_are_distinguished_from_their_mirror_extensions() -> None:
    """A chiral group and its mirror extension differ only in improper parts.

    The pair must not be confused in either direction: the chiral molecule
    must not acquire mirrors it does not have, and the extended one must not
    lose them.  ``I``/``Ih`` are checked by the sweep above rather than
    rebuilt here, since their orbits are the two largest.
    """
    for chiral, extended in (("T", "Td"), ("T", "Th"), ("O", "Oh")):
        results = {}
        for label in (chiral, extended):
            symbols, coordinates = reference.build_molecule(label)
            results[label] = analyze(
                molecule_from_coordinates(symbols, coordinates),
                tolerance=_tolerance_for(symbols, coordinates),
            )
            assert results[label].label == label
        improper = {
            label: sum(1 for op in result.operations if not op.is_proper)
            for label, result in results.items()
        }
        assert improper[chiral] == 0
        assert improper[extended] == expected_order(chiral)
        assert expected_order(extended) == 2 * expected_order(chiral)


def test_buckminsterfullerene_is_icosahedral() -> None:
    """A real 60-atom single-element molecule, not a synthetic orbit."""
    phi = reference.GOLDEN
    seeds = ((0.0, 1.0, 3.0 * phi), (1.0, 2.0 + phi, 2.0 * phi), (2.0, 1.0 + 2.0 * phi, phi))
    vertices: list[np.ndarray] = []
    for seed in seeds:
        for signs in np.ndindex(2, 2, 2):
            candidate = np.array(seed) * (1 - 2 * np.array(signs))
            for roll in range(3):
                point = np.roll(candidate, roll)
                if not any(np.allclose(point, other) for other in vertices):
                    vertices.append(point)
    coordinates = np.asarray(vertices)
    assert len(coordinates) == 60
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    coordinates = coordinates * (1.4 / distances.min())

    result = analyze(molecule_from_coordinates(["C"] * 60, coordinates))
    assert result.label == "Ih"
    assert result.order == 120
    assert result.consistent
