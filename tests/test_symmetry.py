"""Unit tests for the geometry and symmetry-operation internals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pgmol.classify import expected_order
from pgmol.elements import ELEMENT_SYMBOLS, canonical_symbol
from pgmol.geometry import (
    IDENTITY,
    INVERSION,
    StructureMatcher,
    candidate_directions,
    centered,
    centroid,
    deduplicate_directions,
    element_blocks,
    fit_orthogonal,
    improper_rotation_matrix,
    is_linear,
    minimum_equivalent_separation,
    principal_axes,
    reflection_matrix,
    rotation_matrix,
)
from pgmol.symmetry import detect_operations, identify_operation

Z = np.array([0.0, 0.0, 1.0])


# --- elements -------------------------------------------------------------


def test_element_table_is_complete() -> None:
    assert len(ELEMENT_SYMBOLS) == 118
    assert len(set(ELEMENT_SYMBOLS)) == 118


@pytest.mark.parametrize(
    ("token", "expected"),
    [("H", "H"), ("h", "H"), ("CL", "Cl"), ("  Fe ", "Fe"), ("1", "H"), ("118", "Og")],
)
def test_canonical_symbol_accepts(token: str, expected: str) -> None:
    assert canonical_symbol(token) == expected


@pytest.mark.parametrize("token", ["", "   ", "Xx", "0", "119", "-1", "H2"])
def test_canonical_symbol_rejects(token: str) -> None:
    assert canonical_symbol(token) is None


# --- matrix constructors ---------------------------------------------------


def test_rotation_matrix_is_a_proper_rotation() -> None:
    matrix = rotation_matrix([1.0, 2.0, 3.0], 0.7)
    assert matrix.T @ matrix == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_rotation_by_a_full_turn_is_the_identity() -> None:
    assert rotation_matrix(Z, 2 * math.pi) == pytest.approx(IDENTITY, abs=1e-12)


def test_reflection_matrix_is_improper_and_involutive() -> None:
    matrix = reflection_matrix([0.0, 1.0, 1.0])
    assert np.linalg.det(matrix) == pytest.approx(-1.0)
    assert matrix @ matrix == pytest.approx(IDENTITY, abs=1e-12)


def test_s2_is_inversion() -> None:
    assert improper_rotation_matrix(Z, math.pi) == pytest.approx(INVERSION, abs=1e-12)


# --- operation identification ---------------------------------------------


@pytest.mark.parametrize(
    ("matrix", "kind", "order"),
    [
        (IDENTITY, "E", 1),
        (INVERSION, "i", 2),
        (reflection_matrix(Z), "sigma", 2),
        (rotation_matrix(Z, math.pi), "Cn", 2),
        (rotation_matrix(Z, 2 * math.pi / 3), "Cn", 3),
        (rotation_matrix(Z, 4 * math.pi / 3), "Cn", 3),
        (rotation_matrix(Z, 2 * math.pi / 5), "Cn", 5),
        (rotation_matrix(Z, 4 * math.pi / 5), "Cn", 5),
        (improper_rotation_matrix(Z, math.pi / 2), "Sn", 4),
        (improper_rotation_matrix(Z, math.pi / 3), "Sn", 6),
        (improper_rotation_matrix(Z, 2 * math.pi / 5), "Sn", 10),
    ],
)
def test_identify_operation(matrix: np.ndarray, kind: str, order: int) -> None:
    operation = identify_operation(matrix)
    assert operation.kind == kind
    assert operation.order == order
    assert operation.determinant == pytest.approx(1.0 if kind in {"E", "Cn"} else -1.0)
    assert operation.is_proper == (kind in {"E", "Cn"})


@pytest.mark.parametrize(
    "matrix",
    [
        rotation_matrix([1.0, 1.0, 0.0], math.pi),
        rotation_matrix([0.3, -0.7, 0.2], 2 * math.pi / 4),
        improper_rotation_matrix([1.0, 0.0, 2.0], math.pi / 2),
        reflection_matrix([2.0, -1.0, 0.5]),
    ],
)
def test_identified_axis_is_recovered(matrix: np.ndarray) -> None:
    operation = identify_operation(matrix)
    assert operation.axis is not None
    assert np.linalg.norm(operation.axis) == pytest.approx(1.0)
    # The axis is an eigenvector: +1 for rotations, -1 for S_n and mirrors.
    eigenvalue = 1.0 if operation.kind == "Cn" else -1.0
    assert matrix @ operation.axis == pytest.approx(eigenvalue * operation.axis, abs=1e-9)


def test_operation_labels() -> None:
    assert identify_operation(IDENTITY).label == "E"
    assert identify_operation(INVERSION).label == "i"
    assert identify_operation(reflection_matrix(Z)).label == "sigma"
    assert identify_operation(rotation_matrix(Z, 2 * math.pi / 3)).label == "C3"
    assert identify_operation(improper_rotation_matrix(Z, math.pi / 2)).label == "S4"


def test_non_orthogonal_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="not orthogonal"):
        identify_operation(np.diag([1.0, 1.0, 2.0]))


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="3x3"):
        identify_operation(np.eye(2))


def test_rotation_by_an_irrational_fraction_has_no_finite_order() -> None:
    with pytest.raises(ValueError, match="no finite order"):
        identify_operation(rotation_matrix(Z, 1.0), max_order=64)


# --- geometry helpers ------------------------------------------------------


def test_centroid_and_centering() -> None:
    coordinates = np.array([[1.0, 0.0, 0.0], [3.0, 2.0, 4.0]])
    assert centroid(coordinates) == pytest.approx([2.0, 1.0, 2.0])
    assert centered(coordinates).mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])


def test_centering_does_not_modify_the_input() -> None:
    coordinates = np.array([[1.0, 0.0, 0.0], [3.0, 2.0, 4.0]])
    before = coordinates.copy()
    centered(coordinates)
    assert coordinates == pytest.approx(before)


def test_is_linear() -> None:
    line = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    bent = np.array([[0.0, 0.0, -1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert is_linear(centered(line), 0.1)
    assert not is_linear(centered(bent), 0.1)
    assert is_linear(centered(bent), 0.6)
    assert is_linear(np.zeros((1, 3)), 0.1)
    assert is_linear(np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), 0.1)


def test_principal_axes_are_orthonormal_and_ordered() -> None:
    rng = np.random.default_rng(3)
    coordinates = rng.normal(size=(12, 3)) * np.array([5.0, 2.0, 0.5])
    axes = principal_axes(coordinates)
    assert axes @ axes.T == pytest.approx(np.eye(3), abs=1e-10)
    extents = [np.abs(centered(coordinates) @ axis).max() for axis in axes]
    assert extents[0] >= extents[2]


def test_element_blocks_group_by_symbol_in_order() -> None:
    blocks = element_blocks(["H", "O", "H", "C"])
    assert list(blocks) == ["H", "O", "C"]
    assert blocks["H"].tolist() == [0, 2]


def test_minimum_equivalent_separation() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 3.0, 0.0]])
    blocks = element_blocks(["H", "H", "O"])
    assert minimum_equivalent_separation(coordinates, blocks) == pytest.approx(1.5)
    singles = element_blocks(["H", "O", "N"])
    assert minimum_equivalent_separation(coordinates, singles) == math.inf


def test_deduplicate_directions_merges_parallel_and_antiparallel() -> None:
    directions = [[0, 0, 1], [0, 0, -1], [0, 0, 2], [1, 0, 0], [0.0, 0.0, 0.0]]
    unique = deduplicate_directions(directions, 1e-3)
    assert len(unique) == 2
    assert np.linalg.norm(unique, axis=1) == pytest.approx([1.0, 1.0])


def test_deduplicate_directions_on_empty_input() -> None:
    assert deduplicate_directions([], 1e-3).shape == (0, 3)
    assert deduplicate_directions([[0.0, 0.0, 0.0]], 1e-3).shape == (0, 3)


def test_candidate_directions_include_the_obvious_axes() -> None:
    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    blocks = element_blocks(["O", "H", "H"])
    directions = candidate_directions(coordinates, blocks, 1e-3, 0.1)
    assert any(abs(abs(float(d @ Z)) - 1.0) < 1e-6 for d in directions), "C2 axis missing"
    normal = np.array([0.0, 1.0, 0.0])
    assert any(abs(abs(float(d @ normal)) - 1.0) < 1e-6 for d in directions)


# --- structure matching and fitting ---------------------------------------


def test_structure_matcher_accepts_and_rejects() -> None:
    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    matcher = StructureMatcher(coordinates, element_blocks(["O", "H", "H"]), 0.1)
    assert matcher.matches(IDENTITY)
    assert matcher.matches(rotation_matrix(Z, math.pi))
    assert not matcher.matches(rotation_matrix(Z, math.pi / 2))
    assert not matcher.matches(INVERSION)


def test_permutation_reports_the_atom_mapping() -> None:
    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    matcher = StructureMatcher(coordinates, element_blocks(["O", "H", "H"]), 0.1)
    assert matcher.permutation(IDENTITY).tolist() == [0, 1, 2]
    assert matcher.permutation(rotation_matrix(Z, math.pi)).tolist() == [0, 2, 1]
    assert matcher.permutation(rotation_matrix(Z, math.pi / 2)) is None


def test_fit_orthogonal_recovers_the_generating_matrix() -> None:
    turn = rotation_matrix(Z, 2 * math.pi / 3)
    first = np.array([1.0, 0.2, 0.3])
    points = np.array([first, turn @ first, turn @ turn @ first])
    fitted = fit_orthogonal(points, np.array([1, 2, 0]), 1)
    assert fitted == pytest.approx(turn, abs=1e-12)


def test_fit_orthogonal_respects_the_requested_determinant() -> None:
    planar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    identity_permutation = np.array([0, 1, 2])
    assert fit_orthogonal(planar, identity_permutation, 1) == pytest.approx(
        IDENTITY, abs=1e-12
    )
    # The same permutation with the opposite sign is the molecular plane.
    assert fit_orthogonal(planar, identity_permutation, -1) == pytest.approx(
        reflection_matrix(Z), abs=1e-12
    )


# --- group assembly --------------------------------------------------------


def test_detected_group_is_closed_and_reports_counts() -> None:
    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    group = detect_operations(["O", "H", "H"], coordinates, 0.1)
    assert group.order == 4
    assert group.closed
    assert group.has_inversion is False
    assert group.counts() == {"C2": 1, "E": 1, "sigma": 2}
    assert len(group.mirror_normals) == 2
    assert [order for _, order in group.rotation_axes()] == [2]
    assert group.improper_axes() == []
    assert len(list(group)) == 4


def test_detected_group_products_stay_inside_the_group() -> None:
    coordinates = centered(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
    )
    group = detect_operations(["C", "H", "H", "H", "H"], coordinates, 0.1)
    assert group.order == 24
    matrices = [operation.matrix for operation in group]
    for left in matrices:
        for right in matrices:
            product = left @ right
            assert any(np.abs(product - other).max() < 1e-6 for other in matrices)


# --- label bookkeeping -----------------------------------------------------


@pytest.mark.parametrize(
    ("label", "order"),
    [
        ("C1", 1),
        ("Cs", 2),
        ("Ci", 2),
        ("C3", 3),
        ("C4v", 8),
        ("C2h", 4),
        ("S4", 4),
        ("S6", 6),
        ("D2", 4),
        ("D6h", 24),
        ("D3d", 12),
        ("T", 12),
        ("Td", 24),
        ("Th", 24),
        ("O", 24),
        ("Oh", 48),
        ("I", 60),
        ("Ih", 120),
    ],
)
def test_expected_order(label: str, order: int) -> None:
    assert expected_order(label) == order


@pytest.mark.parametrize("label", ["Kh", "Cinfv", "Dinfh"])
def test_infinite_groups_have_no_order(label: str) -> None:
    assert expected_order(label) is None


@pytest.mark.parametrize("label", ["Q7", "C3x", "Dxh", "C"])
def test_unknown_labels_are_rejected(label: str) -> None:
    with pytest.raises(ValueError, match="unrecognised"):
        expected_order(label)


# --- rejecting accidental near-symmetries ---------------------------------

SCALENE = np.array([[1.0, 0.0, 0.0], [-0.6, 0.9, 0.0], [-0.4, -0.9, 0.3]])


def _matcher_for(coordinates: np.ndarray, symbols, tolerance: float) -> StructureMatcher:
    centred = centered(coordinates)
    return StructureMatcher(centred, element_blocks(symbols), tolerance)


def test_accept_operation_canonicalises_a_valid_candidate() -> None:
    from pgmol.symmetry import accept_operation

    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    matcher = StructureMatcher(coordinates, element_blocks(["O", "H", "H"]), 0.1)
    operation = accept_operation(
        coordinates,
        matcher,
        np.array([0, 2, 1]),
        1,
        identify_atol=1e-4,
        order_limit=4,
        proper_limit=2,
    )
    assert operation is not None
    assert operation.kind == "Cn"
    assert operation.order == 2


def test_accept_operation_rejects_a_fit_that_no_longer_matches() -> None:
    """Swapping two inequivalent atoms: the best-fit matrix misses them both."""
    from pgmol.symmetry import accept_operation

    coordinates = centered(SCALENE)
    matcher = StructureMatcher(coordinates, element_blocks(["H", "H", "H"]), 0.05)
    assert (
        accept_operation(
            coordinates,
            matcher,
            np.array([1, 0, 2]),
            1,
            identify_atol=1e-4,
            order_limit=6,
            proper_limit=3,
        )
        is None
    )
    # With enough slack the same swap becomes an acceptable near-symmetry,
    # which is exactly what the tolerance is for.
    generous = StructureMatcher(coordinates, element_blocks(["H", "H", "H"]), 0.1)
    assert (
        accept_operation(
            coordinates,
            generous,
            np.array([1, 0, 2]),
            1,
            identify_atol=1e-4,
            order_limit=6,
            proper_limit=3,
        )
        is not None
    )


def test_accept_operation_rejects_a_cycle_with_no_finite_order() -> None:
    """A three-cycle of inequivalent atoms fits no rotation of finite order."""
    from pgmol.symmetry import accept_operation

    coordinates = centered(SCALENE)
    matcher = StructureMatcher(coordinates, element_blocks(["H", "H", "H"]), 0.3)
    assert (
        accept_operation(
            coordinates,
            matcher,
            np.array([1, 2, 0]),
            1,
            identify_atol=1e-4,
            order_limit=6,
            proper_limit=3,
        )
        is None
    )


def test_accept_operation_rejects_an_order_above_the_atom_count() -> None:
    """A rotation of order n needs an n-atom orbit, so high orders are bogus."""
    from pgmol.symmetry import accept_operation

    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    matcher = StructureMatcher(coordinates, element_blocks(["O", "H", "H"]), 0.1)
    kwargs = {"identify_atol": 1e-4, "order_limit": 4}
    assert (
        accept_operation(
            coordinates, matcher, np.array([0, 2, 1]), 1, proper_limit=2, **kwargs
        )
        is not None
    )
    assert (
        accept_operation(
            coordinates, matcher, np.array([0, 2, 1]), 1, proper_limit=1, **kwargs
        )
        is None
    )


def test_accept_operation_rejects_when_no_order_fits_the_limit() -> None:
    from pgmol.symmetry import accept_operation

    coordinates = centered(
        np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    )
    matcher = StructureMatcher(coordinates, element_blocks(["O", "H", "H"]), 0.1)
    assert (
        accept_operation(
            coordinates,
            matcher,
            np.array([0, 2, 1]),
            1,
            identify_atol=1e-4,
            order_limit=1,
            proper_limit=2,
        )
        is None
    )


def test_identify_or_none_returns_none_instead_of_raising() -> None:
    from pgmol.symmetry import _identify_or_none

    assert _identify_or_none(np.diag([1.0, 1.0, 2.0]), 1e-4, 64) is None
    assert _identify_or_none(rotation_matrix(Z, 1.0), 1e-9, 64) is None
    assert _identify_or_none(IDENTITY, 1e-4, 64) is not None


def test_matcher_rejects_an_ambiguous_many_to_one_match() -> None:
    """Two atoms landing on the same partner is not a permutation.

    Four atoms on a circle at 0, 12, 180 and 192 degrees: a 7 degree turn
    puts the first two both nearest to the atom at 12 degrees, so with a
    tolerance that wide the correspondence is ambiguous and is refused
    rather than guessed at.
    """
    angles = [0.0, 12.0, 180.0, 192.0]
    coordinates = np.array(
        [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0] for a in angles]
    )
    assert coordinates.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    generous = StructureMatcher(coordinates, element_blocks(["H"] * 4), 0.5)
    assert generous.permutation(rotation_matrix(Z, math.radians(7.0))) is None
    # A tolerance below half the 12-degree separation is unambiguous again.
    strict = StructureMatcher(coordinates, element_blocks(["H"] * 4), 0.05)
    assert strict.permutation(rotation_matrix(Z, math.radians(7.0))) is None
    assert strict.permutation(IDENTITY).tolist() == [0, 1, 2, 3]


# --- group bookkeeping edge cases -----------------------------------------


def test_identity_and_inversion_contribute_no_axes() -> None:
    from pgmol.symmetry import SymmetryGroup

    group = SymmetryGroup(
        operations=(
            identify_operation(IDENTITY),
            identify_operation(INVERSION),
            identify_operation(rotation_matrix(Z, math.pi)),
        ),
        angular_tolerance=1e-3,
    )
    assert [order for _, order in group.rotation_axes()] == [2]
    assert group.has_inversion


def test_angular_tolerance_falls_back_for_a_zero_radius() -> None:
    from pgmol.symmetry import _angular_tolerance

    assert _angular_tolerance(0.1, 0.0) == 1e-3
    assert _angular_tolerance(0.1, 1.0) == pytest.approx(0.1)
    assert _angular_tolerance(100.0, 1.0) == 0.2  # clamped
    assert _angular_tolerance(1e-12, 1.0) == 1e-6  # clamped


def test_axis_candidates_yield_nothing_when_every_atom_is_on_the_axis() -> None:
    """``detect_operations`` is for non-linear molecules; it must not crash."""
    coordinates = centered(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]))
    group = detect_operations(["H", "H"], coordinates, 0.1)
    assert group.order >= 1


# --- closure ---------------------------------------------------------------


def _closure_harness(seed_permutations, accept=True):
    from pgmol.symmetry import _close_under_composition

    permutations = [np.asarray(p) for p in seed_permutations]
    signs = [1] * len(permutations)
    operations = [object() for _ in permutations]
    known = {(1, *map(int, p)) for p in permutations}

    def register(permutation, sign):
        key = (sign, *(int(v) for v in permutation))
        if key in known:
            return True
        if not accept:
            return False
        known.add(key)
        operations.append(object())
        permutations.append(np.asarray(permutation))
        signs.append(sign)
        return True

    closed = _close_under_composition(operations, permutations, signs, register)
    return closed, operations


def test_closure_generates_the_cyclic_group_from_one_generator() -> None:
    closed, operations = _closure_harness([[0, 1, 2, 3], [1, 2, 3, 0]])
    assert closed
    assert len(operations) == 4  # E, C4, C2, C4^3


def test_closure_reports_a_set_that_is_not_a_group() -> None:
    closed, operations = _closure_harness([[0, 1, 2, 3], [1, 2, 3, 0]], accept=False)
    assert not closed
    assert len(operations) == 2


def test_closure_bails_out_on_a_runaway_group(monkeypatch) -> None:
    import pgmol.symmetry as symmetry

    monkeypatch.setattr(symmetry, "_MAX_GROUP_ORDER", 3)
    closed, _ = _closure_harness([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    assert not closed
