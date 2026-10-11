"""Symmetry *elements* — the geometry, as opposed to the operations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import acceptance_files
from pgmol import SymmetryElement, analyze, read_xyz

# Counts straight from the character tables: axes (merged up to sign),
# mirror planes, and the inversion centre.
EXPECTED = {
    "H2O_C2v": ["C2", "sigma", "sigma"],
    "H3N_C3v": ["C3", "sigma", "sigma", "sigma"],
    "CH4_Td": ["C3"] * 4 + ["C2/S4"] * 3 + ["sigma"] * 6,
    "C2H6_D3d": ["C3/S6"] + ["C2"] * 3 + ["sigma"] * 3 + ["i"],
    "C6H6_D6h": ["C6/S6"] + ["C2"] * 6 + ["sigma"] * 7 + ["i"],
    "C2Cl2F2H2_Ci": ["i"],
    "CClFH2_Cs": ["sigma"],
    "BrCClFH_C1": [],
    "F6S_Oh": ["C4/S4"] * 3 + ["C3/S6"] * 4 + ["C2"] * 6 + ["sigma"] * 9 + ["i"],
}


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_elements_match_the_character_tables(stem: str) -> None:
    result = analyze(read_xyz(Path("core_xyz") / f"{stem}.xyz"))
    assert sorted(e.label for e in result.elements) == sorted(EXPECTED[stem])


def test_an_axis_reports_both_its_orders() -> None:
    """The C3 axis of D3d is also its S6 axis; one element, two orders."""
    result = analyze(read_xyz(Path("core_xyz") / "C2H6_D3d.xyz"))
    principal = max(result.elements, key=lambda e: e.rotation_order)
    assert principal.kind == "axis"
    assert principal.rotation_order == 3
    assert principal.improper_order == 6
    assert principal.label == "C3/S6"


def test_improper_order_is_the_axis_order_not_the_matrix_order() -> None:
    """S3 has matrix order 6 but is still S3 — the same trap as the labels."""
    result = analyze(read_xyz(Path("core_xyz") / "BF3_D3h.xyz"))
    principal = max(result.elements, key=lambda e: e.rotation_order)
    assert (principal.rotation_order, principal.improper_order) == (3, 3)
    assert principal.label == "C3/S3"


@pytest.mark.parametrize("path", acceptance_files(), ids=lambda p: p.stem)
def test_element_directions_are_unit_vectors_and_distinct(path: Path) -> None:
    result = analyze(read_xyz(path))
    directions = []
    for element in result.elements:
        if element.kind == "inversion":
            assert element.direction is None
            continue
        assert element.direction is not None
        assert np.linalg.norm(element.direction) == pytest.approx(1.0)
        directions.append((element.kind, element.direction))
    # No two elements of the same kind may share a direction, up to sign.
    for i, (kind_a, a) in enumerate(directions):
        for kind_b, b in directions[i + 1 :]:
            if kind_a == kind_b:
                assert abs(float(np.dot(a, b))) < math.cos(1e-3)


@pytest.mark.parametrize("path", acceptance_files(), ids=lambda p: p.stem)
def test_elements_are_consistent_with_the_operations(path: Path) -> None:
    """Every element must be justified by an operation that was found."""
    result = analyze(read_xyz(path))
    kinds = {op.kind for op in result.operations}
    for element in result.elements:
        if element.kind == "inversion":
            assert "i" in kinds
        elif element.kind == "plane":
            assert "sigma" in kinds
        else:
            assert element.rotation_order > 1 or element.improper_order > 0
    if "i" in kinds:
        assert any(e.kind == "inversion" for e in result.elements)
    assert sum(1 for e in result.elements if e.kind == "plane") == sum(
        1 for op in result.operations if op.kind == "sigma"
    )


def test_infinite_groups_have_no_finite_element_set() -> None:
    for stem in ("CO2_Dinfh", "CHN_Cinfv", "Ar_Kh"):
        assert analyze(read_xyz(Path("core_xyz") / f"{stem}.xyz")).elements == ()


def test_elements_are_ordered_deterministically() -> None:
    """Rendering order must not depend on which operation was found first."""
    molecule = read_xyz(Path("core_xyz") / "C6H6_D6h.xyz")
    rng = np.random.default_rng(4)
    first = [e.label for e in analyze(molecule).elements]
    for _ in range(3):
        order = rng.permutation(molecule.natoms)
        shuffled = type(molecule)(
            tuple(molecule.symbols[i] for i in order), molecule.coordinates[order]
        )
        assert [e.label for e in analyze(shuffled).elements] == first


def test_elements_compare_and_hash_by_value() -> None:
    direction = np.array([0.0, 0.0, 1.0])
    a = SymmetryElement("axis", direction, 3, 6)
    b = SymmetryElement("axis", direction.copy(), 3, 6)
    centre = SymmetryElement("inversion", None)
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert a != SymmetryElement("axis", direction, 3, 0)
    assert a != centre and centre == SymmetryElement("inversion", None)
    assert a.__eq__("C3") is NotImplemented
    # Negative zero must not split an element from its own equal.
    assert SymmetryElement("plane", np.array([0.0, 1.0, 0.0])) == SymmetryElement(
        "plane", np.array([0.0, 1.0, -0.0])
    )
    assert hash(SymmetryElement("plane", np.array([0.0, 1.0, 0.0]))) == hash(
        SymmetryElement("plane", np.array([0.0, 1.0, -0.0]))
    )


def test_c1_label_for_an_axis_carrying_nothing() -> None:
    assert SymmetryElement("axis", np.array([0.0, 0.0, 1.0])).label == "C1"
