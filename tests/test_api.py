"""Public API behaviour: entry points, analysis records and special cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pgmol
from pgmol import (
    DEFAULT_TOLERANCE,
    SymmetryError,
    analyze,
    find_point_group,
    point_group_from_coordinates,
    read_xyz,
)


def test_version_is_exposed() -> None:
    assert isinstance(pgmol.__version__, str) and pgmol.__version__


def test_public_names_are_importable() -> None:
    for name in pgmol.__all__:
        assert hasattr(pgmol, name), name


def test_find_point_group_on_the_example(water_xyz: Path) -> None:
    assert find_point_group(water_xyz) == "C2v"
    assert find_point_group(str(water_xyz)) == "C2v"


def test_default_tolerance_is_documented_value() -> None:
    assert DEFAULT_TOLERANCE == 0.1


# --- single atom and linear molecules ------------------------------------


def test_single_atom_is_spherical() -> None:
    assert point_group_from_coordinates(["Ar"], [[0.3, -0.2, 1.0]]) == "Kh"


def test_homonuclear_diatomic_is_dinfh() -> None:
    assert point_group_from_coordinates(["H", "H"], [[0, 0, 0.37], [0, 0, -0.37]]) == "Dinfh"


def test_heteronuclear_diatomic_is_cinfv() -> None:
    assert point_group_from_coordinates(["C", "O"], [[0, 0, 0.6], [0, 0, -0.5]]) == "Cinfv"


def test_linear_triatomic_with_and_without_inversion() -> None:
    symmetric = point_group_from_coordinates(
        ["O", "C", "O"], [[0, 0, 1.16], [0, 0, 0], [0, 0, -1.16]]
    )
    asymmetric = point_group_from_coordinates(
        ["H", "C", "N"], [[0, 0, 1.6], [0, 0, 0.5], [0, 0, -0.65]]
    )
    assert symmetric == "Dinfh"
    assert asymmetric == "Cinfv"


def test_linear_molecule_off_the_coordinate_axes() -> None:
    direction = np.array([1.0, 2.0, -2.0]) / 3.0
    coordinates = np.array([5.0 * direction, 4.0 * direction, 3.0 * direction])
    assert point_group_from_coordinates(["H", "C", "N"], coordinates) == "Cinfv"


def test_bent_triatomic_is_not_linear() -> None:
    assert (
        point_group_from_coordinates(
            ["O", "H", "H"], [[0, 0, 0], [0.757, 0, 0.587], [-0.757, 0, 0.587]]
        )
        == "C2v"
    )


def test_infinite_groups_report_no_order() -> None:
    for symbols, coordinates in (
        (["Ar"], [[0.0, 0.0, 0.0]]),
        (["H", "H"], [[0, 0, 0.37], [0, 0, -0.37]]),
    ):
        result = analyze(pgmol.molecule_from_coordinates(symbols, coordinates))
        assert result.order is None
        assert result.operations == ()
        assert result.consistent


# --- the analysis record -------------------------------------------------


def test_analysis_fields(water_xyz: Path) -> None:
    result = analyze(read_xyz(water_xyz))
    assert result.label == "C2v"
    assert result.tolerance == DEFAULT_TOLERANCE
    assert result.effective_tolerance == DEFAULT_TOLERANCE
    assert result.order == 4
    assert result.consistent
    assert result.operation_counts() == {"C2": 1, "E": 1, "sigma": 2}


def test_operations_are_orthogonal_and_map_the_molecule(water_xyz: Path) -> None:
    molecule = read_xyz(water_xyz)
    result = analyze(molecule)
    centred = molecule.coordinates - molecule.coordinates.mean(axis=0)
    for operation in result.operations:
        assert operation.matrix.T @ operation.matrix == pytest.approx(np.eye(3), abs=1e-9)
        moved = centred @ operation.matrix.T
        for position, symbol in zip(moved, molecule.symbols, strict=True):
            distances = [
                np.linalg.norm(position - other)
                for other, other_symbol in zip(centred, molecule.symbols, strict=True)
                if other_symbol == symbol
            ]
            assert min(distances) <= DEFAULT_TOLERANCE


def test_analysis_is_immutable(water_xyz: Path) -> None:
    result = analyze(read_xyz(water_xyz))
    with pytest.raises(AttributeError):
        result.label = "C1"  # type: ignore[misc]


# --- tolerance validation -------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_non_positive_tolerance_is_rejected(water_xyz: Path, bad: float) -> None:
    with pytest.raises(SymmetryError, match="positive finite"):
        find_point_group(water_xyz, tolerance=bad)


def test_symmetry_error_is_a_value_error() -> None:
    assert issubclass(SymmetryError, ValueError)
    assert issubclass(SymmetryError, pgmol.PGMolError)


# --- caller data is never touched ----------------------------------------


def test_coordinates_passed_in_are_not_modified() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    before = coordinates.copy()
    assert point_group_from_coordinates(["O", "H", "H"], coordinates) == "C2v"
    assert coordinates == pytest.approx(before)


def test_molecule_is_not_modified_by_analysis(water_xyz: Path) -> None:
    molecule = read_xyz(water_xyz)
    before = molecule.coordinates.copy()
    analyze(molecule)
    assert molecule.coordinates == pytest.approx(before)


def test_results_do_not_depend_on_where_the_molecule_sits() -> None:
    base = np.array([[0.0, 0.0, 0.0], [0.757, 0.0, 0.587], [-0.757, 0.0, 0.587]])
    shifted = base + np.array([123.0, -45.0, 6.7])
    assert point_group_from_coordinates(["O", "H", "H"], shifted) == "C2v"


def test_importing_the_package_has_no_side_effects(tmp_path: Path) -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", "import pgmol"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
