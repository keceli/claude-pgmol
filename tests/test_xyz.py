"""XYZ reading: the dialect that is accepted and everything that is not."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pgmol import Molecule, XYZFormatError, molecule_from_coordinates, parse_xyz, read_xyz

WATER = """3
water
O  0.0 0.0 0.0
H  0.757 0.0 0.587
H -0.757 0.0 0.587
"""


def test_parses_a_plain_file() -> None:
    molecule = parse_xyz(WATER)
    assert molecule.symbols == ("O", "H", "H")
    assert molecule.natoms == len(molecule) == 3
    assert molecule.comment == "water"
    assert molecule.coordinates.shape == (3, 3)
    assert molecule.coordinates.dtype == np.float64


def test_symbols_are_case_insensitive_and_canonicalised() -> None:
    text = "2\n\ncL 0 0 0\nh 0 0 1.3\n"
    assert parse_xyz(text).symbols == ("Cl", "H")


def test_atomic_numbers_are_accepted() -> None:
    text = "2\n\n17 0 0 0\n1 0 0 1.3\n"
    assert parse_xyz(text).symbols == ("Cl", "H")


def test_comment_line_may_be_empty() -> None:
    assert parse_xyz("1\n\nHe 0 0 0\n").comment == ""


def test_leading_and_trailing_blank_lines_are_tolerated() -> None:
    molecule = parse_xyz("\n\n" + WATER + "\n   \n\n")
    assert molecule.natoms == 3


def test_scientific_notation_and_signs_parse() -> None:
    text = "2\n\nH 1e-1 -2.5E0 +3.0\nH 0 0 0\n"
    assert parse_xyz(text).coordinates[0] == pytest.approx([0.1, -2.5, 3.0])


def test_reading_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "water.xyz"
    path.write_text(WATER, encoding="utf-8")
    assert read_xyz(path).natoms == 3
    assert read_xyz(str(path)).natoms == 3


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "file is empty"),
        ("   \n\n", "file is empty"),
        ("three\n\nH 0 0 0\n", "must be an atom count"),
        ("0\n\n", "must be positive"),
        ("-2\n\nH 0 0 0\n", "must be positive"),
        ("2\n", "missing comment line"),
        ("3\nc\nH 0 0 0\nH 0 0 1\n", "only 2 lines follow"),
        ("2\nc\nH 0 0 0\n\nH 0 0 1\n", "blank line inside the atom block"),
        ("1\nc\nH 0 0 0 0.5\n", "expected 4 fields"),
        ("1\nc\nH 0 0\n", "expected 4 fields"),
        ("1\nc\nXx 0 0 0\n", "unknown element symbol"),
        ("1\nc\n119 0 0 0\n", "unknown element symbol"),
        ("1\nc\nH 0 0 abc\n", "not a floating-point number"),
        ("1\nc\nH 0 0 nan\n", "not finite"),
        ("1\nc\nH 0 0 inf\n", "not finite"),
        ("1\nc\nH 0 0 0\n1\nc\nH 0 0 0\n", "multi-frame"),
        ("2\nc\nH 0 0 0\nH 0 0 0\n", "coincident"),
    ],
)
def test_malformed_input_is_rejected(text: str, fragment: str) -> None:
    with pytest.raises(XYZFormatError) as excinfo:
        parse_xyz(text, source="bad.xyz")
    assert fragment in str(excinfo.value)


def test_error_messages_carry_the_source_and_line() -> None:
    with pytest.raises(XYZFormatError, match=r"bad\.xyz:3"):
        parse_xyz("1\nc\nQq 0 0 0\n", source="bad.xyz")


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(XYZFormatError, match="cannot read file"):
        read_xyz(tmp_path / "absent.xyz")


def test_directory_instead_of_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(XYZFormatError, match="cannot read file"):
        read_xyz(tmp_path)


def test_non_utf8_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "binary.xyz"
    path.write_bytes(b"1\n\xff\xfe\nH 0 0 0\n")
    with pytest.raises(XYZFormatError, match="not valid UTF-8"):
        read_xyz(path)


def test_xyz_format_error_is_a_value_error() -> None:
    assert issubclass(XYZFormatError, ValueError)


def test_molecule_coordinates_are_read_only() -> None:
    molecule = parse_xyz(WATER)
    with pytest.raises(ValueError):
        molecule.coordinates[0, 0] = 1.0


def test_molecule_copies_its_input_array() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    molecule = Molecule(("H", "H"), coordinates)
    coordinates[0, 0] = 99.0
    assert molecule.coordinates[0, 0] == 0.0


def test_round_trip_through_to_xyz() -> None:
    molecule = parse_xyz(WATER)
    again = parse_xyz(molecule.to_xyz())
    assert again.symbols == molecule.symbols
    assert again.coordinates == pytest.approx(molecule.coordinates)
    assert parse_xyz(molecule.to_xyz(comment="other")).comment == "other"


def test_to_xyz_keeps_a_multiline_comment_on_one_line() -> None:
    molecule = Molecule(("H",), [[0.0, 0.0, 0.0]], comment="a\nb")
    assert len(molecule.to_xyz().splitlines()) == 3


@pytest.mark.parametrize(
    ("symbols", "coordinates", "fragment"),
    [
        (["H"], [[0.0, 0.0]], "shape"),
        (["H", "H"], [[0.0, 0.0, 0.0]], "coordinate rows"),
        ([], np.zeros((0, 3)), "at least one atom"),
        (["Zz"], [[0.0, 0.0, 0.0]], "unknown element symbol"),
        (["H"], [[0.0, 0.0, float("nan")]], "finite"),
    ],
)
def test_molecule_from_coordinates_validates(symbols, coordinates, fragment) -> None:
    with pytest.raises(XYZFormatError, match=fragment):
        molecule_from_coordinates(symbols, coordinates)


def test_molecule_from_coordinates_does_not_mutate_the_caller_array() -> None:
    coordinates = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    before = coordinates.copy()
    molecule_from_coordinates(["H", "H"], coordinates)
    assert coordinates == pytest.approx(before)


@pytest.mark.parametrize(
    ("symbols", "coordinates", "fragment"),
    [
        (("H",), [[0.0, 0.0]], "shape"),
        (("H", "H"), [[0.0, 0.0, 0.0]], "coordinate rows"),
        ((), np.zeros((0, 3)), "at least one atom"),
        (("H",), [[0.0, 0.0, float("inf")]], "finite"),
    ],
)
def test_molecule_constructor_validates(symbols, coordinates, fragment) -> None:
    with pytest.raises(ValueError, match=fragment):
        Molecule(symbols, coordinates)
