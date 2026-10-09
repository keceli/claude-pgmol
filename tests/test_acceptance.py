"""Acceptance: every public fixture, through the public API only."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import acceptance_files, expected_label
from pgmol import analyze, find_point_group, point_group_from_coordinates, read_xyz
from pgmol.classify import expected_order

FIXTURES = acceptance_files()


def test_fixture_directory_is_populated() -> None:
    assert len(FIXTURES) >= 30, "acceptance fixtures are missing"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_point_group_matches_file_name(path: Path) -> None:
    assert find_point_group(path) == expected_label(path)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_coordinate_entry_point_agrees_with_file_entry_point(path: Path) -> None:
    molecule = read_xyz(path)
    from_coordinates = point_group_from_coordinates(molecule.symbols, molecule.coordinates)
    assert from_coordinates == expected_label(path)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_detected_group_has_the_textbook_order(path: Path) -> None:
    """The operations found must form a group of the size the label implies.

    This is an independent check: the order comes from group theory, not
    from the detection code.
    """
    result = analyze(read_xyz(path))
    assert result.order == expected_order(result.label)
    assert result.consistent
