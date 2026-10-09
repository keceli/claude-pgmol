"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCEPTANCE_DIR = REPO_ROOT / "core_xyz"
EXAMPLES_DIR = REPO_ROOT / "examples"

# Make the sibling helper module importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def acceptance_files() -> list[Path]:
    """Every public acceptance fixture, sorted for deterministic test ids."""
    return sorted(ACCEPTANCE_DIR.glob("*.xyz"))


def expected_label(path: Path) -> str:
    """The point group a fixture file name promises.

    Expectations are derived here, in the tests, and never inside the
    package: the implementation only ever sees the file's contents.
    """
    return path.stem.rsplit("_", 1)[-1]


@pytest.fixture(scope="session")
def acceptance_dir() -> Path:
    return ACCEPTANCE_DIR


@pytest.fixture(scope="session")
def water_xyz() -> Path:
    return EXAMPLES_DIR / "water.xyz"
