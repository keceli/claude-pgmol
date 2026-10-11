"""pgmol -- determine the Schoenflies point group of a molecule from its geometry.

Example
-------
>>> from pgmol import find_point_group
>>> find_point_group("examples/water.xyz")
'C2v'

Importing the package has no side effects beyond importing :mod:`numpy`.
"""

from __future__ import annotations

from .api import (
    DEFAULT_TOLERANCE,
    SymmetryAnalysis,
    analyze,
    find_point_group,
    point_group_from_coordinates,
)
from .errors import PGMolError, SymmetryError, XYZFormatError
from .symmetry import SymmetryElement, SymmetryOperation
from .xyz import Molecule, molecule_from_coordinates, parse_xyz, read_xyz

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TOLERANCE",
    "Molecule",
    "PGMolError",
    "SymmetryAnalysis",
    "SymmetryElement",
    "SymmetryError",
    "SymmetryOperation",
    "XYZFormatError",
    "__version__",
    "analyze",
    "find_point_group",
    "molecule_from_coordinates",
    "parse_xyz",
    "point_group_from_coordinates",
    "read_xyz",
]
