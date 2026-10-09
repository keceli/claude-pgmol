"""Public entry points for molecular point-group determination."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classify import classify_group, expected_order
from .errors import SymmetryError
from .geometry import (
    INVERSION,
    StructureMatcher,
    centered,
    element_blocks,
    is_linear,
    minimum_equivalent_separation,
)
from .symmetry import SymmetryGroup, SymmetryOperation, detect_operations
from .xyz import Molecule, molecule_from_coordinates, read_xyz

__all__ = [
    "SymmetryAnalysis",
    "analyze",
    "find_point_group",
    "point_group_from_coordinates",
]

DEFAULT_TOLERANCE = 0.1

#: Fraction of the shortest same-element interatomic distance that the
#: matching tolerance is capped at.  Below one half, a transformed atom can
#: lie within tolerance of at most one partner, so the atom correspondence --
#: and therefore the whole analysis -- is unambiguous.
_AMBIGUITY_FRACTION = 0.49


@dataclass(frozen=True)
class SymmetryAnalysis:
    """Result of a point-group determination.

    Attributes
    ----------
    label:
        Canonical Schoenflies label, for example ``"C2v"`` or ``"Dinfh"``.
    tolerance:
        Distance tolerance, in ångströms, as requested by the caller.
    effective_tolerance:
        Tolerance actually used for atom matching.  Equal to ``tolerance``
        unless that would exceed half the shortest distance between two
        atoms of the same element, in which case it is capped so that each
        transformed atom has at most one possible partner.
    order:
        Number of symmetry operations, or ``None`` for the infinite groups
        ``Kh``, ``Cinfv`` and ``Dinfh``.
    operations:
        The operations found, empty for the infinite groups.
    consistent:
        ``True`` when the operations are closed under composition *and* their
        count matches the order implied by ``label``.  ``False`` flags a
        geometry that is only approximately symmetric at this tolerance.
    """

    label: str
    tolerance: float
    effective_tolerance: float
    order: int | None
    operations: tuple[SymmetryOperation, ...]
    consistent: bool

    def operation_counts(self) -> dict[str, int]:
        """Number of operations of each type, keyed by label such as ``"C3"``."""
        tally: dict[str, int] = {}
        for op in self.operations:
            tally[op.label] = tally.get(op.label, 0) + 1
        return dict(sorted(tally.items()))


def _validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0.0:
        raise SymmetryError(f"tolerance must be a positive finite number, got {tolerance!r}")
    return value


def _effective_tolerance(
    coordinates: np.ndarray, blocks: dict[str, np.ndarray], tolerance: float
) -> float:
    """Cap ``tolerance`` so that atom-to-atom matching stays unambiguous."""
    separation = minimum_equivalent_separation(coordinates, blocks)
    if not math.isfinite(separation):
        return tolerance
    return min(tolerance, _AMBIGUITY_FRACTION * separation)


def analyze(molecule: Molecule, tolerance: float = DEFAULT_TOLERANCE) -> SymmetryAnalysis:
    """Determine the point group of ``molecule``.

    Parameters
    ----------
    molecule:
        Geometry to analyse.  Its coordinates are never modified.
    tolerance:
        Maximum distance, in ångströms, that an atom may be displaced from
        the position of its symmetry partner for an operation to be accepted.
        Larger values make the search more permissive and can promote a
        slightly distorted geometry to a higher-symmetry label.  The value is
        capped internally at just under half the shortest same-element
        interatomic distance, where atom matching would otherwise become
        ambiguous; see :attr:`SymmetryAnalysis.effective_tolerance`.

    Returns
    -------
    SymmetryAnalysis

    Raises
    ------
    SymmetryError
        If ``tolerance`` is not a positive finite number.
    """
    tol = _validate_tolerance(tolerance)
    coords = centered(molecule.coordinates)

    if molecule.natoms == 1:
        return SymmetryAnalysis("Kh", tol, tol, None, (), True)

    blocks = element_blocks(molecule.symbols)
    effective = _effective_tolerance(coords, blocks, tol)

    # Collinearity is a plain geometric threshold with no matching ambiguity,
    # so it uses the tolerance exactly as requested.
    if is_linear(coords, tol):
        matcher = StructureMatcher(coords, blocks, effective)
        label = "Dinfh" if matcher.matches(INVERSION) else "Cinfv"
        return SymmetryAnalysis(label, tol, effective, None, (), True)

    group: SymmetryGroup = detect_operations(molecule.symbols, coords, effective)
    label = classify_group(group)
    consistent = group.closed and expected_order(label) == group.order
    return SymmetryAnalysis(label, tol, effective, group.order, group.operations, consistent)


def find_point_group(path: str | Path, tolerance: float = DEFAULT_TOLERANCE) -> str:
    """Return the Schoenflies point group of the molecule in an XYZ file.

    Parameters
    ----------
    path:
        Path to a single-frame XYZ file in ångströms.
    tolerance:
        Maximum distance, in ångströms, between a symmetry-transformed atom
        and the atom it is matched with.  Defaults to ``0.1`` Å.

    Returns
    -------
    str
        A canonical Schoenflies label: ``"Kh"`` for a single atom,
        ``"Cinfv"``/``"Dinfh"`` for linear molecules, and labels such as
        ``"C1"``, ``"Cs"``, ``"Ci"``, ``"C2v"``, ``"D3h"``, ``"S4"``,
        ``"Td"``, ``"Oh"`` or ``"Ih"`` otherwise.

    Raises
    ------
    XYZFormatError
        If the file is missing, unreadable or not a well-formed XYZ file.
    SymmetryError
        If ``tolerance`` is not a positive finite number.

    Examples
    --------
    >>> find_point_group("examples/water.xyz")
    'C2v'
    """
    return analyze(read_xyz(path), tolerance=tolerance).label


def point_group_from_coordinates(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> str:
    """Return the Schoenflies point group of an in-memory geometry.

    Parameters
    ----------
    symbols:
        One element symbol per atom.  Case-insensitive; atomic numbers are
        also accepted.
    coordinates:
        ``(natoms, 3)`` array-like of ångström coordinates.  The input is
        copied, so the caller's array is never modified.
    tolerance:
        As for :func:`find_point_group`.

    Returns
    -------
    str

    Raises
    ------
    XYZFormatError
        If the symbols or coordinates are inconsistent or non-finite.
    SymmetryError
        If ``tolerance`` is not a positive finite number.

    Examples
    --------
    >>> point_group_from_coordinates(["H", "H"], [[0, 0, 0.37], [0, 0, -0.37]])
    'Dinfh'
    """
    molecule = molecule_from_coordinates(symbols, coordinates)
    return analyze(molecule, tolerance=tolerance).label
