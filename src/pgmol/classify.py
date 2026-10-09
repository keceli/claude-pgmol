"""Turn a detected set of symmetry operations into a Schoenflies label."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .symmetry import SymmetryGroup

__all__ = ["SUPPORTED_LABELS", "classify_group", "expected_order"]

#: Labels the classifier can emit for the special cases it handles outside the
#: finite-group flow chart.  Axial labels (``C3v``, ``D5h``, ...) are built from
#: the detected axis order and so are not enumerable in advance.
SUPPORTED_LABELS = (
    "Kh",
    "Cinfv",
    "Dinfh",
    "C1",
    "Cs",
    "Ci",
    "T",
    "Th",
    "Td",
    "O",
    "Oh",
    "I",
    "Ih",
)


def _parallel(first: np.ndarray, second: np.ndarray, cos_cut: float) -> bool:
    return abs(float(np.dot(first, second))) >= cos_cut


def _perpendicular(first: np.ndarray, second: np.ndarray, sin_cut: float) -> bool:
    return abs(float(np.dot(first, second))) <= sin_cut


def _select_principal_axis(
    axes: Sequence[tuple[np.ndarray, int]],
    group: SymmetryGroup,
    cos_cut: float,
    sin_cut: float,
) -> tuple[np.ndarray, int]:
    """Pick the principal axis among those of maximal rotation order.

    ``D2d`` is the case that matters: all three of its axes are ``C2`` axes
    and only the one that also carries ``S4`` is the principal axis.  Ties are
    broken on the number of perpendicular ``C2`` axes, then on the number of
    mirror planes containing the axis, then lexicographically on the axis
    components so the result never depends on atom ordering.
    """
    highest = max(order for _, order in axes)
    contenders = [(axis, order) for axis, order in axes if order == highest]
    if len(contenders) == 1:
        return contenders[0]

    improper = group.improper_axes()
    mirrors = group.mirror_normals

    def sort_key(entry: tuple[np.ndarray, int]) -> tuple:
        axis = entry[0]
        improper_order = max(
            (order for other, order in improper if _parallel(axis, other, cos_cut)),
            default=0,
        )
        perpendicular_axes = sum(
            1
            for other, _ in axes
            if not _parallel(axis, other, cos_cut) and _perpendicular(axis, other, sin_cut)
        )
        containing_planes = sum(
            1 for normal in mirrors if _perpendicular(axis, normal, sin_cut)
        )
        rounded = tuple(np.round(np.abs(axis), 6))
        return (-improper_order, -perpendicular_axes, -containing_planes, rounded)

    return min(contenders, key=sort_key)


def classify_group(group: SymmetryGroup) -> str:
    """Return the Schoenflies label of a finite, non-linear point group.

    The decision follows the standard flow chart: cubic and icosahedral
    groups are recognised by having several high-order axes, then the
    principal axis fixes ``n`` and the presence of ``n`` perpendicular ``C2``
    axes, a horizontal mirror, ``n`` vertical mirrors and ``S_2n`` selects
    among the ``D`` and ``C`` families.
    """
    angular_tolerance = max(group.angular_tolerance, 1e-6)
    cos_cut = math.cos(angular_tolerance)
    sin_cut = math.sin(angular_tolerance)

    axes = group.rotation_axes()
    mirrors = group.mirror_normals
    has_inversion = group.has_inversion

    high_order = [order for _, order in axes if order >= 3]
    if len(high_order) >= 2:
        return _classify_polyhedral(axes, has_inversion, bool(mirrors))

    if not axes:
        if has_inversion:
            return "Ci"
        if mirrors:
            return "Cs"
        return "C1"

    principal, n = _select_principal_axis(axes, group, cos_cut, sin_cut)

    perpendicular_c2 = sum(
        1
        for axis, _ in axes
        if not _parallel(principal, axis, cos_cut) and _perpendicular(principal, axis, sin_cut)
    )
    horizontal_mirror = any(_parallel(principal, nrm, cos_cut) for nrm in mirrors)
    vertical_mirrors = sum(1 for nrm in mirrors if _perpendicular(principal, nrm, sin_cut))
    improper_on_axis = any(
        _parallel(principal, axis, cos_cut) for axis, _ in group.improper_axes()
    )

    if perpendicular_c2 >= n:
        if horizontal_mirror:
            return f"D{n}h"
        if vertical_mirrors >= n:
            return f"D{n}d"
        return f"D{n}"
    if horizontal_mirror:
        return f"C{n}h"
    if vertical_mirrors >= n:
        return f"C{n}v"
    if improper_on_axis:
        return f"S{2 * n}"
    return f"C{n}"


def _classify_polyhedral(
    axes: Sequence[tuple[np.ndarray, int]], has_inversion: bool, has_mirror: bool
) -> str:
    """Name a group with more than one axis of order three or higher."""
    if sum(1 for _, order in axes if order == 5) >= 6:
        return "Ih" if has_inversion else "I"
    if sum(1 for _, order in axes if order == 4) >= 3:
        return "Oh" if has_inversion else "O"
    if has_inversion:
        return "Th"
    if has_mirror:
        return "Td"
    return "T"


def expected_order(label: str) -> int | None:
    """Group order implied by a Schoenflies ``label``, or ``None`` if infinite.

    Used by the test-suite and by :func:`pgmol.api.analyze` as an independent
    consistency check on the number of operations actually found.
    """
    fixed = {
        "C1": 1,
        "Cs": 2,
        "Ci": 2,
        "T": 12,
        "Td": 24,
        "Th": 24,
        "O": 24,
        "Oh": 48,
        "I": 60,
        "Ih": 120,
    }
    if label in fixed:
        return fixed[label]
    if label in {"Kh", "Cinfv", "Dinfh"}:
        return None
    family = label[0]
    suffix = label[-1] if label[-1].isalpha() else ""
    digits = label[1 : len(label) - len(suffix)]
    if not digits.isdigit():
        raise ValueError(f"unrecognised Schoenflies label {label!r}")
    n = int(digits)
    multipliers = {
        ("S", ""): 1,
        ("C", ""): 1,
        ("C", "v"): 2,
        ("C", "h"): 2,
        ("D", ""): 2,
        ("D", "h"): 4,
        ("D", "d"): 4,
    }
    multiplier = multipliers.get((family, suffix))
    if multiplier is None or n < 1:
        raise ValueError(f"unrecognised Schoenflies label {label!r}")
    return multiplier * n
