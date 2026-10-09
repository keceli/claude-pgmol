"""Independent construction of point-group operation matrices.

Nothing here imports :mod:`pgmol`.  The matrices come from textbook
generators, so a molecule built by applying them to a seed point has a known
point group that the package has to rediscover from coordinates alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

GOLDEN = (1.0 + 5.0**0.5) / 2.0
Z_AXIS = (0.0, 0.0, 1.0)
X_AXIS = (1.0, 0.0, 0.0)

#: Every label the sweep covers: the 32 crystallographic classes plus the
#: non-crystallographic axial and icosahedral ones.
ALL_LABELS: tuple[str, ...] = (
    "C1",
    "Cs",
    "Ci",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C2v",
    "C3v",
    "C4v",
    "C5v",
    "C6v",
    "C8v",
    "C2h",
    "C3h",
    "C4h",
    "C5h",
    "C6h",
    "S4",
    "S6",
    "S8",
    "S10",
    "S12",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D2h",
    "D3h",
    "D4h",
    "D5h",
    "D6h",
    "D7h",
    "D8h",
    "D2d",
    "D3d",
    "D4d",
    "D5d",
    "D6d",
    "T",
    "Th",
    "Td",
    "O",
    "Oh",
    "I",
    "Ih",
)

POLYHEDRAL = frozenset({"T", "Th", "Td", "O", "Oh", "I", "Ih"})


def rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Proper rotation matrix, built independently of the package."""
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    x, y, z = unit
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        math.cos(angle) * np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * np.outer(unit, unit)
    )


def reflection(normal: Sequence[float]) -> np.ndarray:
    """Reflection matrix for the plane with the given normal."""
    unit = np.asarray(normal, dtype=float)
    unit = unit / np.linalg.norm(unit)
    return np.eye(3) - 2.0 * np.outer(unit, unit)


def improper(axis: Sequence[float], angle: float) -> np.ndarray:
    """``S`` operation: rotate about ``axis`` then reflect across it."""
    return rotation(axis, angle) @ reflection(axis)


def _close(generators: Sequence[np.ndarray], cap: int = 256) -> list[np.ndarray]:
    operations = [np.eye(3)]

    def present(matrix: np.ndarray) -> bool:
        return any(np.abs(other - matrix).max() < 1e-7 for other in operations)

    for generator in generators:
        if not present(generator):
            operations.append(generator)
    index = 0
    while index < len(operations):
        for other in list(operations):
            product = operations[index] @ other
            if not present(product):
                operations.append(product)
                if len(operations) > cap:
                    raise RuntimeError("generators do not close to a finite group")
        index += 1
    return operations


def axial_order(label: str) -> int:
    """The ``n`` in ``Cn``/``Dn``/``Sn``; ``1`` for the groups without one."""
    if label in {"C1", "Cs", "Ci"} or label in POLYHEDRAL:
        return 1
    suffix = label[-1] if label[-1].isalpha() else ""
    return int(label[1 : len(label) - len(suffix)])


def operations(label: str) -> list[np.ndarray]:
    """All operation matrices of the point group named by ``label``."""
    if label in POLYHEDRAL:
        if label.startswith("T"):
            generators = [rotation(Z_AXIS, math.pi), rotation((1, 1, 1), 2 * math.pi / 3)]
            if label == "Th":
                generators.append(-np.eye(3))
            if label == "Td":
                generators.append(reflection((1, -1, 0)))
        elif label.startswith("O"):
            generators = [
                rotation(Z_AXIS, math.pi / 2),
                rotation((1, 1, 1), 2 * math.pi / 3),
            ]
            if label == "Oh":
                generators.append(-np.eye(3))
        else:
            generators = [
                rotation((0, 1, GOLDEN), 2 * math.pi / 5),
                rotation(Z_AXIS, math.pi),
            ]
            if label == "Ih":
                generators.append(-np.eye(3))
        return _close(generators)

    if label == "C1":
        return [np.eye(3)]
    if label == "Cs":
        return _close([reflection(Z_AXIS)])
    if label == "Ci":
        return _close([-np.eye(3)])

    suffix = label[-1] if label[-1].isalpha() else ""
    order = axial_order(label)
    if label[0] == "S":
        return _close([improper(Z_AXIS, 2 * math.pi / order)])
    if label[0] == "C":
        generators = [rotation(Z_AXIS, 2 * math.pi / order)]
        if suffix == "v":
            generators.append(reflection((0, 1, 0)))
        elif suffix == "h":
            generators.append(reflection(Z_AXIS))
        return _close(generators)
    if label[0] == "D":
        if suffix == "d":
            generators = [improper(Z_AXIS, math.pi / order), rotation(X_AXIS, math.pi)]
        else:
            generators = [
                rotation(Z_AXIS, 2 * math.pi / order),
                rotation(X_AXIS, math.pi),
            ]
            if suffix == "h":
                generators.append(reflection(Z_AXIS))
        return _close(generators)
    raise ValueError(f"unsupported label {label!r}")


# Seeds in cylindrical coordinates (radius, azimuth as a fraction of the
# n-fold sector, height).  The generated groups put their mirrors and C2'
# axes at fractions 0, 1/4, 1/2 and 3/4, and an accidental extra vertical
# mirror would need the two orbits to differ by a multiple of 1/2, so 0.13
# and 0.38 avoid both.  The heights differ and are never zero, which rules
# out an accidental horizontal mirror, S_2n axis or perpendicular C2.
_SEEDS = (
    (1.37, 0.13, 0.61),
    (0.94, 0.38, -1.13),
    (1.71, 0.61, 0.29),
    (1.22, 0.87, -0.47),
)
#: Direction used for the polyhedral groups; verified to be in general
#: position with respect to both Oh and Ih by :func:`check_general_position`.
_POLYHEDRAL_DIRECTION = np.array([0.3137, 0.5219, 0.7933])
_POLYHEDRAL_DIRECTION = _POLYHEDRAL_DIRECTION / np.linalg.norm(_POLYHEDRAL_DIRECTION)

_ELEMENTS = ("C", "N", "O", "F")


def orbit(matrices: Sequence[np.ndarray], seed: np.ndarray) -> list[np.ndarray]:
    """Distinct images of ``seed`` under ``matrices``."""
    points: list[np.ndarray] = []
    for matrix in matrices:
        image = matrix @ seed
        if not any(np.linalg.norm(image - other) < 1e-9 for other in points):
            points.append(image)
    return points


def seed_count(label: str) -> int:
    """How many orbits to superpose so the result has exactly this group.

    A polyhedral orbit already pins the group down.  ``C1``, ``Cs`` and
    ``Ci`` have tiny orbits, so they need four to avoid the accidental
    symmetry that any three or fewer points always have.
    """
    if label in POLYHEDRAL:
        return 1
    if label in {"C1", "Cs", "Ci"}:
        return 4
    return 2


def build_molecule(label: str) -> tuple[list[str], np.ndarray]:
    """Return ``(symbols, coordinates)`` for a molecule with point group ``label``.

    Raises
    ------
    AssertionError
        If a seed turns out not to be in general position, which would make
        the generated molecule more symmetric than requested.
    """
    matrices = operations(label)
    sector = 2.0 * math.pi / axial_order(label)
    symbols: list[str] = []
    points: list[np.ndarray] = []
    for index in range(seed_count(label)):
        if label in POLYHEDRAL:
            seed = (1.4 + 0.3 * index) * _POLYHEDRAL_DIRECTION
        else:
            radius, fraction, height = _SEEDS[index]
            azimuth = fraction * sector
            seed = np.array([radius * math.cos(azimuth), radius * math.sin(azimuth), height])
        images = orbit(matrices, seed)
        assert len(images) == len(matrices), (
            f"{label}: seed {index} is not in general position "
            f"({len(images)} images for a group of order {len(matrices)})"
        )
        symbols.extend([_ELEMENTS[index]] * len(images))
        points.extend(images)
    return symbols, np.asarray(points)


def check_general_position() -> None:
    """Verify the polyhedral seeds lie on no symmetry element of Oh or Ih.

    This is the non-circular part of the construction: it uses only the
    reference matrices, never the package under test.
    """
    for supergroup, expected in (("Oh", 48), ("Ih", 120)):
        matrices = operations(supergroup)
        for index in range(2):
            seed = (1.4 + 0.3 * index) * _POLYHEDRAL_DIRECTION
            found = len(orbit(matrices, seed))
            assert found == expected, (
                f"polyhedral seed {index} is special in {supergroup}: "
                f"{found} images, expected {expected}"
            )


def minimum_equivalent_distance(symbols: Sequence[str], coordinates: np.ndarray) -> float:
    """Shortest distance between two atoms of the same element."""
    labels = np.asarray(symbols)
    shortest = math.inf
    for element in set(symbols):
        block = coordinates[labels == element]
        if len(block) < 2:
            continue
        distances = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        shortest = min(shortest, float(distances.min()))
    return shortest
