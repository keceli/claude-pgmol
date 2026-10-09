"""Pure geometric helpers shared by the symmetry search.

All routines operate on plain ``(natoms, 3)`` float arrays in ångströms and
never mutate their inputs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "IDENTITY",
    "INVERSION",
    "StructureMatcher",
    "candidate_directions",
    "centered",
    "centroid",
    "element_blocks",
    "fit_orthogonal",
    "improper_rotation_matrix",
    "is_linear",
    "minimum_equivalent_separation",
    "principal_axes",
    "reflection_matrix",
    "rotation_matrix",
]

IDENTITY = np.eye(3)
INVERSION = -np.eye(3)

#: Directions shorter than this (after normalising inputs) are discarded as
#: numerically meaningless rather than normalised into noise.
_DIRECTION_EPS = 1e-9


def centroid(coordinates: np.ndarray) -> np.ndarray:
    """Return the unweighted centroid of ``coordinates``.

    Every symmetry operation of a molecule permutes atoms of the same element,
    so it leaves the sum of all position vectors -- hence the centroid --
    invariant.  The centroid is therefore a fixed point of the full point
    group and a valid origin for the symmetry search, without needing atomic
    masses.
    """
    return np.asarray(coordinates, dtype=float).mean(axis=0)


def centered(coordinates: np.ndarray) -> np.ndarray:
    """Return a copy of ``coordinates`` translated so its centroid is at the origin."""
    coords = np.asarray(coordinates, dtype=float)
    return coords - coords.mean(axis=0)


def is_linear(coordinates: np.ndarray, tolerance: float) -> bool:
    """Return ``True`` when every atom lies within ``tolerance`` of one line.

    ``coordinates`` must already be centred.  A single atom counts as linear
    for the caller's purposes but is handled separately by the classifier.
    """
    coords = np.asarray(coordinates, dtype=float)
    if len(coords) < 3:
        return True
    axis = principal_axes(coords)[0]
    perpendicular = coords - np.outer(coords @ axis, axis)
    return bool(np.linalg.norm(perpendicular, axis=1).max() <= tolerance)


def principal_axes(coordinates: np.ndarray) -> np.ndarray:
    """Return three orthonormal axes ordered by decreasing spatial extent.

    These are the right singular vectors of the centred coordinate matrix,
    i.e. the principal axes of the unweighted gyration tensor.  Row ``0`` is
    the direction of greatest extent.
    """
    coords = np.asarray(coordinates, dtype=float)
    coords = coords - coords.mean(axis=0)
    # Degenerate singular values make individual vectors arbitrary, but the
    # set is still a useful source of candidate symmetry axes.
    _, _, vt = np.linalg.svd(coords, full_matrices=True)
    return np.asarray(vt, dtype=float)


def element_blocks(symbols: Sequence[str]) -> dict[str, np.ndarray]:
    """Group atom indices by element symbol, preserving first-seen order."""
    blocks: dict[str, list[int]] = {}
    for index, symbol in enumerate(symbols):
        blocks.setdefault(symbol, []).append(index)
    return {symbol: np.asarray(idx, dtype=int) for symbol, idx in blocks.items()}


def minimum_equivalent_separation(
    coordinates: np.ndarray, blocks: dict[str, np.ndarray]
) -> float:
    """Shortest distance between two atoms of the same element.

    Returns ``inf`` when no element occurs twice.  Atom-to-atom matching is
    only well posed while twice the tolerance stays below this distance: a
    transformed atom can then be within tolerance of at most one partner.
    """
    shortest = math.inf
    for indices in blocks.values():
        if len(indices) < 2:
            continue
        block = coordinates[indices]
        squared = ((block[:, None, :] - block[None, :, :]) ** 2).sum(axis=-1)
        np.fill_diagonal(squared, np.inf)
        shortest = min(shortest, math.sqrt(float(squared.min())))
    return shortest


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms


def _canonical_sign(directions: np.ndarray) -> np.ndarray:
    """Flip each row so its largest-magnitude component is positive."""
    leading = np.argmax(np.abs(directions), axis=1)
    signs = np.sign(directions[np.arange(len(directions)), leading])
    signs[signs == 0.0] = 1.0
    return directions * signs[:, None]


def deduplicate_directions(
    directions: Iterable[Sequence[float]], angular_tolerance: float
) -> np.ndarray:
    """Collapse a list of direction vectors to distinct axes.

    Vectors are compared as *axes*: ``u`` and ``-u`` are the same direction.
    Two directions closer than ``angular_tolerance`` radians are merged.

    A uniform grid over the unit sphere keeps this linear in the number of
    inputs; without it the quadratic all-pairs comparison dominates the whole
    analysis for molecules with many equivalent atoms.
    """
    stacked = np.asarray(list(directions), dtype=float).reshape(-1, 3)
    if len(stacked) == 0:
        return np.zeros((0, 3))
    stacked = stacked[np.linalg.norm(stacked, axis=1) > _DIRECTION_EPS]
    if len(stacked) == 0:
        return np.zeros((0, 3))
    stacked = _canonical_sign(_normalize(stacked))

    cell_size = max(angular_tolerance, 1e-9)
    cos_cut = np.cos(cell_size)
    cells = np.floor(stacked / cell_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    unique: list[np.ndarray] = []
    neighbourhood = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for direction, cell in zip(stacked, cells, strict=True):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        for offset in neighbourhood:
            neighbours = buckets.get(
                (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            )
            if neighbours is not None and any(
                abs(float(np.dot(other, direction))) >= cos_cut for other in neighbours
            ):
                break
        else:
            buckets.setdefault(key, []).append(direction)
            unique.append(direction)
    return np.asarray(unique) if unique else np.zeros((0, 3))


def _reference_atoms(
    coordinates: np.ndarray, blocks: dict[str, np.ndarray], count: int = 3
) -> np.ndarray:
    """Pick a few atoms that constrain any symmetry axis, deterministically.

    The element with the fewest off-centre atoms is used, because the work
    done per reference atom grows with the size of its element class.  Atoms
    are ordered by decreasing distance from the centroid, with their rounded
    coordinates breaking ties, so the choice never depends on the order of
    the XYZ lines.
    """
    radii = np.linalg.norm(coordinates, axis=1)
    smallest: np.ndarray | None = None
    for indices in blocks.values():
        off_centre = indices[radii[indices] > _DIRECTION_EPS]
        if len(off_centre) < 2:
            continue
        if smallest is None or len(off_centre) < len(smallest):
            smallest = off_centre
    if smallest is None:
        return np.zeros(0, dtype=int)
    order = sorted(
        smallest,
        key=lambda i: (-round(float(radii[i]), 9), *np.round(coordinates[i], 9)),
    )
    return np.asarray(order[:count], dtype=int)


def _axes_through_no_atom(
    coordinates: np.ndarray,
    blocks: dict[str, np.ndarray],
    tolerance: float,
) -> list[np.ndarray]:
    """Candidate axes derived from how a rotation must displace atoms.

    A rotation about ``u`` moves every atom ``a`` perpendicular to ``u``, so
    ``u`` is perpendicular to ``a - R a``.  Two atoms whose images are both
    unknown therefore pin ``u`` down to ``(a_p - b) x (a_q - c)`` for some
    candidate images ``b`` and ``c``.  These are the only axes that no atom
    and no atom pair points at -- the ``C3`` axes of a chiral tetrahedral
    group and the ``C5`` axes of a chiral icosahedral group, for instance.
    """
    references = _reference_atoms(coordinates, blocks)
    if len(references) < 2:
        return []
    radii = np.linalg.norm(coordinates, axis=1)
    by_atom: dict[int, np.ndarray] = {}
    for atom in references:
        partners = next(idx for idx in blocks.values() if atom in idx)
        plausible = partners[np.abs(radii[partners] - radii[atom]) <= tolerance]
        by_atom[int(atom)] = coordinates[plausible] - coordinates[atom]
    out: list[np.ndarray] = []
    for first, second in combinations([int(a) for a in references], 2):
        # Each atom is always its own candidate image, so neither set is
        # ever empty; the zero vectors it contributes are dropped when the
        # directions are normalised.
        left, right = by_atom[first], by_atom[second]
        out.append(np.cross(left[:, None, :], right[None, :, :]).reshape(-1, 3))
    return out


def candidate_directions(
    coordinates: np.ndarray,
    blocks: dict[str, np.ndarray],
    angular_tolerance: float,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Enumerate directions that could carry a symmetry element.

    ``coordinates`` must be centred.  The returned unit vectors cover

    * every atom direction (axes through an atom),
    * for each pair of atoms of the *same* element, the bisector of their
      directions, their difference, and their common normal -- which supply
      the axes through bond/edge midpoints, the normals of mirror planes that
      swap the pair, and the axes perpendicular to the pair, and
    * the three principal axes of the gyration tensor, and
    * axes that pass through no atom at all, reconstructed from the
      displacements a rotation must produce (see
      :func:`_axes_through_no_atom`).

    Together with closure of the detected operations under composition this
    reaches every symmetry element of the point groups the package reports.
    """
    coords = np.asarray(coordinates, dtype=float)
    radii = np.linalg.norm(coords, axis=1)
    off_center = radii > _DIRECTION_EPS
    units = np.zeros_like(coords)
    units[off_center] = coords[off_center] / radii[off_center, None]

    pool: list[np.ndarray] = [units[off_center]]
    for indices in blocks.values():
        usable = [i for i in indices if off_center[i]]
        if len(usable) < 2:
            continue
        pairs = np.asarray(list(combinations(usable, 2)), dtype=int)
        first = units[pairs[:, 0]]
        second = units[pairs[:, 1]]
        pool.append(first + second)
        pool.append(first - second)
        pool.append(np.cross(first, second))
    pool.append(principal_axes(coords))
    pool.extend(_axes_through_no_atom(coords, blocks, tolerance))

    return deduplicate_directions(np.concatenate(pool, axis=0), angular_tolerance)


def rotation_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Proper rotation by ``angle`` radians about ``axis`` (right-handed)."""
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    x, y, z = unit
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.cos(angle) * IDENTITY
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.outer(unit, unit)
    )


def reflection_matrix(normal: Sequence[float]) -> np.ndarray:
    """Reflection in the plane through the origin with the given ``normal``."""
    unit = np.asarray(normal, dtype=float)
    unit = unit / np.linalg.norm(unit)
    return IDENTITY - 2.0 * np.outer(unit, unit)


def improper_rotation_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rotation by ``angle`` about ``axis`` followed by reflection across it.

    ``improper_rotation_matrix(u, 2*pi/n)`` is the operation ``S_n``.
    """
    return rotation_matrix(axis, angle) @ reflection_matrix(axis)


@dataclass
class StructureMatcher:
    """Test whether an orthogonal matrix maps a centred molecule onto itself.

    A candidate operation is accepted when, element by element, every
    transformed atom lies within ``tolerance`` ångström of a distinct atom of
    the same element -- that is, when the operation induces a permutation of
    each element's atoms.

    ``tolerance`` must be smaller than half the shortest distance between two
    atoms of the same element, otherwise the nearest-neighbour assignment is
    ambiguous and operations are rejected conservatively.
    """

    coordinates: np.ndarray
    blocks: dict[str, np.ndarray]
    tolerance: float

    #: Atoms used for the cheap pre-check that rejects most candidates.
    PROBE_SIZE = 4

    def __post_init__(self) -> None:
        self._blocks = []
        for indices in sorted(self.blocks.values(), key=len):
            block = np.ascontiguousarray(self.coordinates[indices])
            self._blocks.append((indices, block, (block**2).sum(axis=1)))
        self._tolerance_squared = float(self.tolerance) ** 2
        # Most candidate matrices are wrong, and a handful of atoms is enough
        # to see that.  Checking them first turns the common rejection from
        # an all-atom comparison into a few rows of one.
        _, block, squared = self._blocks[0]
        self._probe = np.ascontiguousarray(block[: self.PROBE_SIZE])
        self._probe_block = block.T.copy()
        self._probe_squared = squared

    def permutation(self, matrix: np.ndarray) -> np.ndarray | None:
        """Return the atom permutation induced by ``matrix``, or ``None``.

        ``result[i] == j`` means ``matrix`` carries atom ``i`` onto atom
        ``j``.  ``None`` means ``matrix`` is not a symmetry operation at this
        tolerance.  Element blocks are visited smallest first so that a
        mismatch is usually found after very little work.
        """
        transpose = np.asarray(matrix, dtype=float).T

        probe = self._probe @ transpose
        probe_squared = (
            (probe**2).sum(axis=1)[:, None]
            + self._probe_squared[None, :]
            - 2.0 * (probe @ self._probe_block)
        )
        if probe_squared.min(axis=1).max() > self._tolerance_squared:
            return None

        mapping = np.empty(len(self.coordinates), dtype=int)
        for indices, block, norms in self._blocks:
            moved = block @ transpose
            if len(indices) == 1:
                if float(((moved - block) ** 2).sum()) > self._tolerance_squared:
                    return None
                mapping[indices[0]] = indices[0]
                continue
            # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, and an orthogonal matrix
            # preserves |a|, so one matrix product replaces the full
            # difference array.
            squared = norms[:, None] + norms[None, :] - 2.0 * (moved @ block.T)
            nearest = squared.argmin(axis=1)
            if squared[np.arange(len(indices)), nearest].max() > self._tolerance_squared:
                return None
            if len(np.unique(nearest)) != len(indices):
                return None
            mapping[indices] = indices[nearest]
        return mapping

    def matches(self, matrix: np.ndarray) -> bool:
        """Return ``True`` when ``matrix`` is a symmetry operation of the molecule."""
        return self.permutation(matrix) is not None


def fit_orthogonal(
    coordinates: np.ndarray, permutation: np.ndarray, determinant_sign: int
) -> np.ndarray:
    """Least-squares orthogonal matrix carrying each atom onto its image.

    Given the permutation an approximate operation induces, this returns the
    orthogonal matrix ``Q`` with the requested determinant sign that
    minimises ``sum |Q r_i - r_perm(i)|^2`` (the Kabsch construction).  Every
    candidate matrix that induces the same permutation therefore collapses
    onto one canonical operation, which makes deduplication exact instead of
    tolerance-dependent, and gives the most accurate axis and order the
    geometry supports.

    The determinant sign must be supplied because it is not determined by
    the permutation alone: for a planar molecule the identity permutation is
    induced both by the identity and by reflection in the molecular plane.
    """
    coords = np.asarray(coordinates, dtype=float)
    covariance = coords.T @ coords[permutation]
    left, _, right = np.linalg.svd(covariance)
    correction = determinant_sign * np.sign(np.linalg.det(left) * np.linalg.det(right))
    middle = np.diag([1.0, 1.0, float(correction)])
    return right.T @ middle @ left.T
