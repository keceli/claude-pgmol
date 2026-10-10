"""Detection of the symmetry operations of a finite molecular geometry."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .geometry import (
    IDENTITY,
    INVERSION,
    FloatArray,
    IntArray,
    StructureMatcher,
    candidate_directions,
    element_blocks,
    fit_orthogonal,
    improper_rotation_matrix,
    reflection_matrix,
    rotation_matrix,
)

__all__ = [
    "SymmetryGroup",
    "SymmetryOperation",
    "accept_operation",
    "detect_operations",
    "identify_operation",
]

#: Hard ceiling on the number of operations collected, as a guard against a
#: pathological input making closure run away.  The largest point group this
#: package reports (``Ih``) has order 120; ``D_nh`` of a large ring is the
#: only realistic way to exceed that and still needs far fewer than this.
_MAX_GROUP_ORDER = 2048

#: Largest matrix power tried when determining the order of an operation.
_MAX_OPERATION_ORDER = 256

PROPER = "proper"
IMPROPER = "improper"


@dataclass(frozen=True, eq=False)
class SymmetryOperation:
    """One orthogonal operation that maps the molecule onto itself.

    Attributes
    ----------
    matrix:
        The ``3x3`` orthogonal matrix, acting on column vectors about the
        molecular centroid.
    kind:
        One of ``"E"``, ``"Cn"``, ``"i"``, ``"sigma"`` or ``"Sn"``.
    order:
        Smallest positive ``k`` with ``matrix ** k == I``.
    axis:
        Unit rotation axis for ``"Cn"``/``"Sn"``, the plane normal for
        ``"sigma"``, and ``None`` for ``"E"`` and ``"i"``.
    """

    matrix: FloatArray
    kind: str
    order: int
    axis: FloatArray | None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymmetryOperation):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.order == other.order
            and bool(np.array_equal(self.matrix, other.matrix))
        )

    def __hash__(self) -> int:
        # Adding zero folds -0.0 onto 0.0, which array_equal treats as equal
        # but whose raw bytes differ.
        return hash((self.kind, self.order, (self.matrix + 0.0).tobytes()))

    @property
    def label(self) -> str:
        """Short Schoenflies-style name such as ``"C3"``, ``"S4"`` or ``"i"``.

        The index is the ``n`` of ``C_n``/``S_n``, which is *not* always the
        matrix order: ``S_n`` for odd ``n`` returns to the identity only
        after ``2n`` applications, so ``S3`` has matrix order 6.  Naming it
        from the rotation angle keeps ``D3h`` reporting ``S3`` rather than a
        nonexistent ``S6``, and keeps the genuinely distinct ``S3`` and
        ``S6`` of ``D6h`` apart.
        """
        if self.kind in {"E", "i", "sigma"}:
            return self.kind
        if self.kind == "Cn":
            return f"C{self.order}"
        return f"S{_schoenflies_index(self.rotation_angle, self.order)}"

    @property
    def rotation_angle(self) -> float:
        """Rotation angle in radians, about :attr:`axis`, in ``[0, pi]``."""
        trace = float(np.trace(self.matrix))
        offset = -1.0 if self.is_proper else 1.0
        return math.acos(min(1.0, max(-1.0, (trace + offset) / 2.0)))

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def is_proper(self) -> bool:
        return self.kind in {"E", "Cn"}


def _schoenflies_index(angle: float, max_index: int, atol: float = 1e-6) -> int:
    """Smallest ``n`` with ``angle`` an exact multiple of ``2*pi/n``.

    This is the denominator of ``angle / 2*pi`` in lowest terms, which is the
    order of the axis the operation belongs to.
    """
    ratio = angle / (2.0 * math.pi)
    for index in range(1, max(max_index, 1) + 1):
        if abs(index * ratio - round(index * ratio)) < atol:
            return index
    return max_index


def _axis_from_skew(matrix: FloatArray) -> FloatArray:
    vector = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def _axis_from_projector(projector: FloatArray) -> FloatArray:
    """Extract the unit vector ``u`` from a rank-one projector ``u u^T``."""
    column = int(np.argmax(np.linalg.norm(projector, axis=0)))
    vector = projector[:, column]
    return vector / np.linalg.norm(vector)


def _matrix_order(matrix: FloatArray, atol: float, limit: int) -> int | None:
    """Smallest ``k <= limit`` with ``matrix ** k == I``, or ``None``.

    Every point-group operation has finite order, so ``None`` identifies a
    matrix that merely happens to map the atoms onto each other within the
    tolerance -- an accidental near-symmetry, not a symmetry.
    """
    power = np.array(matrix, dtype=float)
    for order in range(1, max(limit, 1) + 1):
        if np.max(np.abs(power - IDENTITY)) <= atol:
            return order
        power = power @ matrix
    return None


def _identify_or_none(matrix: FloatArray, atol: float, limit: int) -> SymmetryOperation | None:
    """Like :func:`identify_operation` but returns ``None`` for non-operations."""
    try:
        return identify_operation(matrix, atol=atol, max_order=limit)
    except ValueError:
        return None


def identify_operation(
    matrix: FloatArray, atol: float = 1e-4, max_order: int = _MAX_OPERATION_ORDER
) -> SymmetryOperation:
    """Classify an orthogonal ``3x3`` matrix as a point-group operation.

    Parameters
    ----------
    matrix:
        An orthogonal matrix with determinant ``+/-1``.
    atol:
        Absolute tolerance on matrix entries when recognising special forms
        and when deciding that a matrix power has returned to the identity.
    max_order:
        Largest operation order considered.  A matrix that does not return
        to the identity within this many powers is rejected.

    Raises
    ------
    ValueError
        If the matrix is not orthogonal or has no order at or below
        ``max_order``, and so is not a point-group operation.
    """
    array = np.array(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got shape {array.shape}")
    determinant = float(np.linalg.det(array))
    if not math.isclose(abs(determinant), 1.0, abs_tol=1e-3):
        raise ValueError(f"matrix is not orthogonal (determinant {determinant:.6g})")

    order = _matrix_order(array, atol, max_order)
    if order is None:
        raise ValueError(
            "matrix has no finite order and is therefore not a point-group operation"
        )
    trace = float(np.trace(array))

    if determinant > 0.0:
        if np.max(np.abs(array - IDENTITY)) <= atol:
            return SymmetryOperation(array, "E", 1, None)
        angle = math.acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0)))
        if math.sin(angle) > atol:
            axis = _axis_from_skew(array)
        else:  # angle == pi: a C2 rotation, (R + I)/2 is the axis projector
            axis = _axis_from_projector((array + IDENTITY) / 2.0)
        return SymmetryOperation(array, "Cn", order, axis)

    if np.max(np.abs(array - INVERSION)) <= atol:
        return SymmetryOperation(array, "i", 2, None)
    angle = math.acos(min(1.0, max(-1.0, (trace + 1.0) / 2.0)))
    if math.sin(angle) <= atol:  # angle == 0: a pure reflection
        normal = _axis_from_projector((IDENTITY - array) / 2.0)
        return SymmetryOperation(array, "sigma", 2, normal)
    return SymmetryOperation(array, "Sn", order, _axis_from_skew(array))


@dataclass
class SymmetryGroup:
    """The set of symmetry operations found for one molecule.

    ``closed`` records whether the operations are closed under composition.
    A false value means the geometry is only approximately symmetric at the
    requested tolerance: some product of accepted operations is not itself
    accepted.
    """

    operations: tuple[SymmetryOperation, ...]
    closed: bool = True
    angular_tolerance: float = 1e-3

    @property
    def order(self) -> int:
        """Number of operations in the group."""
        return len(self.operations)

    def __iter__(self) -> Iterator[SymmetryOperation]:
        return iter(self.operations)

    def of_kind(self, kind: str) -> list[SymmetryOperation]:
        """All operations of the given ``kind``."""
        return [op for op in self.operations if op.kind == kind]

    @property
    def has_inversion(self) -> bool:
        return any(op.kind == "i" for op in self.operations)

    @property
    def mirror_normals(self) -> list[FloatArray]:
        return [op.axis for op in self.of_kind("sigma") if op.axis is not None]

    def rotation_axes(self) -> list[tuple[FloatArray, int]]:
        """Distinct proper-rotation axes paired with their highest order.

        Axes are compared up to sign and merged within the group's angular
        tolerance.
        """
        return _collect_axes(self.of_kind("Cn"), self.angular_tolerance)

    def improper_axes(self) -> list[tuple[FloatArray, int]]:
        """Distinct ``S_n`` axes paired with the highest operation order."""
        return _collect_axes(self.of_kind("Sn"), self.angular_tolerance)

    def counts(self) -> dict[str, int]:
        """Number of operations of each kind, for diagnostics and tests."""
        tally: dict[str, int] = {}
        for op in self.operations:
            tally[op.label] = tally.get(op.label, 0) + 1
        return dict(sorted(tally.items()))


def _collect_axes(
    operations: Sequence[SymmetryOperation], angular_tolerance: float
) -> list[tuple[FloatArray, int]]:
    """Merge operations sharing an axis, keeping the highest order on each.

    Every operation passed in must carry an axis, which is true of the
    ``"Cn"`` and ``"Sn"`` kinds this is called with.  Axes are compared up
    to sign, so ``u`` and ``-u`` count once.
    """
    cos_cut = math.cos(angular_tolerance)
    axes: list[tuple[FloatArray, int]] = []
    for op in operations:
        # Only "Cn" and "Sn" reach here, and both always carry an axis.
        axis = cast(FloatArray, op.axis)
        for index, (known, order) in enumerate(axes):
            if abs(float(np.dot(known, axis))) >= cos_cut:
                axes[index] = (known, max(order, op.order))
                break
        else:
            axes.append((np.asarray(axis, dtype=np.float64), op.order))
    return axes


def accept_operation(
    coordinates: FloatArray,
    matcher: StructureMatcher,
    permutation: IntArray,
    sign: int,
    *,
    identify_atol: float,
    order_limit: int,
    proper_limit: int,
) -> SymmetryOperation | None:
    """Canonicalise a candidate permutation into an operation, or reject it.

    The candidate is replaced by the least-squares orthogonal matrix for its
    permutation, which makes the result independent of whichever approximate
    matrix suggested it.  It is then rejected unless

    * the canonical matrix still induces exactly that permutation,
    * it has a finite order no greater than ``order_limit``, and
    * if it is a proper rotation, an order no greater than ``proper_limit``.

    Each check removes an accidental near-coincidence that a tolerant atom
    match would otherwise let through as a symmetry operation.
    """
    matrix = fit_orthogonal(coordinates, permutation, sign)
    refit = matcher.permutation(matrix)
    if refit is None or not np.array_equal(refit, permutation):
        return None
    operation = _identify_or_none(matrix, identify_atol, order_limit)
    if operation is None:
        return None
    if operation.is_proper and operation.order > proper_limit:
        return None
    return operation


def detect_operations(
    symbols: Sequence[str],
    coordinates: FloatArray,
    tolerance: float,
) -> SymmetryGroup:
    """Find every symmetry operation of a centred, non-linear molecule.

    Candidate axes are generated from the geometry (see
    :func:`pgmol.geometry.candidate_directions`), every plausible rotation,
    improper rotation and reflection about them is tested against the atom
    positions, and the accepted operations are then closed under composition
    so that elements no candidate direction pointed at are still recovered.

    Parameters
    ----------
    symbols:
        Element symbol per atom.
    coordinates:
        ``(natoms, 3)`` array already centred on the molecular centroid.
    tolerance:
        Maximum Cartesian displacement, in ångströms, between a transformed
        atom and the atom it is matched with.
    """
    coords = np.asarray(coordinates, dtype=float)
    blocks = element_blocks(symbols)
    matcher = StructureMatcher(coords, blocks, tolerance)

    norms_squared = (coords**2).sum(axis=1)
    radius = float(math.sqrt(np.max(norms_squared)))
    angular_tolerance = _angular_tolerance(tolerance, radius)
    directions = candidate_directions(coords, blocks, min(1e-3, angular_tolerance), tolerance)

    # A rotation of order n carries every off-axis atom round an orbit of n
    # atoms of one element, so n cannot exceed the largest element count; an
    # improper operation has even order whose proper half obeys the same
    # bound.  Anything above that is an accidental near-symmetry.
    largest_class = max((len(idx) for idx in blocks.values()), default=1)
    proper_limit = largest_class
    order_limit = 2 * largest_class
    identify_atol = max(1e-4, angular_tolerance)

    operations: list[SymmetryOperation] = []
    permutations: list[IntArray] = []
    signs: list[int] = []
    # An operation is identified exactly by the permutation it induces
    # together with its determinant sign, so this needs no tolerance.
    verdicts: dict[tuple[int, ...], bool] = {}

    def register(permutation: IntArray, sign: int) -> bool:
        """Accept the operation with this permutation; report acceptance."""
        key = (sign, *(int(value) for value in permutation))
        cached = verdicts.get(key)
        if cached is not None:
            return cached
        operation = accept_operation(
            coords,
            matcher,
            permutation,
            sign,
            identify_atol=identify_atol,
            order_limit=order_limit,
            proper_limit=proper_limit,
        )
        if operation is not None:
            operations.append(operation)
            permutations.append(np.asarray(permutation, dtype=int))
            signs.append(sign)
        verdicts[key] = operation is not None
        return operation is not None

    def consider(matrix: FloatArray) -> bool:
        permutation = matcher.permutation(matrix)
        if permutation is None:
            return False
        sign = 1 if float(np.linalg.det(matrix)) > 0.0 else -1
        return register(permutation, sign)

    consider(IDENTITY)
    consider(INVERSION)
    for direction in directions:
        consider(reflection_matrix(direction))
        for matrix in _axis_candidates(coords, blocks, direction, tolerance, norms_squared):
            consider(matrix)

    closed = _close_under_composition(operations, permutations, signs, register)
    return SymmetryGroup(
        operations=tuple(operations),
        closed=closed,
        angular_tolerance=angular_tolerance,
    )


def _close_under_composition(
    operations: list[SymmetryOperation],
    permutations: list[IntArray],
    signs: list[int],
    register: Callable[[IntArray, int], bool],
) -> bool:
    """Extend the operation list with products; report whether the set closed.

    Composing two operations is pure index arithmetic on their permutations
    -- ``(A B)(i) = A(B(i))`` -- with determinant signs multiplying, so no
    geometry is touched until a genuinely new operation turns up.  Passes
    repeat until nothing new appears, which makes the result independent of
    the order candidates were found in.
    """
    closed = True
    growing = True
    while growing:
        growing = False
        for left in range(len(operations)):
            for right in range(len(operations)):
                if len(operations) >= _MAX_GROUP_ORDER:
                    return False
                composed = permutations[left][permutations[right]]
                before = len(operations)
                if not register(composed, signs[left] * signs[right]):
                    # The product of two accepted operations is not itself
                    # accepted, so the set is not a group: the geometry is
                    # only approximately symmetric at this tolerance.
                    closed = False
                elif len(operations) > before:
                    growing = True
    return closed


def _angular_tolerance(tolerance: float, radius: float) -> float:
    """Angle, in radians, subtended by ``tolerance`` at the molecular radius."""
    if radius <= 0.0:
        return 1e-3
    return float(min(0.2, max(1e-6, tolerance / radius)))


def _axis_candidates(
    coordinates: FloatArray,
    blocks: dict[str, IntArray],
    axis: FloatArray,
    tolerance: float,
    norms_squared: FloatArray,
) -> Iterator[FloatArray]:
    """Yield every rotation and improper rotation about ``axis`` worth testing.

    Rather than guessing rotation orders, the angles are read off the
    geometry.  A symmetry operation about ``axis`` must send a chosen
    reference atom to an atom of the same element at the same height along
    the axis (same height with the sign flipped, for an improper rotation)
    and the same distance from it; each such atom fixes one rotation angle
    exactly.  This makes the search complete without any bound on the
    rotation order, and linear rather than quadratic in the atom count.
    """
    height = coordinates @ axis
    # |r|^2 - (r.u)^2 is the squared distance from the axis; computing it
    # this way avoids materialising the perpendicular components of every
    # atom for every one of the many candidate axes.
    radius_squared = np.maximum(norms_squared - height**2, 0.0)
    tolerance_squared = tolerance * tolerance

    # Every element is used as the reference in turn.  Picking only the
    # element with the fewest off-axis atoms was cheaper, but which element
    # won depended on XYZ line order whenever two tied, and the radius and
    # height screens below can reject a genuine partner for one element
    # while accepting it for another -- so the same atoms in a different
    # order could give a different point group.
    for indices in sorted(blocks.items()):
        partners = indices[1]
        off_axis = partners[radius_squared[partners] > tolerance_squared]
        if len(off_axis) == 0:
            continue  # this element lies entirely on the axis
        reference = int(off_axis[np.argmax(radius_squared[off_axis])])
        reference_radius = math.sqrt(radius_squared[reference])
        first = coordinates[reference] - height[reference] * axis
        first = first / reference_radius
        second = np.cross(axis, first)
        # Both basis vectors are perpendicular to the axis, so projecting
        # the raw coordinates onto them gives the same result as projecting
        # the perpendicular components, for two products in total.
        along_first = coordinates @ first
        along_second = coordinates @ second

        for index in partners:
            if abs(math.sqrt(radius_squared[index]) - reference_radius) > tolerance:
                continue
            angle = math.atan2(float(along_second[index]), float(along_first[index]))
            if abs(height[index] - height[reference]) <= tolerance:
                yield rotation_matrix(axis, angle)
            if abs(height[index] + height[reference]) <= tolerance:
                yield improper_rotation_matrix(axis, angle)
