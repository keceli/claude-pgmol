"""Reading and validating plain XYZ molecular geometry files.

Supported dialect
-----------------
A file holds exactly one frame::

    <natoms>
    <comment>
    <symbol> <x> <y> <z>
    ... (natoms lines total)

* Line 1 is the atom count; leading/trailing whitespace is ignored, blank
  lines *before* it are skipped, and a leading UTF-8 byte-order mark is
  ignored.
* Line 2 is a free-form comment and may be empty.
* Each atom line holds exactly four whitespace-separated fields.  Element
  symbols are case-insensitive (``cl``, ``CL`` and ``Cl`` all parse) and a
  bare atomic number (``17``) is also accepted; both are canonicalised to the
  IUPAC symbol.  Coordinates are ångströms and must be finite.
* Trailing blank lines are ignored.  Anything else after the atom block --
  including a second frame of a trajectory -- is an error.

Everything else (extra columns, multi-frame files, extended-XYZ key/value
comment lines interpreted as data, ``*`` or ``NaN`` coordinates) is rejected
with :class:`XYZFormatError` rather than silently reinterpreted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elements import canonical_symbol
from .errors import XYZFormatError
from .geometry import FloatArray, _grid_key

__all__ = ["Molecule", "parse_xyz", "read_xyz"]

#: Two atoms closer than this (ångström) are treated as a corrupt geometry.
_COINCIDENT_ATOM_DISTANCE = 1e-5

#: Largest coordinate magnitude accepted.  Far beyond any real structure,
#: but low enough that squaring and accumulating coordinates stays well
#: inside the float64 range: past roughly 1e150 the SVDs in the symmetry
#: search stop converging and the grid below overflows to infinity.
_MAX_COORDINATE = 1e100


@dataclass(frozen=True, eq=False)
class Molecule:
    """An immutable set of labelled Cartesian atomic positions.

    Attributes
    ----------
    symbols:
        Element symbols, one per atom, in input order.  They are validated
        and canonicalised on construction, so ``"h"``, ``"H"`` and ``"1"``
        all become ``"H"`` and an unknown symbol is rejected -- building a
        molecule directly cannot smuggle in an element the rest of the
        package would treat as distinct.
    coordinates:
        Read-only ``(natoms, 3)`` float64 array of ångström coordinates.
    comment:
        The XYZ comment line, verbatim (without its newline).

    Two molecules are equal when their symbols and coordinates match
    exactly.  The comment is free-form metadata describing the file rather
    than the chemistry, so it is deliberately left out of equality and
    hashing.  Coordinates are read-only, which is what makes instances
    safely hashable.
    """

    symbols: tuple[str, ...]
    coordinates: FloatArray
    comment: str = ""

    def __post_init__(self) -> None:
        coords = np.array(self.coordinates, dtype=float, copy=True)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coordinates must have shape (natoms, 3), got {coords.shape}")
        canonical: list[str] = []
        for index, token in enumerate(self.symbols):
            symbol = canonical_symbol(str(token))
            if symbol is None:
                raise ValueError(f"unknown element symbol {token!r} at index {index}")
            canonical.append(symbol)
        if len(self.symbols) != coords.shape[0]:
            raise ValueError(
                f"got {len(self.symbols)} symbols but {coords.shape[0]} coordinate rows"
            )
        if coords.shape[0] == 0:
            raise ValueError("a molecule must contain at least one atom")
        if not bool(np.all(np.isfinite(coords))):
            raise ValueError("coordinates must all be finite")
        if bool(np.any(np.abs(coords) > _MAX_COORDINATE)):
            raise ValueError(
                f"coordinates must be smaller than {_MAX_COORDINATE:g} A in "
                "magnitude; larger values are not physical and overflow the "
                "symmetry search"
            )
        coords.flags.writeable = False
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "symbols", tuple(canonical))
        _reject_coincident_atoms(coords)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented
        return self.symbols == other.symbols and bool(
            np.array_equal(self.coordinates, other.coordinates)
        )

    def __hash__(self) -> int:
        # Adding zero folds -0.0 onto 0.0.  np.array_equal treats the two as
        # equal, so without this a molecule and its reflection in its own
        # plane would compare equal but hash differently.
        return hash((self.symbols, (self.coordinates + 0.0).tobytes()))

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.symbols)

    def to_xyz(self, comment: str | None = None) -> str:
        """Render the molecule as the text of a single-frame XYZ file."""
        header = self.comment if comment is None else comment
        # Any line break in the comment would turn one frame into nonsense,
        # so collapse every kind of them, not just "\n".
        lines = [str(self.natoms), " ".join(header.splitlines())]
        lines.extend(
            f"{symbol:<2s} {x:>20.12f} {y:>20.12f} {z:>20.12f}"
            for symbol, (x, y, z) in zip(self.symbols, self.coordinates, strict=True)
        )
        return "\n".join(lines) + "\n"


def _fail(source: str | None, line_number: int | None, message: str) -> XYZFormatError:
    where = source or "<string>"
    if line_number is not None:
        where = f"{where}:{line_number}"
    return XYZFormatError(f"{where}: {message}")


def _parse_coordinate(token: str, source: str | None, line_number: int) -> float:
    try:
        value = float(token)
    except ValueError:
        raise _fail(
            source, line_number, f"coordinate {token!r} is not a floating-point number"
        ) from None
    if not math.isfinite(value):
        raise _fail(source, line_number, f"coordinate {token!r} is not finite")
    return value


def _reject_coincident_atoms(coords: FloatArray) -> None:
    """Reject a geometry in which two atoms sit on top of each other.

    Uses a uniform grid rather than all pairs, so a large file costs linear
    time and memory; the all-pairs form needed ``24 * natoms**2`` bytes and
    exhausted memory on structures of a few thousand atoms.
    """
    if len(coords) < 2:
        return
    cell = _COINCIDENT_ATOM_DISTANCE
    neighbourhood = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, position in enumerate(coords):
        key = _grid_key(position, cell)
        for offset in neighbourhood:
            neighbours = buckets.get(
                (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            )
            if neighbours is None:
                continue
            for other in neighbours:
                separation = float(np.linalg.norm(coords[other] - position))
                if separation < cell:
                    raise ValueError(
                        f"atoms {other + 1} and {index + 1} are coincident "
                        f"(separation {separation:.2e} A); the geometry is not "
                        "a valid molecule"
                    )
        buckets.setdefault(key, []).append(index)


def parse_xyz(text: str, *, source: str | None = None) -> Molecule:
    """Parse the contents of a single-frame XYZ file.

    Parameters
    ----------
    text:
        Full file contents.
    source:
        Optional name used in error messages (typically the file path).

    Returns
    -------
    Molecule
        The parsed geometry, with element symbols canonicalised.

    Raises
    ------
    XYZFormatError
        If the text is not a well-formed single-frame XYZ file.
    """
    # A byte-order mark left by a Windows editor is not data.
    raw_lines = text.lstrip("\ufeff").splitlines()

    start = 0
    while start < len(raw_lines) and not raw_lines[start].strip():
        start += 1
    if start >= len(raw_lines):
        raise _fail(source, None, "file is empty")

    count_line_number = start + 1
    count_token = raw_lines[start].strip()
    try:
        natoms = int(count_token)
    except ValueError:
        raise _fail(
            source,
            count_line_number,
            f"first line must be an atom count, got {count_token!r}",
        ) from None
    if natoms <= 0:
        raise _fail(source, count_line_number, f"atom count must be positive, got {natoms}")

    comment_index = start + 1
    if comment_index >= len(raw_lines):
        raise _fail(source, count_line_number + 1, "missing comment line")
    comment = raw_lines[comment_index]

    body = raw_lines[comment_index + 1 :]
    if len(body) < natoms:
        raise _fail(
            source,
            count_line_number,
            f"header declares {natoms} atoms but only {len(body)} lines follow "
            "the comment line",
        )

    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for offset in range(natoms):
        line_number = comment_index + 2 + offset
        fields = body[offset].split()
        if not fields:
            raise _fail(
                source,
                line_number,
                f"blank line inside the atom block (expected atom {offset + 1} of {natoms})",
            )
        if len(fields) != 4:
            raise _fail(
                source,
                line_number,
                f"expected 4 fields '<symbol> <x> <y> <z>', got {len(fields)}: "
                f"{body[offset].strip()!r}",
            )
        symbol = canonical_symbol(fields[0])
        if symbol is None:
            raise _fail(source, line_number, f"unknown element symbol {fields[0]!r}")
        symbols.append(symbol)
        positions.append(
            (
                _parse_coordinate(fields[1], source, line_number),
                _parse_coordinate(fields[2], source, line_number),
                _parse_coordinate(fields[3], source, line_number),
            )
        )

    for offset, extra in enumerate(body[natoms:]):
        if extra.strip():
            raise _fail(
                source,
                comment_index + 2 + natoms + offset,
                "unexpected content after the atom block "
                f"({extra.strip()!r}); multi-frame XYZ files are not supported",
            )

    try:
        return Molecule(
            symbols=tuple(symbols),
            coordinates=np.asarray(positions, dtype=float).reshape(natoms, 3),
            comment=comment,
        )
    except ValueError as exc:
        raise _fail(source, None, str(exc)) from exc


def read_xyz(path: str | Path) -> Molecule:
    """Read a single-frame XYZ file from disk.

    Parameters
    ----------
    path:
        Path to the ``.xyz`` file.

    Returns
    -------
    Molecule

    Raises
    ------
    XYZFormatError
        If the file cannot be read or is not a well-formed XYZ file.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise XYZFormatError(f"{file_path}: cannot read file ({exc.strerror})") from exc
    except UnicodeDecodeError as exc:
        raise XYZFormatError(f"{file_path}: file is not valid UTF-8 text") from exc
    return parse_xyz(text, source=str(file_path))


def molecule_from_coordinates(
    symbols: Sequence[str], coordinates: FloatArray | Sequence[Sequence[float]]
) -> Molecule:
    """Build a :class:`Molecule` from in-memory symbols and coordinates.

    The input array is copied, so the caller's data is never mutated.

    Raises
    ------
    XYZFormatError
        If a symbol names no element, or the coordinates are not a numeric
        ``(natoms, 3)`` array of finite, physically sized values, or two
        atoms coincide.
    """
    try:
        array = np.asarray(coordinates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise XYZFormatError(f"coordinates must be numeric: {exc}") from exc
    try:
        return Molecule(symbols=tuple(str(token) for token in symbols), coordinates=array)
    except ValueError as exc:
        raise XYZFormatError(str(exc)) from exc
