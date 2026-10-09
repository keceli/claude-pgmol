"""Reading and validating plain XYZ molecular geometry files.

Supported dialect
-----------------
A file holds exactly one frame::

    <natoms>
    <comment>
    <symbol> <x> <y> <z>
    ... (natoms lines total)

* Line 1 is the atom count; leading/trailing whitespace is ignored and blank
  lines *before* it are skipped.
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
from .geometry import FloatArray

__all__ = ["Molecule", "parse_xyz", "read_xyz"]

#: Two atoms closer than this (ångström) are treated as a corrupt geometry.
_COINCIDENT_ATOM_DISTANCE = 1e-5


@dataclass(frozen=True)
class Molecule:
    """An immutable set of labelled Cartesian atomic positions.

    Attributes
    ----------
    symbols:
        Canonical IUPAC element symbols, one per atom, in input order.
    coordinates:
        Read-only ``(natoms, 3)`` float64 array of ångström coordinates.
    comment:
        The XYZ comment line, verbatim (without its newline).
    """

    symbols: tuple[str, ...]
    coordinates: FloatArray
    comment: str = ""

    def __post_init__(self) -> None:
        coords = np.array(self.coordinates, dtype=float, copy=True)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coordinates must have shape (natoms, 3), got {coords.shape}")
        if len(self.symbols) != coords.shape[0]:
            raise ValueError(
                f"got {len(self.symbols)} symbols but {coords.shape[0]} coordinate rows"
            )
        if coords.shape[0] == 0:
            raise ValueError("a molecule must contain at least one atom")
        if not np.isfinite(coords).all():
            raise ValueError("coordinates must all be finite")
        coords.flags.writeable = False
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "symbols", tuple(self.symbols))

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.symbols)

    def to_xyz(self, comment: str | None = None) -> str:
        """Render the molecule as the text of a single-frame XYZ file."""
        header = self.comment if comment is None else comment
        lines = [str(self.natoms), header.replace("\n", " ")]
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


def _check_distinct(molecule: Molecule, source: str | None) -> None:
    coords = molecule.coordinates
    if len(coords) < 2:
        return
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    i, j = np.unravel_index(int(np.argmin(distances)), distances.shape)
    if distances[i, j] < _COINCIDENT_ATOM_DISTANCE:
        raise _fail(
            source,
            None,
            f"atoms {min(i, j) + 1} and {max(i, j) + 1} are coincident "
            f"(separation {distances[i, j]:.2e} A); the geometry is not a valid molecule",
        )


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
    raw_lines = text.splitlines()

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

    molecule = Molecule(
        symbols=tuple(symbols),
        coordinates=np.asarray(positions, dtype=float).reshape(natoms, 3),
        comment=comment,
    )
    _check_distinct(molecule, source)
    return molecule


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
        text = file_path.read_text(encoding="utf-8")
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
    """
    canonical: list[str] = []
    for index, token in enumerate(symbols):
        symbol = canonical_symbol(str(token))
        if symbol is None:
            raise XYZFormatError(f"unknown element symbol {token!r} at index {index}")
        canonical.append(symbol)
    array = np.array(coordinates, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise XYZFormatError(f"coordinates must have shape (natoms, 3), got {array.shape}")
    if len(canonical) != array.shape[0]:
        raise XYZFormatError(
            f"got {len(canonical)} symbols but {array.shape[0]} coordinate rows"
        )
    if array.shape[0] == 0:
        raise XYZFormatError("a molecule must contain at least one atom")
    if not np.isfinite(array).all():
        raise XYZFormatError("coordinates must all be finite")
    molecule = Molecule(symbols=tuple(canonical), coordinates=array)
    _check_distinct(molecule, None)
    return molecule
