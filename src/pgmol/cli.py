"""Command-line interface: ``pgmol path/to/molecule.xyz``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .api import DEFAULT_TOLERANCE, analyze
from .errors import PGMolError
from .xyz import read_xyz

__all__ = ["build_parser", "main"]

#: Exit status used for unreadable, malformed or unanalysable input.
EXIT_INPUT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``pgmol`` command."""
    parser = argparse.ArgumentParser(
        prog="pgmol",
        description="Determine the Schoenflies point group of a molecule "
        "given a single-frame XYZ file (coordinates in angstrom).",
    )
    parser.add_argument("xyz", metavar="XYZFILE", help="path to the XYZ file")
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        metavar="ANGSTROM",
        help="maximum displacement between a symmetry-transformed atom and "
        f"its partner (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also report the group order and the operations found",
    )
    parser.add_argument("--version", action="version", version=f"pgmol {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        molecule = read_xyz(args.xyz)
        result = analyze(molecule, tolerance=args.tolerance)
    except PGMolError as exc:
        print(f"pgmol: error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # Flush before writing diagnostics so the two streams stay in order when
    # both are attached to the same terminal.
    # Flush before writing diagnostics so the two streams stay in order when
    # both are attached to the same terminal.
    print(result.label, flush=True)
    if args.verbose:
        order = "infinite" if result.order is None else str(result.order)
        print(f"atoms: {molecule.natoms}", file=sys.stderr)
        print(f"tolerance: {result.tolerance} A", file=sys.stderr)
        if result.effective_tolerance != result.tolerance:
            print(
                f"effective tolerance: {result.effective_tolerance:.4g} A "
                "(capped: atoms of one element are too close for the "
                "requested value to be unambiguous)",
                file=sys.stderr,
            )
        print(f"order: {order}", file=sys.stderr)
        if result.operations:
            summary = ", ".join(
                f"{count}x{label}" for label, count in result.operation_counts().items()
            )
            print(f"operations: {summary}", file=sys.stderr)
    if not result.consistent:
        # Always warn, not only under --verbose: the package has proved the
        # operations it found are not a group, so the label on stdout is
        # unreliable and a caller reading only stdout must still be told.
        print(
            f"pgmol: warning: {args.xyz}: the operations found at tolerance "
            f"{result.effective_tolerance:g} A do not form a group, so "
            f"'{result.label}' is only approximate; try a smaller --tolerance",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
