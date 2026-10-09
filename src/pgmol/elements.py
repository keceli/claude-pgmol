"""Chemical element symbols used to validate XYZ input.

The package never needs atomic masses: every symmetry operation of a molecule
permutes atoms of the *same* element, so the unweighted centroid of the atoms
is a fixed point of the full point group and serves as the symmetry origin.
Element data is therefore only used to validate and canonicalise symbols.
"""

from __future__ import annotations

__all__ = ["ELEMENT_SYMBOLS", "canonical_symbol"]

#: Standard IUPAC symbols for elements 1-118, indexed by atomic number - 1.
ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

_BY_CASEFOLD = {symbol.casefold(): symbol for symbol in ELEMENT_SYMBOLS}


def canonical_symbol(token: str) -> str | None:
    """Return the canonical element symbol for ``token``, or ``None``.

    Accepts an element symbol in any capitalisation (``"cl"``, ``"CL"``,
    ``"Cl"``) and an atomic number written as a plain integer (``"17"``).
    Returns ``None`` when the token names no element in 1-118.
    """
    stripped = token.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        number = int(stripped)
        if 1 <= number <= len(ELEMENT_SYMBOLS):
            return ELEMENT_SYMBOLS[number - 1]
        return None
    return _BY_CASEFOLD.get(stripped.casefold())
