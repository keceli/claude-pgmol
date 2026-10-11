"""The web page runs pgmol in the browser through a small Python shim.

These tests execute the *actual* shim extracted from ``web/worker.js``, so a
refactor that changes the public API cannot silently break the deployed page
-- the page itself cannot be exercised from pytest, but its only Python is
this snippet, and it is covered here against the real package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER = REPO_ROOT / "web" / "worker.js"
PAGE = REPO_ROOT / "web" / "index.html"
APP = REPO_ROOT / "web" / "app.js"


def load_shim():
    """Compile the Python embedded in the worker and return its entry point."""
    source = WORKER.read_text(encoding="utf-8")
    match = re.search(r"const GLUE = `\n(.*?)`;", source, re.DOTALL)
    assert match, "could not find the GLUE template literal in worker.js"
    namespace: dict = {}
    exec(compile(match.group(1), str(WORKER), "exec"), namespace)
    return namespace["_analyse"]


@pytest.fixture(scope="module")
def analyse():
    return load_shim()


WATER = "3\nwater\nO 0 0 0\nH 0.757 0 0.587\nH -0.757 0 0.587\n"


def test_shim_returns_the_fields_the_page_renders(analyse) -> None:
    payload = json.loads(analyse(WATER, 0.1))
    assert payload["ok"] is True
    # Every key app.js reads must be present.
    assert set(payload) >= {
        "label",
        "natoms",
        "comment",
        "order",
        "consistent",
        "tolerance",
        "effective_tolerance",
        "counts",
    }
    assert payload["label"] == "C2v"
    assert payload["natoms"] == 3
    assert payload["order"] == 4
    assert payload["consistent"] is True
    assert payload["tolerance"] == 0.1
    assert payload["effective_tolerance"] == 0.1
    assert payload["counts"] == {"C2": 1, "E": 1, "sigma": 2}
    assert payload["comment"] == "water"


def test_shim_output_is_json_serialisable(analyse) -> None:
    """It crosses a worker boundary as text, so nothing may be a Python object."""
    for text in (WATER, "1\n\nAr 0 0 0\n", "2\n\nC 0 0 0.6\nO 0 0 -0.5\n"):
        assert json.loads(analyse(text, 0.1))["ok"] is True


def test_shim_reports_an_infinite_group_as_null_order(analyse) -> None:
    payload = json.loads(analyse("2\n\nN 0 0 0.55\nN 0 0 -0.55\n", 0.1))
    assert payload["label"] == "Dinfh"
    assert payload["order"] is None


def test_shim_surfaces_the_capped_tolerance(analyse) -> None:
    payload = json.loads(
        analyse("4\nclose\nH 1.0 0 0\nH 1.06 0.104 0\nH -1.0 0 0.4\nO 0 0 -0.3\n", 0.5)
    )
    assert payload["tolerance"] == 0.5
    assert payload["effective_tolerance"] < 0.5


def test_shim_surfaces_an_inconsistent_group(analyse) -> None:
    """The page must warn exactly where the CLI warns."""
    import numpy as np

    rng = np.random.default_rng(0)
    cube = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    distorted = cube + rng.normal(scale=0.12, size=cube.shape)
    text = "8\ndistorted cube\n" + "".join(
        f"C {x:.9f} {y:.9f} {z:.9f}\n" for x, y, z in distorted
    )
    assert json.loads(analyse(text, 0.35))["consistent"] is False


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("5\ntruncated\nH 0 0 0\n", "only 1 lines follow"),
        ("", "file is empty"),
        ("1\nc\nXx 0 0 0\n", "unknown element symbol"),
        ("2\nc\nH 0 0 0\nH 0 0 0\n", "coincident"),
    ],
)
def test_shim_reports_bad_input_without_raising(analyse, text: str, fragment: str) -> None:
    payload = json.loads(analyse(text, 0.1))
    assert payload["ok"] is False
    assert fragment in payload["error"]


def test_shim_reports_a_bad_tolerance_as_user_error(analyse) -> None:
    """SymmetryError is a PGMolError, so it reaches the user as a message.

    app.js rejects a non-positive tolerance before sending, but the shim must
    not crash the worker if one gets through.
    """
    payload = json.loads(analyse(WATER, -1.0))
    assert payload["ok"] is False
    assert "positive finite" in payload["error"]


def test_shim_lets_genuinely_unexpected_errors_escape(analyse) -> None:
    """Anything that is not PGMolError is a bug and must not be dressed up
    as a malformed file; worker.js reports those separately."""
    with pytest.raises(AttributeError):
        analyse(None, 0.1)


# --- the page's own wiring ------------------------------------------------


def test_worker_pins_an_exact_pyodide_version() -> None:
    """An unpinned CDN version would change the interpreter under the page."""
    source = WORKER.read_text(encoding="utf-8")
    match = re.search(r'PYODIDE_VERSION = "([\d.]+)"', source)
    assert match, "worker.js must pin a Pyodide version"
    # 0.29.x ships CPython 3.13, inside this package's supported range.
    assert match.group(1).startswith("0.29."), match.group(1)


def test_page_references_the_scripts_it_ships() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert 'src="app.js"' in html
    assert 'new Worker("worker.js")' in APP.read_text(encoding="utf-8")


def test_every_element_app_js_looks_up_exists_in_the_page() -> None:
    html = PAGE.read_text(encoding="utf-8")
    for element_id in sorted(
        set(re.findall(r'\$\("([\w-]+)"\)', APP.read_text(encoding="utf-8")))
    ):
        assert f'id="{element_id}"' in html, (
            f"app.js reads #{element_id}, absent from index.html"
        )


# --- the 3D viewer's data ------------------------------------------------


def test_shim_returns_geometry_in_the_frame_the_elements_use(analyse) -> None:
    """Symmetry elements pass through the centroid, so the viewer needs the
    centred coordinates, not the ones as written in the file."""
    payload = json.loads(analyse("3\nwater\nO 9 9 9\nH 9.757 9 9.587\nH 8.243 9 9.587\n", 0.1))
    lines = payload["xyz"].splitlines()
    assert lines[0] == "3"
    assert len(lines) == 5
    coordinates = [[float(v) for v in line.split()[1:]] for line in lines[2:]]
    centroid = [sum(c[i] for c in coordinates) / 3 for i in range(3)]
    assert centroid == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert payload["radius"] >= 1.0


def test_shim_returns_drawable_symmetry_elements(analyse) -> None:
    payload = json.loads(analyse(WATER, 0.1))
    elements = payload["elements"]
    assert [e["label"] for e in elements] == ["C2", "sigma", "sigma"]
    for element in elements:
        assert set(element) == {
            "kind",
            "label",
            "rotation_order",
            "improper_order",
            "direction",
        }
        assert element["kind"] in {"axis", "plane", "inversion"}
        direction = element["direction"]
        assert direction is not None and len(direction) == 3
        assert all(isinstance(v, float) for v in direction)
        assert sum(v * v for v in direction) == pytest.approx(1.0)


def test_shim_marks_an_inversion_centre_with_no_direction(analyse) -> None:
    payload = json.loads(analyse(Path("core_xyz/C2Cl2F2H2_Ci.xyz").read_text(), 0.1))
    assert [e["label"] for e in payload["elements"]] == ["i"]
    assert payload["elements"][0]["direction"] is None


def test_shim_reports_no_elements_for_an_infinite_group(analyse) -> None:
    payload = json.loads(analyse("3\nco2\nO 0 0 1.16\nC 0 0 0\nO 0 0 -1.16\n", 0.1))
    assert payload["label"] == "Dinfh"
    assert payload["elements"] == []
    assert payload["xyz"]  # the molecule is still drawable


def test_embedded_python_contains_no_backslash() -> None:
    """The shim lives in a JS template literal, where a backslash means one
    thing to the browser and another to the test that extracts it. Keeping
    the Python escape-free keeps the two readings identical."""
    source = WORKER.read_text(encoding="utf-8")
    glue = re.search(r"const GLUE = `\n(.*?)`;", source, re.DOTALL).group(1)
    assert "\\" not in glue, "the embedded Python must not contain backslashes"


def test_viewer_pins_an_exact_3dmol_version() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert re.search(r"3dmol@\d+\.\d+\.\d+/build/3Dmol-min\.js", html), (
        "index.html must load a pinned 3Dmol build"
    )
