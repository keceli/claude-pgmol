"""Command-line interface: success, diagnostics and error exits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import acceptance_files, expected_label
from pgmol.cli import build_parser, main


def test_prints_the_label_and_exits_zero(water_xyz: Path, capsys) -> None:
    assert main([str(water_xyz)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "C2v\n"
    assert captured.err == ""


@pytest.mark.parametrize("path", acceptance_files()[:10], ids=lambda p: p.stem)
def test_cli_agrees_with_the_fixture_names(path: Path, capsys) -> None:
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.strip() == expected_label(path)


def test_tolerance_option_is_honoured(tmp_path: Path, capsys) -> None:
    distorted = tmp_path / "distorted.xyz"
    # One O-H stretched by 0.08 A: C2v only once the tolerance covers it.
    distorted.write_text(
        "3\nstretched water\nO 0 0 0\nH 0.857 0 0.587\nH -0.757 0 0.587\n",
        encoding="utf-8",
    )
    assert main([str(distorted), "--tolerance", "0.05"]) == 0
    assert capsys.readouterr().out.strip() == "Cs"
    assert main([str(distorted)]) == 0
    assert capsys.readouterr().out.strip() == "C2v"
    assert main([str(distorted), "-t", "0.2"]) == 0
    assert capsys.readouterr().out.strip() == "C2v"


def test_verbose_reports_details_on_stderr(water_xyz: Path, capsys) -> None:
    assert main([str(water_xyz), "--verbose"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "C2v\n"
    assert "atoms: 3" in captured.err
    assert "order: 4" in captured.err
    assert "operations:" in captured.err
    assert "tolerance: 0.1" in captured.err


def test_verbose_on_a_linear_molecule_says_infinite(tmp_path: Path, capsys) -> None:
    path = tmp_path / "co.xyz"
    path.write_text("2\ncarbon monoxide\nC 0 0 0.6\nO 0 0 -0.5\n", encoding="utf-8")
    assert main([str(path), "-v"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Cinfv"
    assert "order: infinite" in captured.err


def test_missing_file_exits_nonzero_with_a_message(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path / "nope.xyz")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pgmol: error:" in captured.err
    assert "cannot read file" in captured.err


def test_malformed_file_exits_nonzero(tmp_path: Path, capsys) -> None:
    path = tmp_path / "broken.xyz"
    path.write_text("5\ntruncated\nH 0 0 0\n", encoding="utf-8")
    assert main([str(path)]) == 1
    assert "pgmol: error:" in capsys.readouterr().err


def test_bad_tolerance_exits_nonzero(water_xyz: Path, capsys) -> None:
    assert main([str(water_xyz), "--tolerance", "0"]) == 1
    assert "positive finite" in capsys.readouterr().err


def test_non_numeric_tolerance_is_a_usage_error(water_xyz: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([str(water_xyz), "--tolerance", "wide"])
    assert excinfo.value.code == 2


def test_missing_argument_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_parser_metadata() -> None:
    parser = build_parser()
    assert parser.prog == "pgmol"
    assert "point group" in (parser.description or "")


def test_installed_console_script(water_xyz: Path) -> None:
    """The entry point declared in pyproject really runs."""
    completed = subprocess.run(
        [sys.executable, "-m", "pgmol.cli", str(water_xyz)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "C2v"


def test_console_script_binary_is_available(water_xyz: Path) -> None:
    executable = Path(sys.executable).with_name("pgmol")
    if not executable.exists():  # pragma: no cover - depends on install layout
        pytest.skip("console script not installed in this environment")
    completed = subprocess.run(
        [str(executable), str(water_xyz)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "C2v"


def test_verbose_warns_when_the_group_is_only_approximate(tmp_path: Path, capsys) -> None:
    """A tolerance too loose for the geometry yields a non-group, and says so."""
    import numpy as np

    rng = np.random.default_rng(0)
    cube = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    distorted = cube + rng.normal(scale=0.12, size=cube.shape)
    lines = ["8", "distorted cube"]
    lines += [f"C {x:.9f} {y:.9f} {z:.9f}" for x, y, z in distorted]
    path = tmp_path / "cube.xyz"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert main([str(path), "--tolerance", "0.35", "--verbose"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert "warning:" in captured.err
    assert "approximately symmetric" in captured.err


def test_verbose_reports_a_capped_tolerance(tmp_path: Path, capsys) -> None:
    path = tmp_path / "close.xyz"
    path.write_text(
        "4\nclose hydrogens\nH 1.0 0 0\nH 1.06 0.104 0\nH -1.0 0 0.4\nO 0 0 -0.3\n",
        encoding="utf-8",
    )
    assert main([str(path), "--tolerance", "0.5", "-v"]) == 0
    captured = capsys.readouterr()
    assert "tolerance: 0.5 A" in captured.err
    assert "effective tolerance:" in captured.err
    assert "capped" in captured.err


def test_verbose_omits_the_cap_line_when_nothing_is_capped(water_xyz: Path, capsys) -> None:
    assert main([str(water_xyz), "-v"]) == 0
    assert "effective tolerance:" not in capsys.readouterr().err
