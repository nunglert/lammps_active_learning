"""Tests for the top-level ``lammps-al`` dispatcher (``lammps_al.__main__``)."""

import pytest

from lammps_al import __main__ as lammps_al_main
from lammps_al import __version__

pytestmark = pytest.mark.local


@pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]])
def test_help(argv, capsys):
    assert lammps_al_main.main(argv) == 0
    assert "usage: lammps-al" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version(flag, capsys):
    assert lammps_al_main.main([flag]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_unknown_command(capsys):
    assert lammps_al_main.main(["nope"]) == 2
    assert "unknown command 'nope'" in capsys.readouterr().err


def test_dashboard_dispatches_to_cli(monkeypatch):
    pytest.importorskip("alviz.dashboard")
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 7

    monkeypatch.setattr("lammps_al.dashboard.cli.main", fake_main)
    assert lammps_al_main.main(["dashboard", "calchome", "-v"]) == 7
    assert seen["argv"] == ["calchome", "-v"]
