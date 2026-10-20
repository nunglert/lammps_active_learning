"""Tests for ``lammps_al.label``: ``VaspD3Evaluator`` and its payload.

Needs the [run] extra (remote_handler); the module self-skips otherwise. No
VASP and no simple-dft-d3 needed: the payload runs with a fake ``Vasp``
calculator and a fake ``dftd3`` module (one test cross-checks the real
package when it is installed).
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.units import Bohr, Hartree  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402

from lammps_al.label.evaluator import VaspD3Evaluator  # noqa: E402
from lammps_al.label.scripts import script_dir  # noqa: E402

pytestmark = pytest.mark.integration

_D3 = {"method": "RPBE", "damping": "d3zero", "atm": True}
_CONVERGED = "------ aborting loop because EDIFF is reached ------\n"
_UNCONVERGED = "------ aborting loop EDIFF was not reached (unconverged) ---\n"


# --- VaspD3Evaluator ------------------------------------------------------
def _evaluator(**overrides):
    kwargs = dict(
        remote_fs=MagicMock(),
        cluster_config=ClusterConfig(
            remote_machine="mock",
            partition="default",
            conda_env="test",
            conda_source="/opt/conda",
            savepath="/tmp/remote_handler",
            template="mock.sh.j2",
        ),
        num_parallel=1,
        incar_kwargs={"xc": "RPBE", "encut": 850},
        d3=_D3,
    )
    kwargs.update(overrides)
    return VaspD3Evaluator(**kwargs)


def test_evaluator_ships_d3_payload_and_settings():
    ev = _evaluator()

    assert Path(ev.singlepoint_script) == (
        script_dir() / "vasp_d3_singlepoint.py"
    )
    assert Path(ev.singlepoint_script).is_file()
    assert ev.calc_config["d3"] == _D3
    assert ev.calc_config["INCAR_kwargs"] == {"xc": "RPBE", "encut": 850}


@pytest.mark.parametrize("key", ["ivdw", "IVDW"])
def test_evaluator_rejects_vasp_d3(key):
    with pytest.raises(ValueError, match="IVDW"):
        _evaluator(incar_kwargs={"xc": "RPBE", key: 11})


@pytest.mark.parametrize(
    "d3,match",
    [
        ({"method": "RPBE", "damping": "d3zero"}, "missing"),
        ({"method": "RPBE", "damping": "d4", "atm": True}, "damping"),
    ],
)
def test_evaluator_rejects_incomplete_d3(d3, match):
    with pytest.raises(ValueError, match=match):
        _evaluator(d3=d3)


# --- payload --------------------------------------------------------------
def _load_payload():
    path = script_dir() / "vasp_d3_singlepoint.py"
    spec = importlib.util.spec_from_file_location("vasp_d3_singlepoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_dftd3(monkeypatch, energy_ha=-0.5, grad=0.01):
    """A ``dftd3.interface`` returning fixed results (Hartree, Hartree/Bohr)
    and recording the damping-parameter arguments."""
    calls = []
    interface = types.ModuleType("dftd3.interface")

    class _Param:
        def __init__(self, **kwargs):
            calls.append((type(self).__name__, kwargs))

    class ZeroDampingParam(_Param):
        pass

    class RationalDampingParam(_Param):
        pass

    class DispersionModel:
        def __init__(self, numbers, positions, lattice, periodic):
            self.n = len(numbers)

        def get_dispersion(self, param, grad):
            return {
                "energy": energy_ha,
                "gradient": np.full((self.n, 3), grad_value),
            }

    grad_value = grad
    interface.ZeroDampingParam = ZeroDampingParam
    interface.RationalDampingParam = RationalDampingParam
    interface.DispersionModel = DispersionModel
    package = types.ModuleType("dftd3")
    package.interface = interface
    monkeypatch.setitem(sys.modules, "dftd3", package)
    monkeypatch.setitem(sys.modules, "dftd3.interface", interface)
    return calls


class _FakeVasp(Calculator):
    """Stand-in for ``ase.calculators.vasp.Vasp``."""

    implemented_properties = ["energy", "forces"]
    outcar_tail = _CONVERGED

    def __init__(self, directory, **kwargs):
        super().__init__(directory=directory)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes or all_changes)
        (Path(self.directory) / "OUTCAR").write_text(self.outcar_tail)
        self.results = {"energy": -10.0, "forces": np.ones((len(atoms), 3))}


def _calcdir(tmp_path, incar=None, d3=_D3):
    calcdir = tmp_path / "calc"
    calcdir.mkdir()
    ase.io.write(
        calcdir / "dft.extxyz",
        Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5], pbc=True
        ),
    )
    (calcdir / "input_dft.json").write_text(
        json.dumps(
            {
                "DATABASE": "dft.extxyz",
                "launch_command": "vasp_std",
                "INCAR_kwargs": incar or {"encut": 300},
                "d3": d3,
            }
        )
    )
    return calcdir


def test_payload_adds_d3_with_explicit_atm(tmp_path, monkeypatch):
    calls = _install_fake_dftd3(monkeypatch, energy_ha=-0.5, grad=0.01)
    payload = _load_payload()
    monkeypatch.setattr(payload, "Vasp", _FakeVasp)
    calcdir = _calcdir(tmp_path)

    payload.main(["vasp_d3_singlepoint.py", str(calcdir)])

    assert calls == [("ZeroDampingParam", {"method": "RPBE", "atm": True})]
    labelled = ase.io.read(calcdir / "dft_dft.extxyz")
    d3_energy = -0.5 * Hartree
    assert labelled.get_potential_energy() == pytest.approx(-10.0 + d3_energy)
    # forces = -gradient, converted from Hartree/Bohr.
    np.testing.assert_allclose(
        labelled.get_forces(), 1.0 - 0.01 * Hartree / Bohr
    )
    assert labelled.info["dft_energy"] == pytest.approx(-10.0)
    assert labelled.info["d3_energy"] == pytest.approx(d3_energy)
    assert (calcdir / "is_finished.info").exists()


def test_payload_does_not_label_unconverged_scf(tmp_path, monkeypatch):
    _install_fake_dftd3(monkeypatch)
    payload = _load_payload()

    class _Unconverged(_FakeVasp):
        outcar_tail = _UNCONVERGED

    monkeypatch.setattr(payload, "Vasp", _Unconverged)
    calcdir = _calcdir(tmp_path)

    with pytest.raises(SystemExit, match="SCF not converged"):
        payload.main(["vasp_d3_singlepoint.py", str(calcdir)])

    assert not (calcdir / "dft_dft.extxyz").exists()
    assert not (calcdir / "is_finished.info").exists()
    assert (calcdir / "scf_unconverged.info").exists()


def test_payload_refuses_vasp_d3(tmp_path, monkeypatch):
    _install_fake_dftd3(monkeypatch)
    payload = _load_payload()
    monkeypatch.setattr(payload, "Vasp", _FakeVasp)
    calcdir = _calcdir(tmp_path, incar={"encut": 300, "IVDW": 11})

    with pytest.raises(ValueError, match="IVDW"):
        payload.main(["vasp_d3_singlepoint.py", str(calcdir)])


def test_d3_three_body_term_with_real_simple_dft_d3():
    """With the real package: the atm switch changes only the three-body
    term, and forces are the negative energy gradient."""
    pytest.importorskip("dftd3.interface")
    from ase.build import bulk

    payload = _load_payload()
    # A smooth environment (random points can sit on a cutoff kink, where
    # finite differences disagree with the analytic gradient).
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.1, seed=1)
    two_body = {**_D3, "atm": False}

    e3, f3 = payload.d3_energy_forces(atoms, _D3)
    e2, _ = payload.d3_energy_forces(atoms, two_body)
    assert e3 != pytest.approx(e2)

    # Finite-difference check of one force component.
    h = 1e-4
    shifted = atoms.copy()
    shifted.positions[0, 0] += h
    e_plus, _ = payload.d3_energy_forces(shifted, _D3)
    shifted.positions[0, 0] -= 2 * h
    e_minus, _ = payload.d3_energy_forces(shifted, _D3)
    assert f3[0, 0] == pytest.approx(-(e_plus - e_minus) / (2 * h), rel=1e-4)
