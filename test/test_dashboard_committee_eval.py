"""Tests for ``lammps_al.dashboard.committee_eval``.

Two layers: the scoring/normalization plumbing via the injectable
``_build_calculator`` seam (no pyace needed), and the *real* wiring into
``lammps_al.run.scripts.lammps_sampling._build_committee_calculator`` via a
fake ``pyace`` module installed into ``sys.modules`` -- the same technique
``test_payload_unit.py`` already uses for that function's ace/grace
dispatch, reused here rather than reinvented.
"""

import json
import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from lammps_al.dashboard.committee_eval import (
    _default_committee_calculator,
    read_reference_energies,
    score_with_committee,
)

pytestmark = pytest.mark.local


class _FakeCommitteeCalculator(Calculator):
    """A deterministic, position-independent stand-in calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, model_paths, energy=1.0, force_value=0.5):
        super().__init__()
        self.model_paths = model_paths
        self._energy = energy
        self._force_value = force_value

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.results["energy"] = self._energy
        self.results["forces"] = np.full((n, 3), self._force_value)


def _seam(model_files, device):
    return _FakeCommitteeCalculator(model_files)


def test_score_with_committee_uses_injected_calculator():
    atoms_list = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])]

    energies, forces = score_with_committee(
        ["m0.yaml", "m1.yaml"], atoms_list, _build_calculator=_seam
    )

    assert energies == [1.0]
    np.testing.assert_array_equal(forces[0], np.full((2, 3), 0.5))


def test_score_with_committee_does_not_mutate_input_atoms():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    assert atoms.calc is None

    score_with_committee(["m.yaml"], [atoms], _build_calculator=_seam)

    assert atoms.calc is None  # scored on a copy, not the original


def test_score_with_committee_want_forces_false_skips_forces():
    atoms_list = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])]

    energies, forces = score_with_committee(
        ["m.yaml"], atoms_list, want_forces=False, _build_calculator=_seam
    )

    assert energies == [1.0]
    assert forces is None


def test_score_with_committee_multiple_configs():
    atoms_list = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]),
        Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    ]

    energies, forces = score_with_committee(
        ["m.yaml"], atoms_list, _build_calculator=_seam
    )

    assert energies == [1.0, 1.0]
    assert forces[0].shape == (2, 3)
    assert forces[1].shape == (3, 3)


def _write_potential(path, reference_energy=None):
    lines = ["metadata:", "  pacemaker_version: 0.2.7"]
    if reference_energy is not None:
        # pacemaker's own format: a JSON string, YAML double-quoted.
        encoded = json.dumps(json.dumps(reference_energy))
        lines.append(f"  reference_energy: {encoded}")
    lines += ["  user: someone", "global:", "  reference_energy: 99", ""]
    path.write_text("\n".join(lines))
    return path


def test_read_reference_energies_from_metadata(tmp_path):
    ref = {"H": -0.18, "O": -0.09, "Cu": -0.036}
    files = [_write_potential(tmp_path / f"m{i}.yaml", ref) for i in range(2)]

    assert read_reference_energies(files) == pytest.approx(ref)


def test_read_reference_energies_absent(tmp_path):
    files = [_write_potential(tmp_path / "m.yaml"), tmp_path / "gone.yaml"]

    assert read_reference_energies(files) == {}


def test_read_reference_energies_disagreeing_members_raise(tmp_path):
    files = [
        _write_potential(tmp_path / "m0.yaml", {"H": -0.18}),
        _write_potential(tmp_path / "m1.yaml", {"H": -0.20}),
    ]

    with pytest.raises(ValueError, match="different reference energies"):
        read_reference_energies(files)


def test_score_with_committee_adds_reference_energy(tmp_path):
    model = _write_potential(tmp_path / "m.yaml", {"H": -0.5, "O": -2.0})
    atoms_list = [
        Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    ]

    energies, _ = score_with_committee(
        [model], atoms_list, _build_calculator=_seam
    )

    # Fake calculator's 1.0 plus 2 * (-0.5) + (-2.0).
    assert energies == [pytest.approx(-2.0)]


# ---------------------------------------------------------------------------
# The real default path: wiring into lammps_sampling._build_committee_calculator
# ---------------------------------------------------------------------------
def _install_fake_pyace_module(monkeypatch):
    """Make ``from pyace import PyACEEnsembleCalculator`` resolve to the
    fake, mirroring test_payload_unit.py's ``_install_fake_ace_module``."""
    module = types.ModuleType("pyace")
    module.PyACEEnsembleCalculator = _FakeCommitteeCalculator
    monkeypatch.setitem(sys.modules, "pyace", module)


def test_default_committee_calculator_wires_into_real_build_function(
    monkeypatch,
):
    """Confirms committee_eval actually calls lammps_al's own
    _build_committee_calculator (backend="ace"), not a reimplemented
    pyace-loading path -- with no real pyace installed."""
    _install_fake_pyace_module(monkeypatch)

    calc = _default_committee_calculator(["m0.yaml", "m1.yaml"], "cpu")

    assert isinstance(calc, _FakeCommitteeCalculator)
    assert calc.model_paths == ["m0.yaml", "m1.yaml"]


def test_score_with_committee_default_path_end_to_end(monkeypatch):
    _install_fake_pyace_module(monkeypatch)
    atoms_list = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])]

    energies, forces = score_with_committee(["m0.yaml", "m1.yaml"], atoms_list)

    assert energies == [1.0]
    np.testing.assert_array_equal(forces[0], np.full((2, 3), 0.5))
