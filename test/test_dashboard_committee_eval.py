"""Tests for ``lammps_al.dashboard.committee_eval``.

Two layers: the scoring/normalization plumbing via the injectable
``_build_calculator`` seam (no pyace needed), and the *real* wiring into
``lammps_al.run.scripts.lammps_sampling._build_committee_calculator`` via a
fake ``pyace`` module installed into ``sys.modules`` -- the same technique
``test_payload_unit.py`` already uses for that function's ace/grace
dispatch, reused here rather than reinvented.
"""

import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from lammps_al.dashboard.committee_eval import (
    _default_committee_calculator,
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
