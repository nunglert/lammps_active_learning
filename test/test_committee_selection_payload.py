"""Unit tests for the pure logic of the committee-selection payload.

These exercise the payload's dependency-light helpers -- local-uncertainty
aggregation and the segment/argmax peak-frame logic (the latter no longer
called by this payload itself, but still defined here for
``PeakUncertaintySelector``'s host-side reuse -- see
test_committee_selector.py) -- with only ase + numpy. No remote_handler, no
MACE/ACE/GRACE stack, no LAMMPS binary, no GPU. They are the fast,
always-run tier that validates the annotation *logic*.

Also covers the ``ace``/``grace`` branches of ``_build_committee_calculator``/
``_committee_force_stdev`` -- the only backend-specific pieces of the payload
(see its module docstring). No real tiny trained ACE/GRACE committee
potentials exist yet (unlike MACE's, borrowed from ``examples/mace_md_AL``),
so these use fake stand-in ASE calculators that mimic
``pyace.PyACEEnsembleCalculator``'s/``tensorpotential``'s ``TPCalculator``'s
constructor signature and already-aggregated ``forces_std`` result,
installed into ``sys.modules`` so the payload's real (lazy) ``from pyace
import ...``/``from tensorpotential ... import ...`` statements resolve
without either package installed. This checks the *dispatch and result-key
wiring*, not real ACE/GRACE inference.

The payload is imported by file path so the import never pulls in
``lammps_al.select.inputs`` (its ``remote_handler`` dependency is confined to
``main``).
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

pytestmark = pytest.mark.local


def _load_payload():
    """Import the payload module by path (its main() is not executed)."""
    path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "lammps_al"
        / "select"
        / "scripts"
        / "committee_selection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "committee_selection_payload", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


payload = _load_payload()


# --- aggregate_uncertainties ----------------------------------------------
def test_aggregate_uncertainties_full_cutoff_averages_neighbours():
    # Two atoms 1 A apart; per-atom mean uncertainty 3 and 1. With a cutoff
    # that includes both (and each atom itself), every atom's aggregate is the
    # mean over the pair = 2.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cell = np.eye(3) * 20.0
    pbc = np.array([False, False, False])
    atomic_unc = np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])  # means 3, 1

    agg = payload.aggregate_uncertainties(
        pos, cell, pbc, atomic_unc, r_cut=1.5
    )

    np.testing.assert_allclose(agg, [2.0, 2.0])


def test_aggregate_uncertainties_small_cutoff_is_local():
    # With a cutoff below the 1 A separation, each atom sees only itself, so
    # the aggregate is just its own mean uncertainty.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cell = np.eye(3) * 20.0
    pbc = np.array([False, False, False])
    atomic_unc = np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])

    agg = payload.aggregate_uncertainties(
        pos, cell, pbc, atomic_unc, r_cut=0.5
    )

    np.testing.assert_allclose(agg, [3.0, 1.0])


# --- select_max_uncertainty_frames ----------------------------------------
def _frame(uncertainties, tag):
    at = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[9, 9, 9], pbc=True
    )
    at.arrays["uncertainty"] = np.asarray(uncertainties, dtype=float)
    at.info["tag"] = tag
    return at


def test_select_max_uncertainty_frames_picks_peak_per_segment():
    # 4 frames, 2 bins -> 2 frames per bin. In bin 0 the 2nd frame has the
    # largest single-atom uncertainty; in bin 1 the 1st frame does.
    structs = [
        _frame([1.0, 1.0], "b0f0"),
        _frame([5.0, 1.0], "b0f1"),  # peak of bin 0
        _frame([4.0, 1.0], "b1f0"),  # peak of bin 1
        _frame([2.0, 1.0], "b1f1"),
    ]

    selected = payload.select_max_uncertainty_frames(structs, bin_count=2)

    assert [s.info["tag"] for s in selected] == ["b0f1", "b1f0"]


def test_select_max_uncertainty_frames_one_bin_picks_global_peak():
    structs = [
        _frame([1.0, 2.0], "f0"),
        _frame([9.0, 1.0], "f1"),  # global peak
        _frame([3.0, 3.0], "f2"),
    ]

    selected = payload.select_max_uncertainty_frames(structs, bin_count=1)

    assert len(selected) == 1
    assert selected[0].info["tag"] == "f1"


# --- _build_committee_calculator dispatch ----------------------------------
def test_build_committee_calculator_rejects_unknown_backend():
    """The backend dispatch itself needs no MACE/ACE/GRACE stack to test."""
    with pytest.raises(ValueError, match="mace.*ace.*grace"):
        payload._build_committee_calculator(
            "quip", model_files=["m.pt"], device="cpu"
        )


class _FakeEnsembleCalculator(Calculator):
    """Stand-in for PyACEEnsembleCalculator/TPCalculator: a real ASE
    Calculator that reports an already-aggregated ``forces_std``, like the
    two real ACE/GRACE ensemble calculators do (unlike MACE's
    ``MACECalculator``, which reports per-member ``forces_comm`` instead --
    see ``_committee_force_stdev``). Deterministic but position-dependent, so
    uncertainty still varies frame to frame in the full-pipeline tests below.
    """

    implemented_properties = ["energy", "forces", "forces_std"]

    def __init__(self, model_paths):
        super().__init__()
        self.model_paths = model_paths

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        seed = int(abs(self.atoms.positions.sum()) * 1000) % (2**32)
        rng = np.random.default_rng(seed)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((n, 3))
        self.results["forces_std"] = rng.uniform(0.0, 1.0, (n, 3))


class _FakeACECalculator(_FakeEnsembleCalculator):
    """Matches ``PyACEEnsembleCalculator(model_paths)``'s positional arg."""


class _FakeGraceCalculator(_FakeEnsembleCalculator):
    """Matches ``TPCalculator(model=model_paths)``'s keyword arg."""

    def __init__(self, model):
        super().__init__(model)


def _install_fake_ace_module(monkeypatch):
    """Make ``from pyace import PyACEEnsembleCalculator`` resolve to the fake."""
    module = types.ModuleType("pyace")
    module.PyACEEnsembleCalculator = _FakeACECalculator
    monkeypatch.setitem(sys.modules, "pyace", module)


def _install_fake_grace_module(monkeypatch):
    """Make ``from tensorpotential.calculator.asecalculator import
    TPCalculator`` resolve to the fake, across all three dotted-path levels
    the real package would have."""
    tensorpotential = types.ModuleType("tensorpotential")
    calculator = types.ModuleType("tensorpotential.calculator")
    asecalculator = types.ModuleType(
        "tensorpotential.calculator.asecalculator"
    )
    asecalculator.TPCalculator = _FakeGraceCalculator
    calculator.asecalculator = asecalculator
    tensorpotential.calculator = calculator
    monkeypatch.setitem(sys.modules, "tensorpotential", tensorpotential)
    monkeypatch.setitem(sys.modules, "tensorpotential.calculator", calculator)
    monkeypatch.setitem(
        sys.modules, "tensorpotential.calculator.asecalculator", asecalculator
    )


_FAKE_BACKEND_INSTALLERS = {
    "ace": _install_fake_ace_module,
    "grace": _install_fake_grace_module,
}


@pytest.mark.parametrize("backend", ["ace", "grace"])
def test_build_committee_calculator_dispatches_ace_grace(monkeypatch, backend):
    _FAKE_BACKEND_INSTALLERS[backend](monkeypatch)
    model_files = ["member_a", "member_b"]

    calc = payload._build_committee_calculator(
        backend, model_files=model_files, device="cpu"
    )

    assert calc.model_paths == model_files


@pytest.mark.parametrize("backend", ["ace", "grace"])
def test_committee_force_stdev_reads_forces_std_not_forces_comm(backend):
    """Unlike MACE's ``forces_comm`` (stdev computed by this repo), ACE's and
    GRACE's ensemble calculators already aggregate one -- read straight
    through, not recomputed."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[9, 9, 9])
    calc = _FakeEnsembleCalculator(model_paths=["m.yaml"])

    stdev = payload._committee_force_stdev(backend, atoms, calc)

    np.testing.assert_array_equal(stdev, calc.results["forces_std"])


def _write_extxyz_trajectory(path, n_frames=4, box=8.0, seed=0):
    """Write a minimal multi-frame extxyz trajectory (already carrying real
    chemical symbols -- as ``lammps_al.run.scripts.lammps_sampling`` would
    have produced upstream of this payload).

    Four atoms per frame (Cu, Cu, Co, H), positions jittered per frame so the
    committee uncertainty varies.
    """
    import ase.io

    rng = np.random.default_rng(seed)
    base = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [1.2, 1.2, 1.2]]
    )
    frames = []
    for _ in range(n_frames):
        pos = base + rng.uniform(-0.3, 0.3, base.shape)
        atoms = Atoms("CuCuCoH", positions=pos, cell=[box, box, box], pbc=True)
        frames.append(atoms)
    ase.io.write(path, frames, format="extxyz")
    return path


@pytest.mark.parametrize("backend", ["ace", "grace"])
def test_annotate_uncertainties_scores_every_frame_ace_grace(
    tmp_path, monkeypatch, backend
):
    """Full annotation pipeline (read trajectory -> score -> aggregate ->
    write) for the ace/grace backends -- with fake calculators instead of a
    real trained committee (none exist yet for these backends). No
    reduction: every frame that went in comes back out."""
    _FAKE_BACKEND_INSTALLERS[backend](monkeypatch)

    import ase.io

    trajectory = _write_extxyz_trajectory(
        tmp_path / "trajectory.extxyz", n_frames=4
    )
    output = tmp_path / "annotated.extxyz"

    payload.annotate_uncertainties(
        trajectory_file=trajectory,
        model_files=["member_a", "member_b"],
        cutoff_radius=4.0,
        output_file=output,
        logger=logging.getLogger("test"),
        device="cpu",
        backend=backend,
    )

    annotated = ase.io.read(output, index=":")
    assert len(annotated) == 4
    for at in annotated:
        assert list(at.symbols) == ["Cu", "Cu", "Co", "H"]
        unc = at.arrays["uncertainty"]
        assert unc.shape == (4,)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)
