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


def test_annotate_uncertainties_scores_every_frame_grace(
    tmp_path, monkeypatch
):
    """Full annotation pipeline (read trajectory -> score -> aggregate ->
    write) for the grace backend (still routed through
    _build_committee_calculator/_committee_force_stdev, unlike ace -- see
    test_ace_backend_uses_tensorpotential_not_pyace_ensemble below) -- with
    a fake calculator instead of a real trained committee (none exist yet
    for this backend). No reduction: every frame that went in comes back
    out."""
    _FAKE_BACKEND_INSTALLERS["grace"](monkeypatch)

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
        backend="grace",
    )

    annotated = ase.io.read(output, index=":")
    assert len(annotated) == 4
    for at in annotated:
        assert list(at.symbols) == ["Cu", "Cu", "Co", "H"]
        unc = at.arrays["uncertainty"]
        assert unc.shape == (4,)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)


# --- backend="ace": tensorpotential (GPU-capable), not PyACEEnsembleCalculator
def _install_fake_ace_tensorpotential_stack(monkeypatch, member_forces):
    """Fake pyace.BBasisConfiguration + tensorpotential's ACE/TensorPotential
    + matscipy's neighbour_list, so the tensorpotential-based ace path
    (_build_ace_tensorpotential_models / _tp_atoms_from_matscipy /
    _ace_committee_force_stdev) can be exercised without any of those
    packages installed.

    Args:
        member_forces: list of (n_atoms, 3) arrays, one per committee
            member -- each fake TensorPotential.evaluate() returns the
            corresponding array (tiled per frame via e_map), so the test
            can assert the exact resulting stdev.
    """
    pyace_module = types.ModuleType("pyace")

    class _FakeBBasisConfiguration:
        def __init__(self, path):
            self.path = path

    pyace_module.BBasisConfiguration = _FakeBBasisConfiguration
    monkeypatch.setitem(sys.modules, "pyace", pyace_module)

    tensorpotential = types.ModuleType("tensorpotential")
    potentials = types.ModuleType("tensorpotential.potentials")
    potentials_ace = types.ModuleType("tensorpotential.potentials.ace")
    tensorpot_mod = types.ModuleType("tensorpotential.tensorpot")
    utils = types.ModuleType("tensorpotential.utils")
    utilities = types.ModuleType("tensorpotential.utils.utilities")

    class _FakeACE:
        def __init__(self, bbasisconfig, **kwargs):
            self.bbasisconfig = bbasisconfig

    # One fake TensorPotential per member_forces entry, constructed in the
    # same order _build_ace_tensorpotential_models builds them (one per
    # model_files entry) -- a mutable counter closes over member_forces so
    # each successive construction picks the next member's fixed forces.
    construction_order = {"i": 0}

    class _FakeTensorPotential:
        def __init__(self, potential):
            self.potential = potential
            self.forces = member_forces[construction_order["i"]]
            construction_order["i"] += 1

        def evaluate(self, batch):
            n_frames = len(np.unique(batch["e_map"]))
            forces = np.tile(self.forces, (n_frames, 1))
            energy = np.zeros((n_frames,))
            return energy, forces, None

    gpu_config_calls = []

    def _fake_init_gpu_config(config):
        gpu_config_calls.append(config)

    def _fake_supply_batch(tp_atoms_list):
        e_map = np.concatenate(
            [
                np.full(len(d["_positions"]), j)
                for j, d in enumerate(tp_atoms_list)
            ]
        )
        return {"e_map": e_map}

    potentials_ace.ACE = _FakeACE
    tensorpot_mod.TensorPotential = _FakeTensorPotential
    utilities.init_gpu_config = _fake_init_gpu_config
    utilities.supply_batch = _fake_supply_batch
    potentials.ace = potentials_ace
    tensorpotential.potentials = potentials
    tensorpotential.tensorpot = tensorpot_mod
    tensorpotential.utils = utils
    utils.utilities = utilities

    monkeypatch.setitem(sys.modules, "tensorpotential", tensorpotential)
    monkeypatch.setitem(sys.modules, "tensorpotential.potentials", potentials)
    monkeypatch.setitem(
        sys.modules, "tensorpotential.potentials.ace", potentials_ace
    )
    monkeypatch.setitem(
        sys.modules, "tensorpotential.tensorpot", tensorpot_mod
    )
    monkeypatch.setitem(sys.modules, "tensorpotential.utils", utils)
    monkeypatch.setitem(
        sys.modules, "tensorpotential.utils.utilities", utilities
    )

    matscipy_module = types.ModuleType("matscipy")
    neighbours = types.ModuleType("matscipy.neighbours")

    def _fake_neighbour_list(quantities, atoms, cutoff):
        assert quantities == "ijS"
        n = len(atoms)
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = i != j
        ind_i = i[mask].astype(np.int32)
        ind_j = j[mask].astype(np.int32)
        offsets = np.zeros((len(ind_i), 3), dtype=np.float64)
        return ind_i, ind_j, offsets

    neighbours.neighbour_list = _fake_neighbour_list
    matscipy_module.neighbours = neighbours
    monkeypatch.setitem(sys.modules, "matscipy", matscipy_module)
    monkeypatch.setitem(sys.modules, "matscipy.neighbours", neighbours)

    return gpu_config_calls


def test_build_ace_tensorpotential_models_one_per_file_and_gpu_index(
    monkeypatch,
):
    """One TensorPotential per model file, in order; init_gpu_config's
    gpu_index follows device ("cuda" -> 0, "cpu" -> -1)."""
    # 3 entries: 2 for the first call below, 1 for the second -- the fake's
    # construction-order counter is shared across both calls in this test.
    gpu_config_calls = _install_fake_ace_tensorpotential_stack(
        monkeypatch, member_forces=[np.zeros((1, 3))] * 3
    )

    models = payload._build_ace_tensorpotential_models(
        ["m0.yaml", "m1.yaml"], device="cuda"
    )
    assert len(models) == 2
    assert models[0].potential.bbasisconfig.path == "m0.yaml"
    assert models[1].potential.bbasisconfig.path == "m1.yaml"
    assert gpu_config_calls == [{"gpu_index": 0}]

    gpu_config_calls.clear()
    payload._build_ace_tensorpotential_models(["m0.yaml"], device="cpu")
    assert gpu_config_calls == [{"gpu_index": -1}]


def test_tp_atoms_from_matscipy_shapes(monkeypatch):
    """The dict shape supply_batch/TensorPotential.evaluate need, built
    from matscipy's neighbour_list output."""
    _install_fake_ace_tensorpotential_stack(
        monkeypatch, member_forces=[np.zeros((1, 3))]
    )
    atoms = Atoms(
        "H3",
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[9, 9, 9],
        pbc=True,
    )

    tp_atoms = payload._tp_atoms_from_matscipy(atoms, cutoff=4.0)

    n_pairs = 3 * 2  # 3 atoms, all-pairs excluding self (fake neighbour_list)
    assert tp_atoms["_ind_i"].shape == (n_pairs,)
    assert tp_atoms["_ind_j"].shape == (n_pairs,)
    assert tp_atoms["_mu_i"].shape == (n_pairs,)
    assert tp_atoms["_mu_j"].shape == (n_pairs,)
    assert tp_atoms["_offsets"].shape == (n_pairs, 3)
    assert tp_atoms["_positions"].shape == (3, 3)
    assert tp_atoms["_cell"].shape == (1, 3, 3)
    assert tp_atoms["_forces"].shape == (3, 3)
    assert np.all(tp_atoms["_forces"] == 0.0)


def test_ace_committee_force_stdev_matches_manual_std(monkeypatch):
    """The returned stdev is exactly np.std(stack(member_forces), axis=0,
    ddof=1) -- the same ensemble-spread math the mace branch of
    _committee_force_stdev already uses, just computed across separately-
    evaluated TensorPotential members instead of one calculator's
    forces_comm."""
    forces_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    forces_b = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    _install_fake_ace_tensorpotential_stack(
        monkeypatch, member_forces=[forces_a, forces_b]
    )
    models = payload._build_ace_tensorpotential_models(
        ["m0.yaml", "m1.yaml"], device="cpu"
    )
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[9, 9, 9], pbc=True
    )

    stdev = payload._ace_committee_force_stdev(
        models, atoms, cutoff_radius=4.0
    )

    expected = np.std(np.stack([forces_a, forces_b]), axis=0, ddof=1)
    np.testing.assert_allclose(stdev, expected)


def test_annotate_uncertainties_ace_backend_uses_tensorpotential(
    tmp_path, monkeypatch
):
    """backend="ace" must route through _build_ace_tensorpotential_models /
    _ace_committee_force_stdev (tensorpotential, GPU-capable), NOT
    _build_committee_calculator's PyACEEnsembleCalculator path (CPU-only,
    confirmed unusably slow on the real campaign system) -- see the module
    docstring and experiments/rough_cu_water_al/build/tp_ace_diag/
    RESULTS.md."""
    n_atoms = 4
    _install_fake_ace_tensorpotential_stack(
        monkeypatch,
        member_forces=[
            np.zeros((n_atoms, 3)),
            np.ones((n_atoms, 3)),
        ],
    )

    import ase.io

    trajectory = _write_extxyz_trajectory(
        tmp_path / "trajectory.extxyz", n_frames=3
    )
    output = tmp_path / "annotated.extxyz"

    payload.annotate_uncertainties(
        trajectory_file=trajectory,
        model_files=["member_a.yaml", "member_b.yaml"],
        cutoff_radius=4.0,
        output_file=output,
        logger=logging.getLogger("test"),
        device="cpu",
        backend="ace",
    )

    annotated = ase.io.read(output, index=":")
    assert len(annotated) == 3
    for at in annotated:
        unc = at.arrays["uncertainty"]
        assert unc.shape == (n_atoms,)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)


# --- _fix_tensorflow_gpu_libs -----------------------------------------------
def test_fix_tensorflow_gpu_libs_noop_when_already_fixed(monkeypatch):
    """The sentinel env var short-circuits before touching sys.path/execv
    at all."""
    monkeypatch.setenv("_TP_GPU_LIBS_FIXED", "1")

    def _boom(*a, **k):
        raise AssertionError("os.execv must not be called when already fixed")

    monkeypatch.setattr(payload.os, "execv", _boom)

    payload._fix_tensorflow_gpu_libs()  # must return, not raise


def test_fix_tensorflow_gpu_libs_noop_when_no_nvidia_dirs(monkeypatch):
    """No nvidia/*/lib directories found -> no-op, no execv."""
    monkeypatch.delenv("_TP_GPU_LIBS_FIXED", raising=False)
    monkeypatch.setattr(payload.glob, "glob", lambda *a, **k: [])

    def _boom(*a, **k):
        raise AssertionError("os.execv must not be called with no nvidia dirs")

    monkeypatch.setattr(payload.os, "execv", _boom)

    payload._fix_tensorflow_gpu_libs()


def test_fix_tensorflow_gpu_libs_reexecs_with_fixed_path(monkeypatch):
    """When nvidia/*/lib dirs are found and not yet fixed, it sets
    LD_LIBRARY_PATH and re-execs -- mocking os.execv so this test doesn't
    actually replace the pytest process."""
    monkeypatch.delenv("_TP_GPU_LIBS_FIXED", raising=False)
    monkeypatch.setattr(
        payload.glob, "glob", lambda *a, **k: ["/fake/nvidia/cudnn/lib"]
    )
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    execv_calls = []
    monkeypatch.setattr(payload.os, "execv", lambda *a: execv_calls.append(a))

    payload._fix_tensorflow_gpu_libs()

    assert len(execv_calls) == 1
    assert payload.os.environ["LD_LIBRARY_PATH"].startswith(
        "/fake/nvidia/cudnn/lib:"
    )
    assert payload.os.environ["_TP_GPU_LIBS_FIXED"] == "1"
