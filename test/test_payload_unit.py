"""Unit tests for the pure logic of the LAMMPS sampling payload.

These exercise the payload's dependency-light helpers -- jinja rendering,
local-uncertainty aggregation, and the segment/argmax candidate selection --
with only ase + numpy + jinja2. No remote_handler, no MACE stack, no LAMMPS
binary, no GPU. They are the fast, always-run tier that validates the sampling
*logic*; the MACE committee scoring and the real ``lmp`` run are covered in the
``lammps`` integration tier (see ``test_lammps_integration.py``).

Also covers the ``ace``/``grace`` branches of ``_build_committee_calculator``/
``_committee_force_stdev`` -- the only backend-specific pieces of the payload
(see its module docstring). No real tiny trained ACE/GRACE committee
potentials exist yet (unlike MACE's, borrowed from ``examples/mace_md_AL`` --
see ``test_committee_scoring.py``), so these use fake stand-in ASE
calculators that mimic ``pyace.PyACEEnsembleCalculator``'s/
``tensorpotential``'s ``TPCalculator``'s constructor signature and
already-aggregated ``forces_std`` result, installed into ``sys.modules`` so
the payload's real (lazy) ``from pyace import ...``/``from tensorpotential
... import ...`` statements resolve without either package installed. This
checks the *dispatch and result-key wiring*, not real ACE/GRACE inference --
promote these to real committee potentials (mirroring
``test_get_samples_from_dump_scores_and_selects`` in
``test_committee_scoring.py``) once tiny fitted ones are available.

The payload is imported by file path so the import never pulls in
``lammps_al.run.inputs`` (its ``remote_handler`` dependency is confined to
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
        / "run"
        / "scripts"
        / "lammps_sampling.py"
    )
    spec = importlib.util.spec_from_file_location("lammps_payload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


payload = _load_payload()


# --- parse_jinja_template --------------------------------------------------
def test_parse_jinja_template_renders_kwargs(tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "lammps.in.j2").write_text(
        "seed {{ seed }}\nread_data {{ data_file }}\n"
    )
    out = tmp_path / "lammps.in"

    payload.parse_jinja_template(
        out, template_dir, "lammps.in.j2", seed=42, data_file="struct.data"
    )

    text = out.read_text()
    assert "seed 42" in text
    assert "read_data struct.data" in text


def test_ace_template_dump_frequency_is_rendered_not_hardcoded():
    """Regression test for the real campaign template: the dump line must
    read ``{{ dump_frequency }}``, not a hardcoded ``1`` -- confirmed as the
    cause of an unthinned, >20GB dump file in ~10 minutes on the real
    206941-atom campaign system, 2026-09-11 (dump_frequency was silently
    dropped upstream as an "unknown key" and never reached the template at
    all).

    Renders a git-tracked copy of the real shipped template
    (test/data/lammps/lammps_ace.in.template), not a synthetic stand-in --
    the actual campaign copy under experiments/rough_cu_water_al/ is
    gitignored (campaign-specific working files), so it doesn't exist in a
    fresh CI clone and can't be read directly here. Keep the two in sync by
    hand; both files cross-reference each other in their header comments.
    """
    template_dir = Path(__file__).resolve().parent / "data" / "lammps"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lammps.in"
        payload.parse_jinja_template(
            out,
            template_dir,
            "lammps_ace.in.template",
            model_file="model.yaml",
            data_file="struct.data",
            lammps_pair_style="pace product",
            seed=42,
            dump_file="traj.dump",
            dump_frequency=20,
            final_data_file="final.data",
        )
        text = out.read_text()

    dump_lines = [line for line in text.splitlines() if "dump" in line]
    assert any(
        "dump" in line and "d_nvt" in line and "20" in line
        for line in dump_lines
    ), f"expected a dump line with frequency 20, got: {dump_lines}"
    assert not any(
        "dump" in line and "d_nvt" in line and " 1 " in line
        for line in dump_lines
    ), "dump frequency looks hardcoded to 1 again"


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


def _write_dump(path, n_frames=4, box=8.0, seed=0):
    """Write a minimal multi-frame LAMMPS ``lammps-dump-text`` trajectory.

    Four atoms per frame with types 1,1,2,3 (Cu,Cu,Co,H after remapping),
    positions jittered per frame so the committee uncertainty varies.
    """
    rng = np.random.default_rng(seed)
    base = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [1.2, 1.2, 1.2]]
    )
    types_ = [1, 1, 2, 3]
    blocks = []
    for step in range(n_frames):
        pos = base + rng.uniform(-0.3, 0.3, base.shape)
        lines = [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(len(types_)),
            "ITEM: BOX BOUNDS pp pp pp",
            f"0.0 {box}",
            f"0.0 {box}",
            f"0.0 {box}",
            "ITEM: ATOMS id type x y z",
        ]
        for i, (t, p) in enumerate(zip(types_, pos), start=1):
            lines.append(f"{i} {t} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n")
    return path


@pytest.mark.parametrize("backend", ["ace", "grace"])
def test_get_samples_from_dump_scores_and_selects_ace_grace(
    tmp_path, monkeypatch, backend
):
    """Full pipeline (parse dump -> score -> aggregate -> select -> write)
    for the ace/grace backends, mirroring
    ``test_get_samples_from_dump_scores_and_selects`` in
    ``test_committee_scoring.py`` -- with fake calculators instead of a real
    trained committee (none exist yet for these backends)."""
    _FAKE_BACKEND_INSTALLERS[backend](monkeypatch)

    import ase.io

    dump = _write_dump(tmp_path / "md.dump", n_frames=4)
    samples = tmp_path / "samples.xyz"
    full = tmp_path / "full.xyz"
    element_mapping = {"1": "Cu", "2": "Co", "3": "H"}

    payload.get_samples_from_dump(
        trajectory_file=dump,
        model_files=["member_a", "member_b"],
        element_mapping=element_mapping,
        cutoff_radius=4.0,
        bin_count=2,
        samples_file=samples,
        logger=logging.getLogger("test"),
        output_full_uncertainties_traj=True,
        full_file=full,
        device="cpu",
        backend=backend,
    )

    selected = ase.io.read(samples, index=":")
    assert len(selected) == 2
    for at in selected:
        assert list(at.symbols) == ["Cu", "Cu", "Co", "H"]

    annotated = ase.io.read(full, index=":")
    assert len(annotated) == 4
    for at in annotated:
        unc = at.arrays["uncertainty"]
        assert unc.shape == (4,)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)
