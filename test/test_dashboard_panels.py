"""Tests for the ACE-committee dashboard panels.

``CommitteeParityPanel``/``HeldOutParityPanel``/``SpeciesForceParityPanel``
patch ``score_with_committee`` at the point each panel module imported it
(the standard way to seam out a dependency that has no per-call injection
parameter of its own) so no real ``pyace`` is needed. The uncertainty panels
need no committee at all -- they read per-atom ``uncertainty`` arrays
directly off synthetic sampler trajectories.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

ase_io = pytest.importorskip("ase.io")
# NOT "alviz" bare: some environments have a stray sys.path entry that
# resolves the bare name to an empty namespace package even with the real
# alviz uninstalled (confirmed locally), so the actual submodule needs to
# be the one checked, or this "skip" never fires and a later import fails
# for real instead.
pytest.importorskip("alviz.dashboard")

from alviz.dashboard import Dashboard  # noqa: E402

from lammps_al.dashboard.panels.committee_parity import (  # noqa: E402
    CommitteeParityPanel,
    HeldOutParityPanel,
)
from lammps_al.dashboard.panels.cross_iteration import (  # noqa: E402
    CrossIterationErrorPanel,
)
from lammps_al.dashboard.panels.error_analysis import (  # noqa: E402
    ErrorVsROIDistancePanel,
    UncertaintyCalibrationPanel,
)
from lammps_al.dashboard.panels.pipeline import (  # noqa: E402
    PipelineAttritionPanel,
)
from lammps_al.dashboard.panels.species_parity import (  # noqa: E402
    SpeciesForceParityPanel,
    SpeciesParityData,
)
from lammps_al.dashboard.panels.training_fit import (  # noqa: E402
    TrainingParityPanel,
)
from lammps_al.dashboard.panels.uncertainty import (  # noqa: E402
    UncertaintyDepthProfilePanel,
    UncertaintyTimelinePanel,
    UncertaintyTrendPanel,
)

pytestmark = pytest.mark.local


def _make_config(rng, e_ref, species=("Cu", "H", "O"), n_each=1):
    symbols = [s for s in species for _ in range(n_each)]
    n = len(symbols)
    atoms = Atoms(
        symbols,
        positions=rng.uniform(0, 5, size=(n, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    atoms.info["REF_energy"] = e_ref
    atoms.arrays["REF_forces"] = rng.normal(size=(n, 3))
    return atoms


def _fake_score_with_committee(model_files, atoms_list, **kwargs):
    """Deterministic stand-in: predicted == reference + a fixed offset."""
    energies = [float(atoms.info["REF_energy"]) + 0.1 for atoms in atoms_list]
    forces = None
    if kwargs.get("want_forces", True):
        forces = [
            np.asarray(atoms.arrays["REF_forces"]) + 0.01
            for atoms in atoms_list
        ]
    return energies, forces


# ---------------------------------------------------------------------------
# CommitteeParityPanel / HeldOutParityPanel
# ---------------------------------------------------------------------------
@pytest.fixture
def evaluated_extxyz(tmp_path):
    rng = np.random.default_rng(0)
    configs = [_make_config(rng, e_ref=-10.0 - i) for i in range(3)]
    path = tmp_path / "evaluated.extxyz"
    ase_io.write(path, configs)
    return path


def test_committee_parity_panel(monkeypatch, tmp_path, evaluated_extxyz):
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.committee_parity.score_with_committee",
        _fake_score_with_committee,
    )
    panel = CommitteeParityPanel(
        [evaluated_extxyz],
        ["m0.yaml", "m1.yaml"],
        per_atom=False,
        datapath=tmp_path,
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert len(panel.data.e_obs) == 3
    assert panel.data.e_rmse == pytest.approx(0.1)
    for ax in (panel.ax_e, panel.ax_f):
        assert all(c.get_rasterized() for c in ax.collections)


def test_committee_parity_panel_no_configs_raises(tmp_path):
    panel = CommitteeParityPanel([], ["m.yaml"], datapath=tmp_path)
    with pytest.raises(ValueError):
        panel.compute_data({})


def _held_out_files(tmp_path, configs0, configs1):
    path0 = tmp_path / "evaluator_it0_0.extxyz"
    path1 = tmp_path / "evaluator_it1_0.extxyz"
    ase_io.write(path0, configs0)
    ase_io.write(path1, configs1)
    return [path1], [path0]


def test_held_out_parity_panel_cumulative_uses_new_configs_only(
    monkeypatch, tmp_path
):
    """iter 1's file contains iter 0's frames as a prefix plus 2 new ones."""
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.committee_parity.score_with_committee",
        _fake_score_with_committee,
    )
    rng = np.random.default_rng(1)
    configs0 = [_make_config(rng, e_ref=-1.0)]
    new_configs = [
        _make_config(rng, e_ref=-2.0),
        _make_config(rng, e_ref=-3.0),
    ]
    current, previous = _held_out_files(
        tmp_path, configs0, configs0 + new_configs
    )

    panel = HeldOutParityPanel(
        current, previous, ["m0.yaml"], per_atom=False, datapath=tmp_path
    )
    data = panel.compute_data({})

    # only the 2 new-to-iteration-1 configs
    assert data.e_obs == pytest.approx([-2.0, -3.0])


def test_held_out_parity_panel_delta_uses_all_configs(monkeypatch, tmp_path):
    """iter 1's file is unrelated to iter 0's -- treated as all-new."""
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.committee_parity.score_with_committee",
        _fake_score_with_committee,
    )
    rng = np.random.default_rng(2)
    configs0 = [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    configs1 = [_make_config(rng, e_ref=-5.0), _make_config(rng, e_ref=-6.0)]
    current, previous = _held_out_files(tmp_path, configs0, configs1)

    panel = HeldOutParityPanel(
        current, previous, ["m0.yaml"], per_atom=False, datapath=tmp_path
    )
    data = panel.compute_data({})

    assert data.e_obs == pytest.approx([-5.0, -6.0])


def test_held_out_parity_panel_no_previous_configs_uses_all(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.committee_parity.score_with_committee",
        _fake_score_with_committee,
    )
    rng = np.random.default_rng(3)
    path = tmp_path / "evaluator_it1_0.extxyz"
    ase_io.write(
        path, [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    )

    panel = HeldOutParityPanel([path], [], ["m0.yaml"], datapath=tmp_path)
    data = panel.compute_data({})

    assert len(data.e_obs) == 2


def test_held_out_parity_panel_no_configs_raises(tmp_path):
    panel = HeldOutParityPanel([], [], ["m.yaml"], datapath=tmp_path)
    with pytest.raises(ValueError):
        panel.compute_data({})


# ---------------------------------------------------------------------------
# SpeciesForceParityPanel
# ---------------------------------------------------------------------------
def test_species_force_parity_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.species_parity.score_with_committee",
        _fake_score_with_committee,
    )
    rng = np.random.default_rng(2)
    configs = [_make_config(rng, e_ref=-5.0, n_each=2)]
    path = tmp_path / "cfg.extxyz"
    ase_io.write(path, configs)

    panel = SpeciesForceParityPanel(
        [path], ["m.yaml"], ["Cu", "H", "O"], datapath=tmp_path
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert isinstance(panel.data, SpeciesParityData)
    for sym in ("Cu", "H", "O"):
        assert len(panel.data.f_obs[sym]) == 2 * 3  # 2 atoms x 3 components
        assert panel.data.f_rmse[sym] == pytest.approx(0.01)


def test_species_force_parity_panel_missing_species_gets_none(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.species_parity.score_with_committee",
        _fake_score_with_committee,
    )
    rng = np.random.default_rng(3)
    configs = [_make_config(rng, e_ref=-1.0, species=("Cu",), n_each=2)]
    path = tmp_path / "cu_only.extxyz"
    ase_io.write(path, configs)

    panel = SpeciesForceParityPanel(
        [path], ["m.yaml"], ["Cu", "H", "O"], datapath=tmp_path
    )
    data = panel.compute_data({})

    assert data.f_obs["H"] == []
    assert data.f_rmse["H"] is None
    assert data.f_rmse["Cu"] is not None


# ---------------------------------------------------------------------------
# UncertaintyTrendPanel / UncertaintyDepthProfilePanel
# ---------------------------------------------------------------------------
def _write_uncertainty_traj(tmp_path, it, frames):
    """frames: list of (positions_z_for_Cu_H_O_triplet, uncertainties).

    Returns the written path -- the shape of one
    ``uncertainty_select_output`` entry (annotated, not reduced).
    """
    configs = []
    for z_positions, uncertainties in frames:
        atoms = Atoms(
            "CuHO",
            positions=[[0.0, 0.0, z] for z in z_positions],
            cell=np.eye(3) * 20,
            pbc=True,
        )
        atoms.arrays["uncertainty"] = np.asarray(uncertainties)
        configs.append(atoms)
    path = tmp_path / f"u_it{it}_0.extxyz"
    ase_io.write(path, configs)
    return path


def _render(panel):
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()


def test_uncertainty_trend_panel(tmp_path):
    trajectories = {
        # Deliberately out of order: the panel sorts by iteration.
        1: _write_uncertainty_traj(
            tmp_path, 1, [([0.0, 1.0, 2.0], [0.01, 0.02, 0.03])]
        ),
        0: _write_uncertainty_traj(
            tmp_path, 0, [([0.0, 1.0, 2.0], [0.1, 0.2, 0.3])]
        ),
    }

    panel = UncertaintyTrendPanel(trajectories, datapath=tmp_path)
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert panel.data.iterations == [0, 1]
    assert panel.data.mean[0] == pytest.approx(0.2)
    assert panel.data.mean[1] == pytest.approx(0.02)
    assert panel.data.mean[1] < panel.data.mean[0]  # converging


def test_uncertainty_trend_panel_no_data_raises(tmp_path):
    panel = UncertaintyTrendPanel({}, datapath=tmp_path)
    with pytest.raises(ValueError):
        panel.compute_data({})


def test_uncertainty_depth_profile_panel_bins_by_depth(tmp_path):
    # Cu at z=0 (interface z0=0); H at z=-5 (inside slab); O at z=5 (in water).
    path = _write_uncertainty_traj(
        tmp_path, 0, [([0.0, -5.0, 5.0], [0.0, 0.5, 1.0])]
    )

    # bin_width=2.0 puts depth=-5.0/+5.0 exactly on bin centers (-9,-7,-5,
    # -3,-1,1,3,5,7,9), avoiding ambiguous edge-of-bin assignment.
    panel = UncertaintyDepthProfilePanel(
        {0: path},
        depth_range=(-10.0, 10.0),
        bin_width=2.0,
        datapath=tmp_path,
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert panel.data.iterations == [0]
    profile = panel.data.profiles["0"]
    centers = np.asarray(panel.data.depth_bin_centers)

    # Bin closest to depth=-5 (H, uncertainty=0.5) and depth=+5 (O, 1.0).
    idx_neg5 = int(np.argmin(np.abs(centers - (-5.0))))
    idx_pos5 = int(np.argmin(np.abs(centers - 5.0)))
    assert profile[idx_neg5] == pytest.approx(0.5)
    assert profile[idx_pos5] == pytest.approx(1.0)


def test_uncertainty_depth_profile_panel_no_interface_species_skipped(
    tmp_path,
):
    path = _write_uncertainty_traj(
        tmp_path, 0, [([0.0, 1.0, 2.0], [0.1, 0.2, 0.3])]
    )

    # interface_species="Ag" never appears -- frame has no interface atoms.
    panel = UncertaintyDepthProfilePanel(
        {0: path},
        interface_species=("Ag",),
        datapath=tmp_path,
    )
    with pytest.raises(ValueError):
        panel.compute_data({})


# ---------------------------------------------------------------------------
# UncertaintyTimelinePanel
# ---------------------------------------------------------------------------
def test_uncertainty_timeline_panel_locates_rois(tmp_path):
    configs = []
    for step, unc in ((0, [0.1, 0.2, 0.3]), (50, [0.4, 0.9, 0.5])):
        atoms = Atoms(
            "CuHO", positions=np.zeros((3, 3)), cell=np.eye(3) * 20, pbc=True
        )
        atoms.arrays["uncertainty"] = np.asarray(unc)
        atoms.info["timestep"] = step
        configs.append(atoms)
    traj = tmp_path / "u.extxyz"
    ase_io.write(traj, configs)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"index": 0, "frame": 1, "roi_index": 1}]))

    panel = UncertaintyTimelinePanel(
        traj, manifest_path=manifest, datapath=tmp_path
    )
    _render(panel)

    assert panel.data.step == [0.0, 50.0]
    assert panel.data.max == pytest.approx([0.3, 0.9])
    assert panel.data.mean == pytest.approx([0.2, 0.6])
    assert panel.data.roi_step == [50.0]
    assert panel.data.roi_uncertainty == pytest.approx([0.9])


# ---------------------------------------------------------------------------
# CrossIterationErrorPanel
# ---------------------------------------------------------------------------
def test_cross_iteration_error_panel_matrix(monkeypatch, tmp_path):
    rng = np.random.default_rng(5)
    initial = tmp_path / "initial.xyz"
    ase_io.write(initial, [_make_config(rng, e_ref=-1.0) for _ in range(4)])
    it1 = [_make_config(rng, e_ref=-2.0) for _ in range(2)]
    it2_new = [_make_config(rng, e_ref=-3.0)]
    path1 = tmp_path / "evaluator_it1_0.extxyz"
    path2 = tmp_path / "evaluator_it2_0.extxyz"
    ase_io.write(path1, it1)
    # Cumulative output: only the trailing frame is new to iteration 2.
    ase_io.write(path2, it1 + it2_new)

    def fake(model_files, atoms_list, **kwargs):
        # Committee "itN" is off by N+1 meV/atom in energy, (N+1)*0.01 in
        # every force component.
        k = int(Path(model_files[0]).stem[-1]) + 1
        energies = [
            float(a.info["REF_energy"]) + 1e-3 * k * len(a) for a in atoms_list
        ]
        forces = [
            np.asarray(a.arrays["REF_forces"]) + 0.01 * k for a in atoms_list
        ]
        return energies, forces

    monkeypatch.setattr(
        "lammps_al.dashboard.panels.cross_iteration.score_with_committee",
        fake,
    )
    panel = CrossIterationErrorPanel(
        {0: ["m_it0.yaml"], 1: ["m_it1.yaml"], 2: ["m_it2.yaml"]},
        {1: [path1], 2: [path2]},
        initial_database=initial,
        max_initial_frames=3,
        datapath=tmp_path,
    )
    _render(panel)

    d = panel.data
    assert d.model_iterations == [0, 1, 2]
    assert d.block_labels == ["init", "it1", "it2"]
    assert d.block_iterations == [0, 1, 2]
    assert d.block_sizes == [3, 2, 1]
    for r, k in enumerate((1, 2, 3)):
        assert d.e_rmse[r] == pytest.approx([k] * 3)
        assert d.f_rmse[r] == pytest.approx([0.01 * k] * 3)
        # Constant offsets: MAE equals RMSE.
        assert d.e_mae[r] == pytest.approx([k] * 3)
        assert d.f_mae[r] == pytest.approx([0.01 * k] * 3)


def test_cross_iteration_error_panel_recomputes_cache_without_mae(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(6)
    path = tmp_path / "evaluator_it1_0.extxyz"
    ase_io.write(path, [_make_config(rng, e_ref=-2.0) for _ in range(2)])
    stale = {
        "model_iterations": [1],
        "block_labels": ["it1"],
        "block_iterations": [1],
        "block_sizes": [2],
        "e_rmse": [[99.0]],
        "f_rmse": [[99.0]],
    }
    (tmp_path / "cross_iteration_errors.json").write_text(json.dumps(stale))
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.cross_iteration.score_with_committee",
        lambda model_files, atoms_list, **kw: (
            [float(a.info["REF_energy"]) for a in atoms_list],
            [np.asarray(a.arrays["REF_forces"]) for a in atoms_list],
        ),
    )
    panel = CrossIterationErrorPanel(
        {1: ["m_it1.yaml"]},
        {1: [path]},
        initial_database=None,
        datapath=tmp_path,
    )
    _render(panel)

    assert panel.data.e_rmse == [[pytest.approx(0.0)]]
    assert panel.data.e_mae == [[pytest.approx(0.0)]]


def test_cross_iteration_error_panel_no_data_raises(tmp_path):
    panel = CrossIterationErrorPanel(
        {0: ["m.yaml"]}, {}, initial_database=None, datapath=tmp_path
    )
    with pytest.raises(ValueError):
        panel.compute_data({})


# ---------------------------------------------------------------------------
# PipelineAttritionPanel
# ---------------------------------------------------------------------------
def test_pipeline_attrition_panel_counts_and_names_losses(tmp_path):
    rng = np.random.default_rng(4)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"index": i} for i in range(3)]))
    ok_log = tmp_path / "ok.log"
    ok_log.write_text("Step Temp\n0 300\n")
    bad_log = tmp_path / "bad.log"
    bad_log.write_text(
        "Step Temp\n80000 5311\nERROR on proc 0: Particle outside "
        "surface of region\n"
    )
    configs = tmp_path / "evaluated.extxyz"
    ase_io.write(configs, [_make_config(rng, e_ref=-1.0)])

    panel = PipelineAttritionPanel(
        {
            1: {
                "selector_manifest": manifest,
                "annealer_boxes": [
                    ("0", True, ok_log),
                    ("1", False, bad_log),
                    ("2", True, ok_log),
                ],
                "dft_calcs": [("0", True), ("1", False)],
                "config_files": [configs],
                "prev_config_files": [],
            },
            2: {
                "selector_manifest": manifest,
                "annealer_boxes": [],
                "dft_calcs": [],
                "config_files": [],
                "prev_config_files": [],
            },
        },
        datapath=tmp_path,
    )
    _render(panel)

    assert panel.data.counts["1"] == [3, 3, 2, 2, 1, 1]
    # Stages that never ran are None, not 0.
    assert panel.data.counts["2"] == [3, None, None, None, None, None]
    assert panel.data.notes == [
        "it1 anneal box 1: ERROR on proc 0: Particle outside surface of "
        "region",
        "it1 DFT calcdir 1: not finished",
    ]


# ---------------------------------------------------------------------------
# UncertaintyCalibrationPanel / ErrorVsROIDistancePanel
# ---------------------------------------------------------------------------
def _labelled(positions, cell=20.0):
    atoms = Atoms(
        "H" * len(positions),
        positions=positions,
        cell=np.eye(3) * cell,
        pbc=True,
    )
    atoms.info["REF_energy"] = -1.0
    atoms.arrays["REF_forces"] = np.zeros((len(positions), 3))
    return atoms


def _patch_spread(monkeypatch, spread_of, error_of):
    """Fake committee: per-atom spread/error as functions of position."""
    calls = []

    def fake(model_files, atoms_list, **kwargs):
        calls.append(len(atoms_list))
        forces, spread = [], []
        for atoms in atoms_list:
            e = error_of(atoms.positions)
            forces.append(atoms.arrays["REF_forces"] + e[:, None])
            spread.append(np.repeat(spread_of(atoms.positions)[:, None], 3, 1))
        return forces, spread

    monkeypatch.setattr(
        "lammps_al.dashboard.panels.error_analysis.committee_force_spread",
        fake,
    )
    return calls


def test_uncertainty_calibration_panel_perfectly_ranked(monkeypatch, tmp_path):
    # Error grows with spread -> Spearman rho == 1.
    _patch_spread(
        monkeypatch,
        spread_of=lambda pos: 0.01 * (1 + pos[:, 0]),
        error_of=lambda pos: 0.02 * (1 + pos[:, 0]),
    )
    path = tmp_path / "evaluated.extxyz"
    ase_io.write(
        path, [_labelled([[float(x), 0.0, 0.0] for x in range(0, 12, 2)])]
    )

    # cutoff_radius below the 2 Å spacing: aggregation is the atom itself.
    panel = UncertaintyCalibrationPanel(
        [path], [], ["m.yaml"], cutoff_radius=0.5, nbins=3, datapath=tmp_path
    )
    _render(panel)

    assert len(panel.data.error) == 6
    assert panel.data.spearman == pytest.approx(1.0)
    assert len(panel.data.bin_uncertainty) == 3


def test_error_vs_roi_distance_panel_bins_with_mic(monkeypatch, tmp_path):
    _patch_spread(
        monkeypatch,
        spread_of=lambda pos: np.full(len(pos), 0.05),
        error_of=lambda pos: np.where(pos[:, 0] < 0, 0.8, 0.1),
    )
    center = [5.0, 5.0, 5.0]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"index": 0, "roi_position": center}] * 2))
    path = tmp_path / "evaluated.extxyz"
    # 0.5 Å and 2.5 Å from the center, plus an unwrapped atom at x=-6.5
    # that is 11.5 Å away naively but 8.5 Å under the minimum image (20 Å
    # cell).
    ase_io.write(
        path,
        [_labelled([[5.5, 5.0, 5.0], [7.5, 5.0, 5.0], [-6.5, 5.0, 5.0]])],
    )

    panel = ErrorVsROIDistancePanel(
        [path], [], ["m.yaml"], manifest, bin_width=1.0, datapath=tmp_path
    )
    _render(panel)

    d = panel.data
    by_bin = {
        c: (n, e)
        for c, n, e in zip(d.bin_centers, d.counts, d.error_median)
        if n
    }
    assert by_bin == {
        0.5: (1, pytest.approx(0.1 * np.sqrt(3))),
        2.5: (1, pytest.approx(0.1 * np.sqrt(3))),
        8.5: (1, pytest.approx(0.8 * np.sqrt(3))),
    }
    assert max(d.bin_centers) < 9  # no 12 Å bin: mic was applied


def test_error_panels_share_one_scoring_pass(monkeypatch, tmp_path):
    calls = _patch_spread(
        monkeypatch,
        spread_of=lambda pos: np.full(len(pos), 0.05),
        error_of=lambda pos: np.full(len(pos), 0.1),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"index": 0, "roi_position": [0, 0, 0]}]))
    path = tmp_path / "evaluated.extxyz"
    ase_io.write(path, [_labelled([[1.0, 0, 0], [3.0, 0, 0]])])

    board = Dashboard(1, 2, fail_silently=False)
    board.register_entry(
        UncertaintyCalibrationPanel([path], [], ["m.yaml"], datapath=tmp_path),
        ("0", "0"),
    )
    board.register_entry(
        ErrorVsROIDistancePanel(
            [path], [], ["m.yaml"], manifest, datapath=tmp_path
        ),
        ("0", "1"),
    )
    board.allocate_panels()
    board.compute_data()

    assert calls == [1]


def _write_pacemaker_preds(calcdir, split, n_atoms, e_shift, f_shift):
    """A minimal ``{split}_pred.pckl.gzip``: per-atom energy error
    ``e_shift`` and force-component error ``f_shift`` everywhere."""
    pd = pytest.importorskip("pandas")
    calcdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    forces = [rng.normal(size=(n, 3)) for n in n_atoms]
    energies = [-3.0 * n for n in n_atoms]
    pd.DataFrame(
        {
            "NUMBER_OF_ATOMS": n_atoms,
            "energy_corrected": energies,
            "energy_pred": [
                e + e_shift * n for e, n in zip(energies, n_atoms)
            ],
            "forces": forces,
            "forces_pred": [f + f_shift for f in forces],
        }
    ).to_pickle(calcdir / f"{split}_pred.pckl.gzip", compression="gzip")


def test_training_parity_panel_per_member(tmp_path):
    calcdirs = [tmp_path / "calcdirs" / str(i) for i in range(3)]
    _write_pacemaker_preds(calcdirs[0], "test", [2, 4], 0.01, 0.1)
    _write_pacemaker_preds(calcdirs[1], "test", [3], 0.03, 0.2)
    # Member 2: no test split written -> skipped, not an error.
    _write_pacemaker_preds(calcdirs[2], "train", [3], 0.0, 0.0)

    panel = TrainingParityPanel(calcdirs, split="test", datapath=tmp_path)
    _render(panel)

    d = panel.data
    assert sorted(d.e_obs) == ["0", "1"]
    assert len(d.e_obs["0"]) == 2
    assert d.e_obs["0"] == pytest.approx([-3.0, -3.0])  # per atom
    assert d.e_rmse["0"] == pytest.approx(0.01)
    assert d.f_rmse["1"] == pytest.approx(0.2)
    assert len(d.f_obs["0"]) == (2 + 4) * 3
    for ax in (panel.ax_e, panel.ax_f):
        assert all(c.get_rasterized() for c in ax.collections)


def test_training_parity_panel_subsamples_forces_not_metrics(tmp_path):
    calcdir = tmp_path / "calcdirs" / "0"
    _write_pacemaker_preds(calcdir, "train", [10, 10], 0.0, 0.5)

    panel = TrainingParityPanel(
        [calcdir], max_force_points=7, datapath=tmp_path
    )
    _render(panel)

    assert len(panel.data.f_obs["0"]) == 7
    assert len(panel.data.f_pred["0"]) == 7
    assert panel.data.f_rmse["0"] == pytest.approx(0.5)
    assert panel.data.f_mae["0"] == pytest.approx(0.5)


def test_training_parity_panel_no_predictions_raises(tmp_path):
    panel = TrainingParityPanel(
        [tmp_path / "calcdirs" / "0"], split="test", datapath=tmp_path
    )
    with pytest.raises(ValueError):
        panel.compute_data({})
