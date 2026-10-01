"""Tests for the ACE-committee dashboard panels.

``CommitteeParityPanel``/``HeldOutParityPanel``/``SpeciesForceParityPanel``
patch ``score_with_committee`` at the point each panel module imported it
(the standard way to seam out a dependency that has no per-call injection
parameter of its own) so no real ``pyace`` is needed. The uncertainty panels
need no committee at all -- they read per-atom ``uncertainty`` arrays
directly off synthetic sampler trajectories.
"""

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
from lammps_al.dashboard.panels.species_parity import (  # noqa: E402
    SpeciesForceParityPanel,
    SpeciesParityData,
)
from lammps_al.dashboard.panels.uncertainty import (  # noqa: E402
    UncertaintyDepthProfilePanel,
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
