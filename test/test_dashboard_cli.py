"""End-to-end tests for ``lammps_al.dashboard.cli`` against a fully
synthetic ``calchome`` tree (no cluster/pyace/LAMMPS needed -- committee
scoring is patched at the panel-module import point, same technique as
``test_dashboard_panels.py``).
"""

import json

import numpy as np
import pytest
from ase import Atoms

ase_io = pytest.importorskip("ase.io")
pytest.importorskip("pypdf")

from lammps_al.dashboard import cli  # noqa: E402
from lammps_al.dashboard import run_layout  # noqa: E402

pytestmark = pytest.mark.local


def _make_config(rng, e_ref, n_atoms=3):
    atoms = Atoms(
        ["Cu", "H", "O"][:n_atoms] if n_atoms <= 3 else ["Cu"] * n_atoms,
        positions=rng.uniform(0, 5, size=(n_atoms, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    atoms.info["REF_energy"] = e_ref
    atoms.arrays["REF_forces"] = rng.normal(size=(n_atoms, 3))
    return atoms


def _fake_score_with_committee(model_files, atoms_list, **kwargs):
    energies = [float(a.info["REF_energy"]) + 0.1 for a in atoms_list]
    forces = None
    if kwargs.get("want_forces", True):
        forces = [
            np.asarray(a.arrays["REF_forces"]) + 0.01 for a in atoms_list
        ]
    return energies, forces


@pytest.fixture(autouse=True)
def _patch_committee_scoring(monkeypatch):
    """No real pyace anywhere in this file -- both panel modules that
    import score_with_committee get the fake."""
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.committee_parity.score_with_committee",
        _fake_score_with_committee,
    )
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.species_parity.score_with_committee",
        _fake_score_with_committee,
    )


@pytest.fixture
def two_iteration_calchome(tmp_path):
    """A synthetic 2-iteration campaign: iteration 0 bootstrap-trains only
    (entry_stage="trainer" semantics -- no evaluator/sampler output yet),
    iteration 1 has a full ring's worth of recorded + raw artifacts."""
    calchome = tmp_path / "calchome"
    (calchome / "trainer_output").mkdir(parents=True)
    (calchome / "evaluator_output").mkdir(parents=True)

    rng = np.random.default_rng(0)

    # Iteration 0: bootstrap train only, 2 committee members.
    for i in range(2):
        (calchome / "trainer_output" / f"trainer_it0_{i}.yaml").write_text(
            "potential"
        )

    # Iteration 1: 2 more committee members + evaluated configs (cumulative
    # over iteration 0's, so new_evaluator_configs_for_iteration exercises
    # the cumulative-detection branch) + a raw uncertainty trajectory.
    for i in range(2):
        (calchome / "trainer_output" / f"trainer_it1_{i}.yaml").write_text(
            "potential"
        )
    configs0 = [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    configs1 = configs0 + [
        _make_config(rng, e_ref=-3.0),
        _make_config(rng, e_ref=-4.0),
    ]
    ase_io.write(
        calchome / "evaluator_output" / "evaluator_it0_0.extxyz", configs0
    )
    ase_io.write(
        calchome / "evaluator_output" / "evaluator_it1_0.extxyz", configs1
    )

    out = calchome / "1" / "sample" / "results" / "out"
    out.mkdir(parents=True)
    traj_atoms = []
    for z in (0.0, 1.0, 2.0):
        atoms = Atoms(
            "CuHO",
            positions=[[0.0, 0.0, z], [0.0, 1.0, z], [0.0, 2.0, z]],
            cell=np.eye(3) * 20,
            pbc=True,
        )
        atoms.arrays["uncertainty"] = np.array([0.1, 0.1, 0.1])
        traj_atoms.append(atoms)
    ase_io.write(out / "traj+uncertainties.xyz", traj_atoms)

    (calchome / "AL_progress.json").write_text(
        json.dumps(
            {
                "stages": [
                    "sampler",
                    "selector",
                    "annealer",
                    "evaluator",
                    "trainer",
                ],
                "outputs": {
                    "trainer": {
                        "0": [
                            "trainer_output/trainer_it0_0.yaml",
                            "trainer_output/trainer_it0_1.yaml",
                        ],
                        "1": [
                            "trainer_output/trainer_it1_0.yaml",
                            "trainer_output/trainer_it1_1.yaml",
                        ],
                    },
                    "evaluator": {
                        "1": ["evaluator_output/evaluator_it1_0.extxyz"]
                    },
                },
            }
        )
    )
    return calchome


def test_build_dashboard_pdfs_produces_pages(tmp_path, two_iteration_calchome):
    out_dir = tmp_path / "dashboard_data"
    pages = cli.build_dashboard_pdfs(
        two_iteration_calchome, out_dir, ["Cu", "H", "O"]
    )

    # Iteration 0 has a trained model but no evaluated configs yet -> no
    # per-iteration page. Iteration 1 has both -> one page. Plus one
    # campaign-summary page (uncertainty trend/depth, from iteration 1's
    # raw trajectory).
    assert len(pages) == 2
    for page in pages:
        assert page.exists() and page.stat().st_size > 0


def test_build_dashboard_pdfs_only_filters_panels(
    tmp_path, two_iteration_calchome
):
    out_dir = tmp_path / "dashboard_data"
    pages = cli.build_dashboard_pdfs(
        two_iteration_calchome,
        out_dir,
        ["Cu", "H", "O"],
        only=["committee_parity"],
    )
    # Only the per-iteration page has a "committee_parity"-named entry; the
    # campaign page's entries are all excluded by `only` and so produces no
    # page at all.
    assert len(pages) == 1


def test_main_writes_merged_pdf(tmp_path, two_iteration_calchome):
    output = tmp_path / "dashboard.pdf"
    rc = cli.main([str(two_iteration_calchome), "--output", str(output)])

    assert rc == 0
    assert output.exists() and output.stat().st_size > 0


def test_main_returns_nonzero_when_no_pages(tmp_path):
    calchome = tmp_path / "empty_calchome"
    calchome.mkdir()
    (calchome / "AL_progress.json").write_text(
        json.dumps({"stages": [], "outputs": {}})
    )
    output = tmp_path / "dashboard.pdf"

    rc = cli.main([str(calchome), "--output", str(output)])

    assert rc == 1
    assert not output.exists()


def test_build_entries_for_iteration_skips_when_no_model_or_configs(tmp_path):
    calchome = tmp_path / "calchome"
    calchome.mkdir()
    (calchome / "AL_progress.json").write_text(
        json.dumps({"stages": [], "outputs": {}})
    )
    progress = run_layout.load_progress(calchome)

    entries = cli.build_entries_for_iteration(
        calchome, progress, 0, ["Cu", "H", "O"], datapath=tmp_path
    )
    assert entries == []
