"""Tests for ``lammps_al.dashboard.run_layout``: the on-disk bridge to a
remote_handler staged-AL campaign directory.

All local-tier: pure synthetic ``AL_progress.json`` + ``calchome`` trees
built in ``tmp_path``, no cluster/pyace/LAMMPS involved.
"""

import json

import numpy as np
import pytest
from ase import Atoms

ase_io = pytest.importorskip("ase.io")

from lammps_al.dashboard import run_layout  # noqa: E402

pytestmark = pytest.mark.local


def _write_progress(calchome, outputs):
    """Write a staged-schema ``AL_progress.json``.

    ``outputs``: ``{stage: {iter: [rel, ...]}}``.
    """
    (calchome / "AL_progress.json").write_text(
        json.dumps({"stages": list(outputs), "outputs": outputs})
    )


def _make_config(rng, e_ref, n_atoms=3):
    atoms = Atoms(
        "H" * n_atoms,
        positions=rng.uniform(0, 5, size=(n_atoms, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    atoms.info["REF_energy"] = e_ref
    atoms.arrays["REF_forces"] = rng.normal(size=(n_atoms, 3))
    return atoms


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------
def test_load_progress_staged_schema_five_stages(tmp_path):
    """The 5-stage ring (sampler/selector/annealer/evaluator/trainer) pivots
    correctly, including the annealer stage jaxrens_al's classic ring has
    none of."""
    _write_progress(
        tmp_path,
        {
            "sampler": {"0": [], "1": ["sampler_output/a.xyz"]},
            "selector": {"1": ["selector_output/b"]},
            "annealer": {"1": ["annealer_output/c.xyz"]},
            "evaluator": {"1": ["evaluator_output/d.xyz"]},
            "trainer": {
                "0": ["trainer_output/m0_0.yaml"],
                "1": ["trainer_output/m1_0.yaml"],
            },
        },
    )

    progress = run_layout.load_progress(tmp_path)

    assert len(progress) == 2  # iterations 0, 1
    assert progress[0]["trainer_output"] == ["trainer_output/m0_0.yaml"]
    assert progress[0]["annealer_output"] == []
    assert progress[1]["annealer_output"] == ["annealer_output/c.xyz"]
    assert progress[1]["evaluator_output"] == ["evaluator_output/d.xyz"]


def test_load_progress_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_layout.load_progress(tmp_path)


def test_load_progress_classic_list_schema(tmp_path):
    """The tolerated classic (bare list) schema still works."""
    (tmp_path / "AL_progress.json").write_text(
        json.dumps(
            [
                {"trainer_output": ["a.yaml"]},
                {"trainer_output": ["b.yaml"], "evaluator_output": ["c.xyz"]},
            ]
        )
    )
    progress = run_layout.load_progress(tmp_path)
    assert len(progress) == 2
    assert progress[1]["evaluator_output"] == ["c.xyz"]


# ---------------------------------------------------------------------------
# Recorded-path accessors
# ---------------------------------------------------------------------------
def test_trained_models_and_evaluated_configs_per_iteration(tmp_path):
    (tmp_path / "trainer_output").mkdir()
    (tmp_path / "trainer_output" / "m_0.yaml").write_text("potential")
    (tmp_path / "evaluator_output").mkdir()
    (tmp_path / "evaluator_output" / "e_0.xyz").write_text("")

    _write_progress(
        tmp_path,
        {
            "trainer": {"0": ["trainer_output/m_0.yaml"]},
            "evaluator": {"0": ["evaluator_output/e_0.xyz"]},
        },
    )
    progress = run_layout.load_progress(tmp_path)

    trained = run_layout.trained_models_per_iteration(tmp_path, progress)
    evaluated = run_layout.evaluated_configs_per_iteration(tmp_path, progress)

    assert trained == [[(tmp_path / "trainer_output" / "m_0.yaml").resolve()]]
    assert evaluated == [
        [(tmp_path / "evaluator_output" / "e_0.xyz").resolve()]
    ]


# ---------------------------------------------------------------------------
# Raw reach-ins (not recorded in AL_progress.json)
# ---------------------------------------------------------------------------
def test_sampler_uncertainty_trajectory_found_and_missing(tmp_path):
    out_dir = tmp_path / "1" / "sample" / "results" / "out"
    out_dir.mkdir(parents=True)
    traj = out_dir / "traj+uncertainties.xyz"
    traj.write_text("")

    assert run_layout.sampler_uncertainty_trajectory(tmp_path, 1) == traj
    assert run_layout.sampler_uncertainty_trajectory(tmp_path, 0) is None


def test_selector_manifest_found_and_missing(tmp_path):
    out_dir = tmp_path / "2" / "extract" / "out"
    out_dir.mkdir(parents=True)
    manifest = out_dir / "manifest.json"
    manifest.write_text("{}")

    assert run_layout.selector_manifest(tmp_path, 2) == manifest
    assert run_layout.selector_manifest(tmp_path, 3) is None


def test_iteration_dirs(tmp_path):
    for i in (0, 2, 1):
        (tmp_path / str(i)).mkdir()
    (tmp_path / "trainer_output").mkdir()  # not numeric -- must be excluded

    dirs = run_layout.iteration_dirs(tmp_path)
    assert [d.name for d in dirs] == ["0", "1", "2"]


def test_iteration_dirs_missing_calchome(tmp_path):
    assert run_layout.iteration_dirs(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# new_evaluator_configs_for_iteration: both delta/cumulative branches
# ---------------------------------------------------------------------------
def _write_evaluator_iteration(tmp_path, it, configs):
    out = tmp_path / "evaluator_output"
    out.mkdir(exist_ok=True)
    path = out / f"evaluator_it{it}_0.extxyz"
    ase_io.write(path, configs)
    return f"evaluator_output/{path.name}"


def test_new_evaluator_configs_iteration_zero_returns_all(tmp_path):
    rng = np.random.default_rng(0)
    configs0 = [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    rel0 = _write_evaluator_iteration(tmp_path, 0, configs0)
    _write_progress(tmp_path, {"evaluator": {"0": [rel0]}})
    progress = run_layout.load_progress(tmp_path)

    new = run_layout.new_evaluator_configs_for_iteration(tmp_path, progress, 0)
    assert len(new) == 2


def test_new_evaluator_configs_iteration_with_no_data_returns_empty(tmp_path):
    _write_progress(tmp_path, {"evaluator": {}})
    progress = run_layout.load_progress(tmp_path)
    assert (
        run_layout.new_evaluator_configs_for_iteration(tmp_path, progress, 0)
        == []
    )


def test_new_evaluator_configs_detects_cumulative(tmp_path):
    """iter 1's file contains iter 0's frames as a prefix plus 2 new ones."""
    rng = np.random.default_rng(1)
    configs0 = [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    new_configs = [
        _make_config(rng, e_ref=-3.0),
        _make_config(rng, e_ref=-4.0),
    ]
    rel0 = _write_evaluator_iteration(tmp_path, 0, configs0)
    rel1 = _write_evaluator_iteration(tmp_path, 1, configs0 + new_configs)
    _write_progress(tmp_path, {"evaluator": {"0": [rel0], "1": [rel1]}})
    progress = run_layout.load_progress(tmp_path)

    new = run_layout.new_evaluator_configs_for_iteration(tmp_path, progress, 1)

    assert len(new) == 2
    assert [a.info["REF_energy"] for a in new] == [-3.0, -4.0]


def test_new_evaluator_configs_detects_delta(tmp_path):
    """iter 1's file is unrelated to iter 0's -- treated as all-new."""
    rng = np.random.default_rng(2)
    configs0 = [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    configs1 = [_make_config(rng, e_ref=-5.0), _make_config(rng, e_ref=-6.0)]
    rel0 = _write_evaluator_iteration(tmp_path, 0, configs0)
    rel1 = _write_evaluator_iteration(tmp_path, 1, configs1)
    _write_progress(tmp_path, {"evaluator": {"0": [rel0], "1": [rel1]}})
    progress = run_layout.load_progress(tmp_path)

    new = run_layout.new_evaluator_configs_for_iteration(tmp_path, progress, 1)

    assert len(new) == 2
    assert [a.info["REF_energy"] for a in new] == [-5.0, -6.0]


def test_new_evaluator_configs_out_of_range_iteration_returns_empty(tmp_path):
    _write_progress(tmp_path, {"evaluator": {}})
    progress = run_layout.load_progress(tmp_path)
    assert (
        run_layout.new_evaluator_configs_for_iteration(tmp_path, progress, 5)
        == []
    )
