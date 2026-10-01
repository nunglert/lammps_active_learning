"""Tests for ``lammps_al.dashboard.run_layout``: the on-disk bridge to a
remote_handler staged-AL campaign directory.

All local-tier: pure synthetic ``AL_progress.json`` + ``calchome`` trees
built in ``tmp_path``, no cluster/pyace/LAMMPS involved.
"""

import json

import pytest

from lammps_al.dashboard import run_layout

pytestmark = pytest.mark.local


def _write_progress(calchome, outputs):
    """Write a staged-schema ``AL_progress.json``.

    ``outputs``: ``{stage: {iter: [rel, ...]}}``.
    """
    (calchome / "AL_progress.json").write_text(
        json.dumps({"stages": list(outputs), "outputs": outputs})
    )


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


def test_sampler_uncertainty_trajectory_found_and_missing(tmp_path):
    """The full annotated trajectory is CommitteeUncertaintySelector's
    recorded output (it annotates without reducing) -- a progress-based
    lookup, not a raw reach-in."""
    (tmp_path / "uncertainty_select_output").mkdir()
    traj = tmp_path / "uncertainty_select_output" / "u_it1_0.extxyz"
    traj.write_text("")
    _write_progress(
        tmp_path,
        {
            "uncertainty_select": {
                "1": ["uncertainty_select_output/u_it1_0.extxyz"]
            }
        },
    )
    progress = run_layout.load_progress(tmp_path)

    assert (
        run_layout.sampler_uncertainty_trajectory(tmp_path, progress, 1)
        == traj.resolve()
    )
    assert (
        run_layout.sampler_uncertainty_trajectory(tmp_path, progress, 0)
        is None
    )


# ---------------------------------------------------------------------------
# Raw reach-ins (not recorded in AL_progress.json)
# ---------------------------------------------------------------------------
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
# Anneal-box / DFT-calcdir status
# ---------------------------------------------------------------------------
def test_annealer_box_status(tmp_path):
    base = tmp_path / "1" / "anneal" / "array" / "calcdirs"
    for box in ("0", "1", "2"):
        (base / box / "out").mkdir(parents=True)
    (base / "0" / "out" / "lmp.log").write_text("ok")
    (base / "0" / "out" / "samples.extxyz").write_text("")
    (base / "1" / "out" / "lmp.log").write_text("ERROR: boom")

    status = run_layout.annealer_box_status(tmp_path, 1)

    assert status == [
        ("0", True, base / "0" / "out" / "lmp.log"),
        ("1", False, base / "1" / "out" / "lmp.log"),
        ("2", False, None),
    ]
    assert run_layout.annealer_box_status(tmp_path, 2) == []


def test_dft_calc_status(tmp_path):
    base = tmp_path / "1" / "dft" / "results" / "calcdirs"
    for calc in ("0", "1", "10"):
        (base / calc).mkdir(parents=True)
    (base / "0" / "is_finished.info").write_text("")
    (base / "10" / "is_finished.info").write_text("")

    # Numeric (not lexicographic) order.
    assert run_layout.dft_calc_status(tmp_path, 1) == [
        ("0", True),
        ("1", False),
        ("10", True),
    ]
    assert run_layout.dft_calc_status(tmp_path, 0) == []
