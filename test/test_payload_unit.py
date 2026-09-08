"""Unit tests for the pure logic of the LAMMPS sampling payload.

These exercise the payload's dependency-light helpers -- jinja rendering,
local-uncertainty aggregation, and the segment/argmax candidate selection --
with only ase + numpy + jinja2. No remote_handler, no MACE stack, no LAMMPS
binary, no GPU. They are the fast, always-run tier that validates the sampling
*logic*; the MACE committee scoring and the real ``lmp`` run are covered in the
``lammps`` integration tier (see ``test_lammps_integration.py``).

The payload is imported by file path so the import never pulls in
``lammps_al.run.inputs`` (its ``remote_handler`` dependency is confined to
``main``).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

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
