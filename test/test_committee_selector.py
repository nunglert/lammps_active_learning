"""Tests for the project's native committee-uncertainty Selector
(CommitteeUncertaintySelector) and the host-side PeakUncertaintySelector.

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode tests run with NO cluster, SSH, or LAMMPS -- only remote_handler +
ase + numpy on the host.

This actor was split out of LammpsSampler -- see
lammps_al.select.selector's module docstring -- so committee-uncertainty
annotation is a separate loop role from MD sampling. It does not reduce the
trajectory; PeakUncertaintySelector is the separate, optional, host-side
reduction step some campaigns need.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from remote_handler.exceptions import WorkflowError  # noqa: E402
from remote_handler.workflows.active_learning_base import (  # noqa: E402
    Selector,
)
from remote_handler.workflows.selectors import ConfigSelector  # noqa: E402

from lammps_al.select.scripts import script_dir  # noqa: E402
from lammps_al.select.selector import (  # noqa: E402
    CommitteeUncertaintySelector,
    PeakUncertaintySelector,
)

pytestmark = pytest.mark.integration


def _selector(remote_fs, **overrides):
    """Build a stub-mode CommitteeUncertaintySelector."""
    kwargs = dict(
        remote_fs=remote_fs,
        cutoff_radius=4.0,
        stub=True,
    )
    kwargs.update(overrides)
    return CommitteeUncertaintySelector(**kwargs)


def test_committee_selector_is_a_selector(mock_remote_fs):
    """CommitteeUncertaintySelector implements remote_handler's Selector
    interface."""
    s = _selector(mock_remote_fs)
    assert isinstance(s, Selector)


def test_default_script_points_at_shipped_payload(mock_remote_fs):
    """Omitting selection_script resolves to the bundled payload."""
    s = _selector(mock_remote_fs, selection_script=None)
    assert Path(s.selection_script) == script_dir() / "committee_selection.py"
    assert Path(s.selection_script).exists()


def _patch_single_script_task(monkeypatch):
    """Replace ``SingleScriptTask`` with a recording stub."""
    import lammps_al.select.selector as selector_mod

    created = {}

    class _StubTask:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.local_savepath = kwargs["local_savepath"]
            self.input_file = kwargs["input_file"]

        def allocate_data(self):
            pass

        def copy_file(self, *a, **k):
            pass

    monkeypatch.setattr(selector_mod, "SingleScriptTask", _StubTask)
    return created


def test_update_requires_model_files(mock_remote_fs, tmp_path):
    """update() needs model_files (the committee), wired via extra_inputs --
    not the primary selector_input."""
    s = _selector(mock_remote_fs, stub=False)
    trajectory = tmp_path / "trajectory.extxyz"
    trajectory.touch()

    with pytest.raises(WorkflowError, match="model_files"):
        s.update(tmp_path / "calcdir", trajectory)


def test_update_writes_script_input(
    mock_remote_fs, mock_model_files, tmp_path, monkeypatch
):
    """update() writes the CommitteeSelectionInput JSON with the trajectory
    and committee resolved correctly -- no bin_count anywhere (this actor
    only annotates)."""
    s = _selector(mock_remote_fs, cutoff_radius=2.5, stub=False)
    created = _patch_single_script_task(monkeypatch)
    trajectory = tmp_path / "trajectory.extxyz"
    trajectory.touch()

    s.update(tmp_path / "calcdir", trajectory, model_files=mock_model_files)

    script_input = json.loads(Path(created["input_file"]).read_text())
    assert script_input["trajectory"] == "trajectory.extxyz"
    assert script_input["model_files"] == [p.name for p in mock_model_files]
    assert script_input["cutoff_radius"] == 2.5
    assert "bin_count" not in script_input


def test_get_selected_configs_stub_mode(mock_remote_fs, tmp_path):
    """Stub mode fabricates a mock annotated trajectory (all frames kept,
    each carrying a per-atom uncertainty array)."""
    s = _selector(mock_remote_fs)
    trajectory = tmp_path / "trajectory.extxyz"
    trajectory.touch()
    s.update(tmp_path / "calcdir", trajectory, model_files=["m.pt"])

    output_path = s.get_selected_configs()

    assert output_path.exists()
    assert output_path.name == "trajectory_uncertainties.extxyz"
    annotated = ase.io.read(output_path, index=":")
    assert len(annotated) == 10
    for atoms in annotated:
        assert "uncertainty" in atoms.arrays


def test_selector_raises_on_unfinished(mock_remote_fs):
    """A job that leaves no is_finished marker aborts with WorkflowError."""
    from unittest.mock import MagicMock

    s = _selector(mock_remote_fs, stub=False)
    task = MagicMock()
    task.finish_check.return_value = False
    task.calcdir = "/remote/calc"
    s.selection_task = task

    with pytest.raises(WorkflowError, match="did not finish"):
        s.get_selected_configs()
    s.selection_task.collect_results.assert_not_called()


# --- PeakUncertaintySelector (host-side reduction) -------------------------
def _annotated_frame(uncertainties, tag):
    at = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[9, 9, 9], pbc=True
    )
    at.arrays["uncertainty"] = np.asarray(uncertainties, dtype=float)
    at.info["tag"] = tag
    return at


def test_peak_uncertainty_selector_is_a_config_selector():
    assert isinstance(PeakUncertaintySelector(bin_count=2), ConfigSelector)


def test_peak_uncertainty_selector_picks_peak_per_segment():
    configs = [
        _annotated_frame([1.0, 1.0], "b0f0"),
        _annotated_frame([5.0, 1.0], "b0f1"),  # peak of bin 0
        _annotated_frame([4.0, 1.0], "b1f0"),  # peak of bin 1
        _annotated_frame([2.0, 1.0], "b1f1"),
    ]

    selected = PeakUncertaintySelector(bin_count=2).select(configs)

    assert [s.info["tag"] for s in selected] == ["b0f1", "b1f0"]


def test_peak_uncertainty_selector_flattens_grouped_input():
    """SelectorActor passes one group per input trajectory file -- a single
    trajectory here is one group of 4 frames."""
    group = [
        _annotated_frame([1.0, 1.0], "f0"),
        _annotated_frame([5.0, 1.0], "f1"),
    ]

    selected = PeakUncertaintySelector(bin_count=1).select([group])

    assert len(selected) == 1
    assert selected[0].info["tag"] == "f1"


def test_peak_uncertainty_selector_trims_uneven_trajectory(caplog):
    """5 frames, bin_count=2 -> trims the last frame before binning."""
    configs = [
        _annotated_frame([1.0, 1.0], "f0"),
        _annotated_frame([5.0, 1.0], "f1"),  # peak of bin 0
        _annotated_frame([4.0, 1.0], "f2"),  # peak of bin 1
        _annotated_frame([2.0, 1.0], "f3"),
        _annotated_frame([9.0, 1.0], "f4"),  # dropped (trimmed tail)
    ]

    with caplog.at_level("WARNING"):
        selected = PeakUncertaintySelector(bin_count=2).select(configs)

    assert [s.info["tag"] for s in selected] == ["f1", "f2"]
    assert any("not evenly divisible" in r.message for r in caplog.records)
