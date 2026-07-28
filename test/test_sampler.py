"""Tests for the project's native LAMMPS Sampler (LammpsSampler).

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode tests run with NO cluster, SSH, or LAMMPS -- only remote_handler +
ase + numpy on the host.
"""

from pathlib import Path

import pytest

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from remote_handler.exceptions import WorkflowError  # noqa: E402
from remote_handler.workflows.active_learning_base import Sampler  # noqa: E402

from lammps_al.run.sampler import LammpsSampler  # noqa: E402
from lammps_al.run.scripts import script_dir  # noqa: E402

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration


def _sampler(remote_fs, tmp_path, **overrides):
    """Build a stub-mode LammpsSampler with touched input files."""
    script = tmp_path / "lammps_sampling.py"
    script.touch()
    template = tmp_path / "lammps.in.template"
    template.touch()
    data_file = tmp_path / "structure.data"
    data_file.touch()
    kwargs = dict(
        remote_fs=remote_fs,
        sampling_script=script,
        template_path=template,
        data_path=data_file,
        element_mapping={1: "Cu", 2: "Co"},
        num_samples=10,
        uncertainty_cutoff_radius=4.0,
        remote_lammps_executable_path="/remote/lammps/lmp",
        stub=True,
    )
    kwargs.update(overrides)
    return LammpsSampler(**kwargs)


def test_lammps_sampler_is_a_sampler(mock_remote_fs, tmp_path):
    """LammpsSampler implements remote_handler's Sampler interface."""
    s = _sampler(mock_remote_fs, tmp_path)
    assert isinstance(s, Sampler)


def test_init_lammps_sampler(mock_remote_fs, tmp_path):
    """Constructor stores the sampling parameters."""
    s = _sampler(
        mock_remote_fs,
        tmp_path,
        element_mapping={1: "Cu", 2: "Co"},
        num_samples=10,
    )
    assert s.stub is True
    assert s.num_samples == 10
    assert s.element_mapping == {1: "Cu", 2: "Co"}


def test_default_script_points_at_shipped_payload(mock_remote_fs, tmp_path):
    """Omitting sampling_script resolves to the bundled payload."""
    s = _sampler(mock_remote_fs, tmp_path, sampling_script=None)
    assert Path(s.sampling_script) == script_dir() / "lammps_sampling.py"
    assert Path(s.sampling_script).exists()


def test_lammps_stub_outputs(
    mock_remote_fs, temp_calcdir, mock_model_files, tmp_path
):
    """get_sampled_configs writes mock samples with uncertainty info."""
    s = _sampler(mock_remote_fs, tmp_path, num_samples=15)
    s.update(temp_calcdir, mock_model_files)
    output_file = s.get_sampled_configs()

    assert output_file.exists()
    assert output_file.name == "samples.xyz"

    sampled_atoms = ase.io.read(output_file, index=":")
    assert len(sampled_atoms) == 15
    for atoms in sampled_atoms:
        assert "max_uncertainty" in atoms.info
        assert "mean_uncertainty" in atoms.info
        assert "step" in atoms.info


def test_lammps_create_stub_outputs(mock_remote_fs, tmp_path):
    """_create_stub_outputs builds the expected 26-atom mock structures."""
    s = _sampler(mock_remote_fs, tmp_path, num_samples=20)
    stub_samples = s._create_stub_outputs(num_samples=8)

    assert len(stub_samples) == 8
    for atoms in stub_samples:
        assert len(atoms) == 26  # 20 Cu + 4 Co + 2 H
        assert "max_uncertainty" in atoms.info
        assert atoms.info["max_uncertainty"] > 0


def test_lammps_raises_on_unfinished(mock_remote_fs, tmp_path):
    """A job that leaves no is_finished marker aborts with WorkflowError."""
    from unittest.mock import MagicMock

    s = _sampler(mock_remote_fs, tmp_path, stub=False)
    task = MagicMock()
    task.finish_check.return_value = False
    task.calcdir = "/remote/calc"
    s.sampling_task = task

    with pytest.raises(WorkflowError, match="did not finish"):
        s.get_sampled_configs()
    s.sampling_task.collect_results.assert_not_called()
