"""Tests for the project's native LAMMPS Sampler (LammpsSampler).

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode tests run with NO cluster, SSH, or LAMMPS -- only remote_handler +
ase + numpy on the host.
"""

import json
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


def test_default_settings_num_samples_maps_to_bin_count(
    mock_remote_fs, mock_model_files, tmp_path
):
    """num_samples keeps its previous meaning: trajectory bin_count."""
    s = _sampler(mock_remote_fs, tmp_path, num_samples=7)
    s.model_files = mock_model_files
    settings = s._load_default_settings()
    assert settings["bin_count"] == 7
    # sampling_frequency is a distinct flag, not derived from num_samples
    assert settings["sampling_frequency"] is None


def test_default_settings_include_legacy_keys(
    mock_remote_fs, mock_model_files, tmp_path
):
    """The legacy uncertainty payload needs element_mapping/cutoff_radius."""
    s = _sampler(
        mock_remote_fs,
        tmp_path,
        element_mapping={1: "Cu", 2: "Co"},
        uncertainty_cutoff_radius=4.2,
    )
    s.model_files = mock_model_files
    settings = s._load_default_settings()
    assert settings["element_mapping"] == {1: "Cu", 2: "Co"}
    assert settings["cutoff_radius"] == 4.2


def test_default_settings_sampling_frequency_independent(
    mock_remote_fs, mock_model_files, tmp_path
):
    """sampling_frequency is its own flag, independent of num_samples."""
    s = _sampler(
        mock_remote_fs, tmp_path, sampling_frequency=25, num_samples=None
    )
    s.model_files = mock_model_files
    settings = s._load_default_settings()
    assert settings["sampling_frequency"] == 25
    assert settings["bin_count"] is None


def _patch_single_script_task(monkeypatch):
    """Replace ``SingleScriptTask`` with a recording stub.

    ``update()`` builds and immediately uses the task to write its input JSON;
    this stub captures the constructor kwargs (incl. ``input_file``) and
    no-ops the file transfers, so the merge logic runs for real with no cluster.
    """
    import lammps_al.run.sampler as sampler_mod

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

    monkeypatch.setattr(sampler_mod, "SingleScriptTask", _StubTask)
    return created


def test_update_warns_and_prefers_num_samples_when_both_specified(
    mock_remote_fs, mock_model_files, tmp_path, monkeypatch, caplog
):
    """Both num_samples and sampling_frequency -> warn, num_samples wins."""
    s = _sampler(mock_remote_fs, tmp_path, num_samples=5, stub=False)
    created = _patch_single_script_task(monkeypatch)

    with caplog.at_level("WARNING"):
        s.update(tmp_path / "calcdir", mock_model_files, sampling_frequency=2)

    assert any("Both num_samples" in r.message for r in caplog.records)
    script_input = json.loads(Path(created["input_file"]).read_text())
    assert script_input["bin_count"] == 5
    assert "sampling_frequency" not in script_input


def test_update_merges_schedule_kwargs_into_script_input(
    mock_remote_fs, mock_model_files, tmp_path, monkeypatch
):
    """Schedule kwargs are merged into the payload input JSON verbatim."""
    s = _sampler(mock_remote_fs, tmp_path, num_samples=None, stub=False)
    created = _patch_single_script_task(monkeypatch)

    s.update(tmp_path / "calcdir", mock_model_files, temperature=1000, n_run=500)

    script_input = json.loads(Path(created["input_file"]).read_text())
    assert script_input["temperature"] == 1000
    assert script_input["n_run"] == 500
    # sampling_frequency not set anywhere -> no conflict, stays null
    assert script_input.get("sampling_frequency") is None
    assert script_input["bin_count"] is None


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


def test_default_accelerator_is_gpu(mock_remote_fs, tmp_path):
    """Defaults preserve the production KOKKOS-on-GPU behaviour."""
    s = _sampler(mock_remote_fs, tmp_path)
    assert s.lammps_args == "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    assert s.device == "cuda"


def test_cpu_override_is_stored(mock_remote_fs, tmp_path):
    """A CPU run overrides the accelerator flags and device."""
    s = _sampler(
        mock_remote_fs,
        tmp_path,
        lammps_args="-k on t 2 -sf kk -pk kokkos newton on neigh half",
        device="cpu",
    )
    assert "t 2" in s.lammps_args
    assert s.device == "cpu"


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
