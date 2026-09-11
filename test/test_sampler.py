"""Tests for the project's native LAMMPS Sampler (LammpsSampler).

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode tests run with NO cluster, SSH, or LAMMPS -- only remote_handler +
ase + numpy on the host.

LammpsSampler only runs MD now -- committee-uncertainty scoring/selection is
a separate actor, see test_committee_selector.py.
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
        remote_lammps_executable_path="/remote/lammps/lmp",
        stub=True,
    )
    kwargs.update(overrides)
    return LammpsSampler(**kwargs)


def test_lammps_sampler_is_a_sampler(mock_remote_fs, tmp_path):
    """LammpsSampler implements remote_handler's Sampler interface."""
    s = _sampler(mock_remote_fs, tmp_path)
    assert isinstance(s, Sampler)


def test_default_settings_include_element_mapping_and_dump_frequency(
    mock_remote_fs, mock_model_files, tmp_path
):
    """element_mapping/dump_frequency are the sampler's own concerns now
    (committee-scoring fields -- cutoff_radius/bin_count/device -- moved to
    CommitteeUncertaintySelector)."""
    s = _sampler(
        mock_remote_fs,
        tmp_path,
        element_mapping={1: "Cu", 2: "Co"},
        dump_frequency=20,
    )
    s.model_files = mock_model_files
    settings = s._load_default_settings()
    assert settings["element_mapping"] == {1: "Cu", 2: "Co"}
    assert settings["dump_frequency"] == 20
    assert "bin_count" not in settings
    assert "cutoff_radius" not in settings
    assert "device" not in settings
    assert "sampling_frequency" not in settings


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


def test_update_keeps_dump_frequency(
    mock_remote_fs, mock_model_files, tmp_path, monkeypatch
):
    """dump_frequency reaches the written script_input JSON. Regression
    guard for the naming-collision bug that once dropped it silently (fixed
    by removing the colliding sampling_frequency field entirely -- see
    lammps_al.run.inputs.LammpsSamplingInput's docstring)."""
    s = _sampler(mock_remote_fs, tmp_path, dump_frequency=20, stub=False)
    created = _patch_single_script_task(monkeypatch)

    s.update(tmp_path / "calcdir", mock_model_files)

    script_input = json.loads(Path(created["input_file"]).read_text())
    assert script_input["dump_frequency"] == 20


def test_update_merges_schedule_kwargs_into_extra_template_kwargs(
    mock_remote_fs, mock_model_files, tmp_path, monkeypatch
):
    """Schedule kwargs that aren't one of LammpsSampler's own known fields
    are captured under extra_template_kwargs (forwarded to the template by
    the payload), not silently dropped as an "unknown key"."""
    s = _sampler(mock_remote_fs, tmp_path, stub=False)
    created = _patch_single_script_task(monkeypatch)

    s.update(
        tmp_path / "calcdir", mock_model_files, temperature=1000, n_run=500
    )

    script_input = json.loads(Path(created["input_file"]).read_text())
    assert script_input["extra_template_kwargs"] == {
        "temperature": 1000,
        "n_run": 500,
    }
    assert "temperature" not in script_input
    assert "n_run" not in script_input


def test_update_known_kwarg_overrides_top_level_field(
    mock_remote_fs, mock_model_files, tmp_path, monkeypatch
):
    """A schedule kwarg matching one of LammpsSampler's own fields (e.g.
    dump_frequency) overrides that field directly, not via
    extra_template_kwargs."""
    s = _sampler(mock_remote_fs, tmp_path, dump_frequency=1, stub=False)
    created = _patch_single_script_task(monkeypatch)

    s.update(tmp_path / "calcdir", mock_model_files, dump_frequency=50)

    script_input = json.loads(Path(created["input_file"]).read_text())
    assert script_input["dump_frequency"] == 50
    assert "dump_frequency" not in script_input["extra_template_kwargs"]


def test_init_lammps_sampler(mock_remote_fs, tmp_path):
    """Constructor stores the sampling parameters."""
    s = _sampler(
        mock_remote_fs,
        tmp_path,
        element_mapping={1: "Cu", 2: "Co"},
    )
    assert s.stub is True
    assert s.element_mapping == {1: "Cu", 2: "Co"}


def test_default_script_points_at_shipped_payload(mock_remote_fs, tmp_path):
    """Omitting sampling_script resolves to the bundled payload."""
    s = _sampler(mock_remote_fs, tmp_path, sampling_script=None)
    assert Path(s.sampling_script) == script_dir() / "lammps_sampling.py"
    assert Path(s.sampling_script).exists()


def test_lammps_stub_outputs(
    mock_remote_fs, temp_calcdir, mock_model_files, tmp_path
):
    """get_sampled_configs writes a mock trajectory."""
    s = _sampler(mock_remote_fs, tmp_path)
    s.update(temp_calcdir, mock_model_files)
    output_file = s.get_sampled_configs()

    assert output_file.exists()
    assert output_file.name == "trajectory.extxyz"

    frames = ase.io.read(output_file, index=":")
    assert len(frames) == 10
    for atoms in frames:
        assert "step" in atoms.info


def test_lammps_create_stub_outputs(mock_remote_fs, tmp_path):
    """_create_stub_outputs builds the expected 26-atom mock structures."""
    s = _sampler(mock_remote_fs, tmp_path)
    stub_frames = s._create_stub_outputs(num_frames=8)

    assert len(stub_frames) == 8
    for atoms in stub_frames:
        assert len(atoms) == 26  # 20 Cu + 4 Co + 2 H


def test_default_accelerator_is_gpu(mock_remote_fs, tmp_path):
    """Defaults preserve the production KOKKOS-on-GPU + ML-IAP behaviour."""
    s = _sampler(mock_remote_fs, tmp_path)
    assert s.lammps_args == "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    assert s.lammps_pair_style == "mliap"
    assert s.enable_cueq is True


def test_cpu_override_is_stored(mock_remote_fs, tmp_path):
    """A CPU-testable run overrides the accelerator flags and pair_style --
    opting into the ``"mace"`` route, not the default."""
    s = _sampler(
        mock_remote_fs,
        tmp_path,
        lammps_args="",
        lammps_pair_style="mace",
        enable_cueq=False,
    )
    assert s.lammps_args == ""
    assert s.lammps_pair_style == "mace"
    assert s.enable_cueq is False


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
