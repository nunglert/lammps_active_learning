"""Tests for the LAMMPS sampling input schema (LammpsSamplingInput).

Needs the [run] extra (remote_handler provides the ``ScriptInput`` base); the
module self-skips otherwise. Committee-scoring fields (cutoff_radius,
bin_count, device) moved to CommitteeSelectionInput -- see
test_committee_selection_inputs.py.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("remote_handler")
from remote_handler.exceptions import ScriptInputError  # noqa: E402

from lammps_al.run.inputs import LammpsSamplingInput  # noqa: E402

pytestmark = pytest.mark.integration


def test_from_file_valid(tmp_path):
    """Loading a well-formed JSON input file."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    input_file = tmp_path / "input.json"
    input_data = {
        "lammps_path": str(lammps_exe),
        "template_file": "lammps.in.template",
        "data_file": "structure.data",
        "model_files": ["model1.pt", "model2.pt"],
        "element_mapping": {"1": "H", "2": "O"},
        "seed": 42,
    }
    input_file.write_text(json.dumps(input_data))

    config = LammpsSamplingInput.from_file(input_file)

    assert config.lammps_path == lammps_exe
    assert config.template_file == "lammps.in.template"
    assert config.model_files == ["model1.pt", "model2.pt"]


def test_validate_lammps_not_found():
    """Validation fails when the LAMMPS executable is missing."""
    config = LammpsSamplingInput(
        lammps_path=Path("/nonexistent/lmp"),
        template_file="template.in",
        data_file="data.in",
        model_files=["model.pt"],
        element_mapping={"1": "H"},
        seed=42,
    )
    with pytest.raises(ScriptInputError, match="not found"):
        config.validate()


def test_validate_empty_model_files(tmp_path):
    """Validation fails with an empty committee."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="template.in",
        data_file="data.in",
        model_files=[],
        element_mapping={"1": "H"},
        seed=42,
    )
    with pytest.raises(ScriptInputError, match="cannot be empty"):
        config.validate()


def test_defaults_are_gpu(tmp_path):
    """The accelerator fields default to the production GPU values."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()
    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["m.pt"],
        element_mapping={"1": "H"},
        seed=42,
    )
    assert config.lammps_args == (
        "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    )
    assert config.lammps_pair_style == "mliap"
    assert config.enable_cueq is True
    config.validate()  # valid without root


def test_validate_bad_pair_style(tmp_path):
    """An unknown lammps_pair_style is rejected."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()
    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["m.pt"],
        element_mapping={"1": "H"},
        seed=42,
        lammps_pair_style="unified",
    )
    with pytest.raises(ScriptInputError, match="lammps_pair_style"):
        config.validate()


def test_ace_backend_allows_arbitrary_pair_style(tmp_path):
    """backend='ace'/'grace' pair styles are native -- any non-empty
    pair_style line is valid, unlike backend='mace's mliap/mace fork."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()
    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["potential.yaml"],
        element_mapping={"1": "H"},
        seed=42,
        backend="ace",
        lammps_pair_style="pace",
    )
    config.validate()  # does not raise


def test_grace_backend_rejects_empty_pair_style(tmp_path):
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()
    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["model_dir"],
        element_mapping={"1": "H"},
        seed=42,
        backend="grace",
        lammps_pair_style="",
    )
    with pytest.raises(ScriptInputError, match="lammps_pair_style"):
        config.validate()


def test_validate_bad_backend(tmp_path):
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()
    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["m.pt"],
        element_mapping={"1": "H"},
        seed=42,
        backend="quip",
    )
    with pytest.raises(ScriptInputError, match="backend"):
        config.validate()


def test_from_file_recognizes_dump_frequency(tmp_path, caplog):
    """``dump_frequency`` must round-trip as a real field, not be silently
    dropped as an "unknown key"."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    input_file = tmp_path / "input.json"
    input_data = {
        "lammps_path": str(lammps_exe),
        "template_file": "lammps.in.template",
        "data_file": "structure.data",
        "model_files": ["model1.pt"],
        "element_mapping": {"1": "H"},
        "seed": 42,
        "dump_frequency": 20,
    }
    input_file.write_text(json.dumps(input_data))

    with caplog.at_level("WARNING"):
        config = LammpsSamplingInput.from_file(input_file)

    assert config.dump_frequency == 20
    assert not any("unknown key" in r.message.lower() for r in caplog.records)


def test_dump_frequency_defaults_to_none(tmp_path):
    """Unset (as opposed to explicitly provided) stays None -- the payload
    side is responsible for falling back to every-step when so."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="template.in",
        data_file="data.in",
        model_files=["model.pt"],
        element_mapping={"1": "H"},
        seed=42,
    )

    assert config.dump_frequency is None
    config.validate()  # None is valid -- only non-positive values are rejected


def test_validate_negative_dump_frequency(tmp_path):
    """Validation fails for a non-positive dump_frequency."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="template.in",
        data_file="data.in",
        model_files=["model.pt"],
        element_mapping={"1": "H"},
        seed=42,
        dump_frequency=0,
    )
    with pytest.raises(ScriptInputError, match="dump_frequency"):
        config.validate()


def test_extra_template_kwargs_defaults_empty_and_round_trips(tmp_path):
    """extra_template_kwargs defaults to {} and round-trips through JSON --
    the passthrough for arbitrary Schedule kwargs (temperature/timestep/
    n_run/...) that used to be silently dropped as unknown keys."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["m.pt"],
        element_mapping={"1": "H"},
        seed=42,
    )
    assert config.extra_template_kwargs == {}

    input_file = tmp_path / "input.json"
    input_data = {
        "lammps_path": str(lammps_exe),
        "template_file": "t.in",
        "data_file": "d.in",
        "model_files": ["m.pt"],
        "element_mapping": {"1": "H"},
        "seed": 42,
        "extra_template_kwargs": {"temperature": 1000, "n_run": 500},
    }
    input_file.write_text(json.dumps(input_data))
    loaded = LammpsSamplingInput.from_file(input_file)
    assert loaded.extra_template_kwargs == {"temperature": 1000, "n_run": 500}


@pytest.mark.parametrize(
    "field, value",
    [("max_restarts", -1), ("restart_frequency", 0)],
)
def test_validate_bad_restart_settings(tmp_path, field, value):
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()
    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="t.in",
        data_file="d.in",
        model_files=["m.pt"],
        element_mapping={"1": "H"},
        seed=42,
        **{field: value},
    )
    with pytest.raises(ScriptInputError, match=field):
        config.validate()
