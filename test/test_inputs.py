"""Tests for the LAMMPS sampling input schema (LammpsSamplingInput).

Needs the [run] extra (remote_handler provides the ``ScriptInput`` base); the
module self-skips otherwise.
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
        "cutoff_radius": 5.0,
        "bin_count": 10,
        "seed": 42,
    }
    input_file.write_text(json.dumps(input_data))

    config = LammpsSamplingInput.from_file(input_file)

    assert config.lammps_path == lammps_exe
    assert config.template_file == "lammps.in.template"
    assert config.model_files == ["model1.pt", "model2.pt"]
    assert config.cutoff_radius == 5.0
    assert config.bin_count == 10


def test_validate_lammps_not_found():
    """Validation fails when the LAMMPS executable is missing."""
    config = LammpsSamplingInput(
        lammps_path=Path("/nonexistent/lmp"),
        template_file="template.in",
        data_file="data.in",
        model_files=["model.pt"],
        element_mapping={"1": "H"},
        cutoff_radius=5.0,
        bin_count=10,
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
        cutoff_radius=5.0,
        bin_count=10,
        seed=42,
    )
    with pytest.raises(ScriptInputError, match="cannot be empty"):
        config.validate()


def test_validate_negative_cutoff(tmp_path):
    """Validation fails for a non-positive cutoff radius."""
    lammps_exe = tmp_path / "lmp"
    lammps_exe.touch()

    config = LammpsSamplingInput(
        lammps_path=lammps_exe,
        template_file="template.in",
        data_file="data.in",
        model_files=["model.pt"],
        element_mapping={"1": "H"},
        cutoff_radius=-1.0,
        bin_count=10,
        seed=42,
    )
    with pytest.raises(ScriptInputError, match="must be positive"):
        config.validate()
