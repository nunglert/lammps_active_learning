"""Tests for the committee-selection input schema (CommitteeSelectionInput).

Needs the [run] extra (remote_handler provides the ``ScriptInput`` base); the
module self-skips otherwise. These fields (model_files, cutoff_radius,
device, backend) previously lived on LammpsSamplingInput before committee
scoring was split out of the MD-running sampler -- see test_inputs.py.

Deliberately no ``bin_count`` field -- this payload only annotates, it does
not select; see test_committee_selector.py's PeakUncertaintySelector tests
for the separate, host-side reduction step.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("remote_handler")
from remote_handler.exceptions import ScriptInputError  # noqa: E402

from lammps_al.select.inputs import CommitteeSelectionInput  # noqa: E402

pytestmark = pytest.mark.integration


def test_from_file_valid(tmp_path):
    input_file = tmp_path / "input.json"
    input_data = {
        "trajectory": "trajectory.extxyz",
        "model_files": ["model1.pt", "model2.pt"],
        "cutoff_radius": 5.0,
    }
    input_file.write_text(json.dumps(input_data))

    config = CommitteeSelectionInput.from_file(input_file)

    assert config.trajectory == Path("trajectory.extxyz")
    assert config.model_files == ["model1.pt", "model2.pt"]
    assert config.cutoff_radius == 5.0


def test_validate_empty_model_files():
    config = CommitteeSelectionInput(
        trajectory=Path("trajectory.extxyz"),
        model_files=[],
        cutoff_radius=5.0,
    )
    with pytest.raises(ScriptInputError, match="cannot be empty"):
        config.validate()


def test_validate_negative_cutoff():
    config = CommitteeSelectionInput(
        trajectory=Path("trajectory.extxyz"),
        model_files=["model.pt"],
        cutoff_radius=-1.0,
    )
    with pytest.raises(ScriptInputError, match="must be positive"):
        config.validate()


def test_validate_bad_device():
    config = CommitteeSelectionInput(
        trajectory=Path("trajectory.extxyz"),
        model_files=["model.pt"],
        cutoff_radius=5.0,
        device="tpu",
    )
    with pytest.raises(ScriptInputError, match="device"):
        config.validate()


def test_validate_bad_backend():
    config = CommitteeSelectionInput(
        trajectory=Path("trajectory.extxyz"),
        model_files=["model.pt"],
        cutoff_radius=5.0,
        backend="quip",
    )
    with pytest.raises(ScriptInputError, match="backend"):
        config.validate()


def test_defaults(tmp_path):
    config = CommitteeSelectionInput(
        trajectory=Path("trajectory.extxyz"),
        model_files=["model.pt"],
        cutoff_radius=5.0,
    )
    assert config.device == "cuda"
    assert config.backend == "mace"
    config.validate()  # valid without root
