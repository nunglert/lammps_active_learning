"""Shared fixtures for the lammps_al test suite.

These are deliberately dependency-light: the mock filesystem is a ``MagicMock``
and the model/calcdir factories touch only ``tmp_path``, so importing this
conftest never needs ``remote_handler`` or a MACE stack. The run-side test
modules gate the actual producer import with ``importorskip('remote_handler')``.
"""

import pickle
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_remote_fs():
    """A mock remote filesystem handler (no real SSH/cluster)."""
    mock_fs = MagicMock()
    mock_fs.slurm_config = {"savepath": "/remote/path"}
    return mock_fs


@pytest.fixture
def temp_calcdir(tmp_path):
    """A temporary per-iteration calculation directory."""
    calcdir = tmp_path / "iter_0"
    calcdir.mkdir()
    return calcdir


@pytest.fixture
def mock_model_files(tmp_path):
    """Three mock committee model files under ``tmp_path/models``."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_files = []
    for i in range(3):
        model_file = model_dir / f"model_{i}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model_id": i, "type": "MACE"}, f)
        model_files.append(model_file)
    return model_files
