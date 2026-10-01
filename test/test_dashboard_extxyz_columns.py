"""Tests for ``lammps_al.dashboard.extxyz_columns`` against ASE's own
extxyz writer/reader (the files it reads are ASE-written)."""

import numpy as np
import pytest
from ase import Atoms

ase_io = pytest.importorskip("ase.io")
pytest.importorskip("pandas")

from lammps_al.dashboard.extxyz_columns import iter_columns  # noqa: E402

pytestmark = pytest.mark.local


@pytest.fixture
def traj(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for step in (0, 100):
        atoms = Atoms(
            "CuHO",
            positions=rng.uniform(0, 5, size=(3, 3)),
            cell=np.eye(3) * 10,
            pbc=True,
        )
        atoms.arrays["uncertainty"] = rng.uniform(size=3)
        atoms.arrays["REF_forces"] = rng.normal(size=(3, 3))
        atoms.info["timestep"] = step
        frames.append(atoms)
    path = tmp_path / "traj.extxyz"
    ase_io.write(path, frames)
    return path, ase_io.read(path, index=":")


def test_iter_columns_matches_ase(traj):
    path, expected = traj
    got = list(
        iter_columns(
            path,
            ["uncertainty", "species", ("pos", 2), "REF_forces"],
            info_keys=("timestep", "not_there"),
        )
    )

    assert len(got) == len(expected)
    for (info, a), atoms in zip(got, expected):
        assert info == {"timestep": atoms.info["timestep"]}
        np.testing.assert_allclose(
            a["uncertainty"], atoms.arrays["uncertainty"]
        )
        assert list(a["species"]) == atoms.get_chemical_symbols()
        np.testing.assert_allclose(a[("pos", 2)], atoms.positions[:, 2])
        np.testing.assert_allclose(a["REF_forces"], atoms.arrays["REF_forces"])


def test_iter_columns_missing_column_raises(traj):
    path, _ = traj
    with pytest.raises(KeyError, match="no_such_column"):
        next(iter_columns(path, ["no_such_column"]))
