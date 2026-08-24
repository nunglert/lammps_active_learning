"""Tests for the frequency-based LAMMPS sampling payload.

Covers the pure, dependency-light helpers of ``lammps_dump_sampling.py`` and
the host-side ``remap_elements`` (in ``sampler.py``): striding a trajectory by
``sampling_frequency``, element remapping from the type column, and writing
extxyz. Needs only ase + numpy + jinja2 -- no remote_handler, MACE, or LAMMPS.
"""

from pathlib import Path

import pytest

from lammps_al.run.sampler import remap_elements
from lammps_al.run.scripts.lammps_dump_sampling import (
    load_dump_frames,
    select_frequency_frames,
    write_samples,
)


def _write_dump(path: Path, n_frames: int = 10, n_atoms: int = 3) -> Path:
    """Write a minimal multi-frame ``lammps-dump-text`` trajectory.

    ``n_atoms`` atoms per frame with type column 1, 2, 3 in that order.
    """
    blocks = []
    for step in range(n_frames):
        lines = [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(n_atoms),
            "ITEM: BOX BOUNDS pp pp pp",
            "0.0 10.0",
            "0.0 10.0",
            "0.0 10.0",
            "ITEM: ATOMS id type x y z",
        ]
        for i in range(1, n_atoms + 1):
            lines.append(f"{i} {i} {float(i)} 0.0 0.0")
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n")
    return path


def test_select_frequency_frames_strides(tmp_path):
    """Every Nth frame is selected, frame 0 always included."""
    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=10))
    selected = select_frequency_frames(structs, 3)
    assert len(selected) == 4  # 0, 3, 6, 9
    # selected lists hold the same Atoms objects at the original positions
    assert selected == [structs[i] for i in (0, 3, 6, 9)]


def test_select_frequency_frames_frequency_one_selects_all(tmp_path):
    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=5))
    assert len(select_frequency_frames(structs, 1)) == 5


def test_select_frequency_frames_keeps_first_frame(tmp_path):
    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=7))
    selected = select_frequency_frames(structs, 4)
    assert structs[0] is selected[0]


def test_remap_elements_changes_type_index_to_symbol(tmp_path):
    """The dump's type column (1,2,3) becomes O,Sr,Ti, not H,He,Li."""
    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=1))
    remap_elements(structs, {"1": "O", "2": "Sr", "3": "Ti"})
    assert structs[0].get_chemical_symbols() == ["O", "Sr", "Ti"]


def test_remap_elements_no_mapping_returns_unchanged(tmp_path):
    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=1))
    before = structs[0].get_chemical_symbols()
    remap_elements(structs, None)
    assert structs[0].get_chemical_symbols() == before


def test_remap_elements_is_in_place_and_preserves_order(tmp_path):
    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=2))
    remap_elements(structs, {"1": "O", "2": "Sr", "3": "Ti"})
    assert structs[0].get_chemical_symbols() == ["O", "Sr", "Ti"]
    assert structs[1].get_chemical_symbols() == ["O", "Sr", "Ti"]


def test_write_samples_writes_extxyz(tmp_path):
    import ase.io

    structs = load_dump_frames(_write_dump(tmp_path / "d.dump", n_frames=3))
    remap_elements(structs, {"1": "O", "2": "Sr", "3": "Ti"})
    out = tmp_path / "samples.xyz"
    write_samples(structs, out)
    assert out.exists()
    reread = ase.io.read(out, index=":")
    assert len(reread) == 3
    assert reread[0].get_chemical_symbols() == ["O", "Sr", "Ti"]
    assert out.read_text().startswith("3\n")  # extxyz header line = natoms
