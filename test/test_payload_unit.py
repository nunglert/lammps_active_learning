"""Unit tests for the pure logic of the LAMMPS sampling payload.

Exercises the payload's dependency-light jinja-rendering helper with only
ase + numpy + jinja2. No remote_handler, no MACE stack, no LAMMPS binary, no
GPU. Committee-uncertainty scoring/selection logic lives in a separate
payload now (``lammps_al.select.scripts.committee_selection``) -- see
test_committee_selection_payload.py.

The payload is imported by file path so the import never pulls in
``lammps_al.run.inputs`` (its ``remote_handler`` dependency is confined to
``main``).
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.local


def _load_payload():
    """Import the payload module by path (its main() is not executed)."""
    path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "lammps_al"
        / "run"
        / "scripts"
        / "lammps_sampling.py"
    )
    spec = importlib.util.spec_from_file_location("lammps_payload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


payload = _load_payload()


# --- parse_jinja_template --------------------------------------------------
def test_parse_jinja_template_renders_kwargs(tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "lammps.in.j2").write_text(
        "seed {{ seed }}\nread_data {{ data_file }}\n"
    )
    out = tmp_path / "lammps.in"

    payload.parse_jinja_template(
        out, template_dir, "lammps.in.j2", seed=42, data_file="struct.data"
    )

    text = out.read_text()
    assert "seed 42" in text
    assert "read_data struct.data" in text


def test_ace_template_dump_frequency_is_rendered_not_hardcoded():
    """Regression test: the dump line must read ``{{ dump_frequency }}``,
    not a hardcoded ``1`` -- confirmed as the cause of an unthinned,
    multi-GB dump file within minutes on a large real system
    (``dump_frequency`` was silently dropped upstream as an "unknown key"
    and never reached the template at all).

    Renders a git-tracked copy of the real shipped template
    (test/data/lammps/lammps_ace.in.template), not a synthetic stand-in --
    a live campaign's own working copy of this template is typically
    gitignored (campaign-specific working files), so it can't be read
    directly here in a fresh clone. Keep the two in sync by hand; both
    files cross-reference each other in their header comments.
    """
    template_dir = Path(__file__).resolve().parent / "data" / "lammps"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lammps.in"
        payload.parse_jinja_template(
            out,
            template_dir,
            "lammps_ace.in.template",
            model_file="model.yaml",
            data_file="struct.data",
            lammps_pair_style="pace product",
            seed=42,
            dump_file="traj.dump",
            dump_frequency=20,
            final_data_file="final.data",
        )
        text = out.read_text()

    dump_lines = [line for line in text.splitlines() if "dump" in line]
    assert any(
        "dump" in line and "d_nvt" in line and "20" in line
        for line in dump_lines
    ), f"expected a dump line with frequency 20, got: {dump_lines}"
    assert not any(
        "dump" in line and "d_nvt" in line and " 1 " in line
        for line in dump_lines
    ), "dump frequency looks hardcoded to 1 again"


# --- convert_dump_to_extxyz -------------------------------------------------
def _write_dump(path, n_frames=2, n_atoms=3):
    """Write a minimal multi-frame ``lammps-dump-text`` trajectory."""
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


def test_convert_dump_to_extxyz_remaps_and_writes(tmp_path):
    """The MD-only payload converts the raw dump to a proper extxyz
    trajectory with correct chemical symbols (element_mapping applied) --
    this is what the committee-selection payload now reads instead of doing
    the remap itself."""
    import ase.io

    dump = _write_dump(tmp_path / "d.dump", n_frames=2)
    out = tmp_path / "trajectory.extxyz"

    payload.convert_dump_to_extxyz(
        trajectory_file=dump,
        element_mapping={"1": "O", "2": "Sr", "3": "Ti"},
        output_file=out,
    )

    frames = ase.io.read(out, index=":")
    assert len(frames) == 2
    for atoms in frames:
        assert atoms.get_chemical_symbols() == ["O", "Sr", "Ti"]
