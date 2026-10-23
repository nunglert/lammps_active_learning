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


# --- crash recovery ---------------------------------------------------------
def _dump_steps(path):
    lines = Path(path).read_text().splitlines()
    return [
        int(lines[i + 1])
        for i, line in enumerate(lines)
        if line == "ITEM: TIMESTEP"
    ]


def test_latest_restart_picks_highest_step(tmp_path):
    for name in ("restart.500", "restart.1500", "restart.1000", "restart.x"):
        (tmp_path / name).write_text("")

    assert payload.latest_restart(tmp_path) == (
        1500,
        tmp_path / "restart.1500",
    )
    assert payload.latest_restart(tmp_path / "missing") is None


def test_truncate_dump_drops_frames_from_restart_step(tmp_path):
    dump = _write_dump(tmp_path / "d.dump", n_frames=4)
    # A crashed run's partly written last frame.
    with open(dump, "a") as f:
        f.write("ITEM: TIMESTEP\n4\nITEM: NUMBER OF ATOMS\n")

    kept = payload.truncate_dump(dump, before_step=2)

    assert kept == 2
    assert _dump_steps(dump) == [0, 1]
    assert dump.read_text().endswith("0.0 0.0\n")


# Stand-in for `lmp`: the run without read_restart dumps steps 0-20, writes
# a restart file at step 20, starts dumping step 30 and dies; a run with
# read_restart dumps steps 20-40 (appending) and succeeds. With
# FAKE_LMP_ALWAYS_FAIL set, every run dies.
_FAKE_LMP = r"""#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
deck = open(args[args.index("-in") + 1]).read().split("\n")
log = args[args.index("-log") + 1]
words = {l.split()[0]: l.split()[1:] for l in deck if l.strip()}
dump = words["dump"][0]
resume = "read_restart" in words

def frame(step):
    return (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
            "ITEM: ATOMS id type x y z\n1 1 0.0 0.0 0.0\n")

with open(log, "w") as f:
    f.write("resumed\n" if resume else "fresh\n")
with open(dump, "a" if resume else "w") as f:
    for step in ((20, 30, 40) if resume else (0, 10, 20)):
        f.write(frame(step))
    if not resume:
        if "restart" in words:
            pattern = words["restart"][1]
            open(pattern.replace("*", "20"), "w").close()
        f.write("ITEM: TIMESTEP\n30\n")
if os.environ.get("FAKE_LMP_ALWAYS_FAIL") or not resume:
    sys.exit(1)
"""

_RESTART_TEMPLATE = """{% if resume_file %}read_restart {{ resume_file }}
{% endif %}dump {{ dump_file }}
{% if restart_frequency %}restart {{ restart_frequency }} {{ restart_file }}
{% endif %}"""


def _run_with_restarts(tmp_path, **kwargs):
    lmp = tmp_path / "lmp"
    lmp.write_text(_FAKE_LMP)
    lmp.chmod(0o755)
    (tmp_path / "deck.j2").write_text(_RESTART_TEMPLATE)
    outpath = tmp_path / "out"
    outpath.mkdir()
    payload.run_lammps_with_restarts(
        outpath=outpath,
        log_file=outpath / "lmp.log",
        dump_file=outpath / "md.dump",
        lmp_cmd=str(lmp),
        input_file=outpath / "lammps.in",
        template_dir=tmp_path,
        template_file_name="deck.j2",
        lammps_args="",
        **kwargs,
    )
    return outpath


def test_crashed_run_resumes_from_restart_file(tmp_path):
    out = _run_with_restarts(tmp_path, max_restarts=2, restart_frequency=10)

    assert _dump_steps(out / "md.dump") == [0, 10, 20, 30, 40]
    log = (out / "lmp.log").read_text()
    assert log.startswith("fresh\n")
    assert "retry 1: resumed from restart.20" in log
    assert log.endswith("resumed\n")
    # Restart files only serve the recovery and are cleaned up after.
    assert not list(out.glob("restart.*"))


def test_crashed_run_without_restart_files_starts_over(tmp_path):
    # Without restart files the retry is a fresh run, which the fake lmp
    # always crashes.
    with pytest.raises(RuntimeError):
        _run_with_restarts(tmp_path, max_restarts=1)

    log = (tmp_path / "out" / "lmp.log").read_text()
    assert "retry 1: started over" in log
    assert _dump_steps(tmp_path / "out" / "md.dump")[:3] == [0, 10, 20]


def test_crash_recovery_gives_up_after_max_restarts(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LMP_ALWAYS_FAIL", "1")

    with pytest.raises(RuntimeError):
        _run_with_restarts(tmp_path, max_restarts=2, restart_frequency=10)

    log = (tmp_path / "out" / "lmp.log").read_text()
    assert "retry 2:" in log
    assert "retry 3:" not in log


def test_no_retry_by_default(tmp_path):
    with pytest.raises(RuntimeError):
        _run_with_restarts(tmp_path, restart_frequency=10)

    assert "retry" not in (tmp_path / "out" / "lmp.log").read_text()
