"""End-to-end smoke test of the payload against a real MACE-enabled LAMMPS.

This is the only tier that needs the ``lmp`` binary (built with ML-IAP +
MLIAP-Python + KOKKOS and a ``mace`` importable in its Python), so it
**self-skips** unless a MACE-enabled ``lmp`` is on PATH -- exactly how
remote_handler's ``slurm``/``sshlive`` tiers self-skip. It runs on any machine
(dev box, cluster) that has such a build, and in CI via the prebuilt
``lammps-mace`` image (see ``docker/Dockerfile.lammps-mace`` and the
``lammps-smoke`` CI job).

It drives the payload's ``main`` on CPU over a tiny system for a handful of MD
steps and asserts a non-empty ``out/samples.xyz`` is produced -- proving the
``pair_style mliap unified`` MACE coupling loads and runs, and that the
compile -> run -> committee-score -> select pipeline holds together. The
numerics are not asserted (a 5-step CPU run is a functional smoke test).
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mace")

pytestmark = [pytest.mark.lammps, pytest.mark.integration]

_REPO = Path(__file__).resolve().parent.parent
_MODELS = sorted((_REPO / "examples/mace_md_AL/inputs/models").glob("*.model"))
_DATA = _REPO / "examples/mace_md_AL/inputs/test_fcc100_hollow_2x2.data"
_TEMPLATE = _REPO / "test/data/lammps/smoke.in.template"

# LAMMPS dump 'type' -> element, matching the template's `pair_coeff * * Co Cu H O`.
_ELEMENT_MAPPING = {"1": "Co", "2": "Cu", "3": "H", "4": "O"}

# CPU KOKKOS (OpenMP) accelerator flags -- the non-GPU analogue of the default.
_CPU_LAMMPS_ARGS = "-k on t 1 -sf kk -pk kokkos newton on neigh half"

_LMP = (
    shutil.which("lmp") or shutil.which("lmp_kokkos") or shutil.which("lammps")
)


def _load_payload():
    path = _REPO / "src/lammps_al/run/scripts/lammps_sampling.py"
    spec = importlib.util.spec_from_file_location("lammps_payload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(_LMP is None, reason="no lmp binary on PATH")
@pytest.mark.skipif(
    len(_MODELS) < 2 or not _DATA.exists(),
    reason="needs the example committee models + data",
)
def test_full_payload_smoke(tmp_path):
    payload = _load_payload()

    # The payload reads all inputs relative to the input file's directory.
    calc = tmp_path / "calc"
    calc.mkdir()
    shutil.copy(_TEMPLATE, calc / "lammps.in.template")
    shutil.copy(_DATA, calc / "structure.data")
    model_names = []
    for m in _MODELS[:2]:
        shutil.copy(m, calc / m.name)
        model_names.append(m.name)

    config = {
        "lammps_path": _LMP,
        "template_file": "lammps.in.template",
        "data_file": "structure.data",
        "model_files": model_names,
        "lammps_config_name": "lammps.in",
        "element_mapping": _ELEMENT_MAPPING,
        "cutoff_radius": 4.0,
        "bin_count": 2,
        "seed": 12345,
        "lammps_args": _CPU_LAMMPS_ARGS,
        "device": "cpu",
        "enable_cueq": False,  # plain e3nn compile: no cuequivariance on CPU
    }
    input_file = calc / "input_sample.json"
    input_file.write_text(json.dumps(config))

    payload.main([None, str(input_file)])

    # The job wrote its success marker and a non-empty candidate set.
    assert (calc / "is_finished.info").exists()
    samples = calc / "out" / "samples.xyz"
    assert samples.exists()

    import ase.io

    selected = ase.io.read(samples, index=":")
    assert len(selected) == config["bin_count"]
