"""End-to-end smoke test of the payload against a real MACE-enabled LAMMPS.

This is the only tier that needs the ``lmp`` binary and a ``mace`` importable
in the host Python, so it **self-skips** unless a MACE-enabled ``lmp`` is on
PATH -- exactly how remote_handler's ``slurm``/``sshlive`` tiers self-skip.
It runs on any machine (dev box, cluster) that has such a build, and in CI via
the prebuilt ``lammps-mace`` image (see ``docker/Dockerfile.lammps-mace`` and
the ``lammps-smoke`` CI job) -- and the exact same image is how to reproduce
this tier locally (see that Dockerfile's header for the `docker run`
invocation).

It drives the payload's ``main`` on CPU over a tiny system for a handful of MD
steps and asserts a non-empty ``out/trajectory.extxyz`` is produced -- proving
the compile -> run -> convert pipeline holds together (this payload is
MD-only now; committee scoring is a separate payload/stage, see
``test_committee_scoring.py``). The numerics are not asserted (a 5-step CPU
run is a functional smoke test).

Explicitly requests ``lammps_pair_style="mace"`` (LAMMPS's native
``pair_style mace``, the ``ML-MACE`` package): unlike the default
``"mliap"`` (the ML-IAP ``mliap unified`` bridge, which needs LAMMPS's
Kokkos-CUDA-only ghost-atom message passing for any multi-layer MACE model),
``"mace"`` runs correctly CPU-only, so this tier gives real coverage in CI
and on a plain dev laptop, not just on a GPU cluster. This does **not**
change the sampler's production default -- see ``lammps_sampling.py``'s
module docstring.
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

# `pair_style mace` needs no accelerator flags at all on a CPU-only build
# (unlike the old ML-IAP route's Kokkos-OpenMP requirement).
_CPU_LAMMPS_ARGS = ""

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
        "seed": 12345,
        "lammps_args": _CPU_LAMMPS_ARGS,
        "device": "cpu",
        "lammps_pair_style": "mace",
    }
    input_file = calc / "input_sample.json"
    input_file.write_text(json.dumps(config))

    payload.main([None, str(input_file)])

    # The job wrote its success marker and a non-empty trajectory.
    assert (calc / "is_finished.info").exists()
    trajectory = calc / "out" / "trajectory.extxyz"
    assert trajectory.exists()

    import ase.io

    frames = ase.io.read(trajectory, index=":")
    assert len(frames) > 0
    assert set(frames[0].get_chemical_symbols()) <= set(
        _ELEMENT_MAPPING.values()
    )
