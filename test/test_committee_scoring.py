"""Integration test for the MACE-committee scoring half of the pipeline.

Exercises the two-stage pipeline end-to-end on CPU with the default
``backend="mace"``: convert a raw LAMMPS dump to extxyz (the sampler's own
payload, ``lammps_sampling.convert_dump_to_extxyz``), then score every
frame's committee force-uncertainty with a real (small) MACE committee and
aggregate locally (the selector's payload,
``committee_selection.annotate_uncertainties``) -- writing the fully
annotated trajectory, no reduction. This needs the MACE stack (``[run]``
extra) and a couple of committee models, but **no LAMMPS binary and no
GPU**; it self-skips when either is missing. The ``lmp`` half is covered
separately (``test_lammps_integration.py``, the ``lammps`` tier). The
``ace``/``grace`` dispatch branches in ``_build_committee_calculator`` are
covered against fake stand-in calculators in
``test_committee_selection_payload.py`` (no real tiny trained ACE/GRACE
committee potentials exist yet to do what this file does for MACE with the
real ``pyace``/``tensorpotential`` libraries -- promote those tests once
such potentials are available).

The committee models are borrowed from ``examples/mace_md_AL`` (Cu/Co/H); the
LAMMPS dump is synthesized here so no dump fixture needs shipping.
"""

import logging
from pathlib import Path

import pytest

pytest.importorskip("mace")
import ase.io  # noqa: E402
import numpy as np  # noqa: E402

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parent.parent
_MODELS = sorted((_REPO / "examples/mace_md_AL/inputs/models").glob("*.model"))

# Map LAMMPS dump 'type' (as read into atom.number) -> element symbol.
_ELEMENT_MAPPING = {"1": "Cu", "2": "Co", "3": "H"}


def _load_module(rel_path, name):
    import importlib.util

    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sampler_payload():
    return _load_module(
        "src/lammps_al/run/scripts/lammps_sampling.py", "lammps_payload"
    )


def _load_selector_payload():
    return _load_module(
        "src/lammps_al/select/scripts/committee_selection.py",
        "committee_selection_payload",
    )


def _write_dump(path, n_frames=4, box=8.0, seed=0):
    """Write a minimal multi-frame LAMMPS ``lammps-dump-text`` trajectory.

    Four atoms per frame with types 1,1,2,3 (Cu,Cu,Co,H after remapping),
    positions jittered per frame so the committee uncertainty varies.
    """
    rng = np.random.default_rng(seed)
    base = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [1.2, 1.2, 1.2]]
    )
    types = [1, 1, 2, 3]
    blocks = []
    for step in range(n_frames):
        pos = base + rng.uniform(-0.3, 0.3, base.shape)
        lines = [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(len(types)),
            "ITEM: BOX BOUNDS pp pp pp",
            f"0.0 {box}",
            f"0.0 {box}",
            f"0.0 {box}",
            "ITEM: ATOMS id type x y z",
        ]
        for i, (t, p) in enumerate(zip(types, pos), start=1):
            lines.append(f"{i} {t} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n")
    return path


@pytest.mark.skipif(
    len(_MODELS) < 2, reason="needs >=2 committee models (examples/mace_md_AL)"
)
def test_annotate_uncertainties_scores_every_frame(tmp_path):
    sampler_payload = _load_sampler_payload()
    selector_payload = _load_selector_payload()

    dump = _write_dump(tmp_path / "md.dump", n_frames=4)
    trajectory = tmp_path / "trajectory.extxyz"
    annotated_path = tmp_path / "annotated.extxyz"

    sampler_payload.convert_dump_to_extxyz(
        trajectory_file=dump,
        element_mapping=_ELEMENT_MAPPING,
        output_file=trajectory,
    )

    selector_payload.annotate_uncertainties(
        trajectory_file=trajectory,
        model_files=[str(m) for m in _MODELS[:2]],
        cutoff_radius=4.0,
        output_file=annotated_path,
        logger=logging.getLogger("test"),
        device="cpu",
    )

    # No reduction: every frame that went in comes back out, annotated.
    annotated = ase.io.read(annotated_path, index=":")
    assert len(annotated) == 4
    for at in annotated:
        assert list(at.symbols) == ["Cu", "Cu", "Co", "H"]
        unc = at.arrays["uncertainty"]
        assert unc.shape == (4,)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)
