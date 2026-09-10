"""Integration test for the MACE-committee scoring half of the payload.

Exercises ``get_samples_from_dump`` end-to-end on CPU with the default
``backend="mace"``: parse a LAMMPS dump, score every frame's committee
force-uncertainty with a real (small) MACE committee, aggregate locally, and
select the peak-uncertainty frame per segment -- writing ``samples.xyz`` and
the annotated full trajectory. This needs the MACE stack (``[run]`` extra)
and a couple of committee models, but **no LAMMPS binary and no GPU**; it
self-skips when either is missing. The ``lmp`` half is covered separately
(``test_lammps_integration.py``, the ``lammps`` tier). The ``ace``/``grace``
dispatch branches in ``_build_committee_calculator`` are covered against fake
stand-in calculators in ``test_payload_unit.py`` (no real tiny trained
ACE/GRACE committee potentials exist yet to do what this file does for MACE
with the real ``pyace``/``tensorpotential`` libraries -- promote those tests
once such potentials are available).

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


def _load_payload():
    import importlib.util

    path = _REPO / "src/lammps_al/run/scripts/lammps_sampling.py"
    spec = importlib.util.spec_from_file_location("lammps_payload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def test_get_samples_from_dump_scores_and_selects(tmp_path):
    payload = _load_payload()

    dump = _write_dump(tmp_path / "md.dump", n_frames=4)
    samples = tmp_path / "samples.xyz"
    full = tmp_path / "full.xyz"

    payload.get_samples_from_dump(
        trajectory_file=dump,
        model_files=[str(m) for m in _MODELS[:2]],
        element_mapping=_ELEMENT_MAPPING,
        cutoff_radius=4.0,
        bin_count=2,
        samples_file=samples,
        logger=logging.getLogger("test"),
        output_full_uncertainties_traj=True,
        full_file=full,
        device="cpu",
    )

    # One selected frame per segment (bin_count=2).
    selected = ase.io.read(samples, index=":")
    assert len(selected) == 2
    for at in selected:
        assert list(at.symbols) == ["Cu", "Cu", "Co", "H"]

    # Every frame of the annotated trajectory carries per-atom uncertainties.
    annotated = ase.io.read(full, index=":")
    assert len(annotated) == 4
    for at in annotated:
        unc = at.arrays["uncertainty"]
        assert unc.shape == (4,)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)
