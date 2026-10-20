"""Cluster-side remote payloads shipped by lammps_al's labelling side.

:class:`lammps_al.label.evaluator.VaspD3Evaluator` ships
``vasp_d3_singlepoint.py`` as its single-point payload, mirroring
``lammps_al.run.scripts`` / ``lammps_al.select.scripts``. The payload is
copied to the remote working directory and run there, not imported in the
normal flow (its functions are import-safe for testing).
"""

from pathlib import Path


def script_dir() -> Path:
    """Return the directory holding lammps_al's labelling payloads."""
    return Path(__file__).resolve().parent


__all__ = ["script_dir"]
