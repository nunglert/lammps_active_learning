"""Cluster-side remote payloads shipped by lammps_al.

``LammpsSampler`` ships a Python payload (``lammps_sampling.py``) to the cluster
and runs it there via a ``remote_handler`` ``SingleScriptTask``. The payload
lives here rather than in ``remote_handler`` -- the orchestrator must not know
about the LAMMPS/MACE sampling method.

The payload is packaged as package-data (see ``[tool.setuptools.package-data]``
in ``pyproject.toml``) so :func:`script_dir` locates it from a non-editable
install too. It is *not* meant to be imported as a module (it runs top-level
code from ``sys.argv``); the sampler passes its filesystem path to the task,
which copies it to the remote working directory.
"""

from pathlib import Path


def script_dir() -> Path:
    """Return the directory holding lammps_al's cluster-side payloads."""
    return Path(__file__).resolve().parent


__all__ = ["script_dir"]
