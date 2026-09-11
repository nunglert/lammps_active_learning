"""Cluster-side remote payloads shipped by lammps_al's select side.

``CommitteeUncertaintySelector`` ships a Python payload
(``committee_selection.py``) to the cluster and runs it there via a
``remote_handler`` ``SingleScriptTask``, mirroring
``lammps_al.run.scripts``.

The payload is packaged as package-data (see
``[tool.setuptools.package-data]`` in ``pyproject.toml``) so :func:`script_dir`
locates it from a non-editable install too. It is *not* meant to be imported
as a module (it runs top-level code from ``sys.argv``); the selector passes
its filesystem path to the task, which copies it to the remote working
directory.
"""

from pathlib import Path


def script_dir() -> Path:
    """Return the directory holding lammps_al's select-side payloads."""
    return Path(__file__).resolve().parent


__all__ = ["script_dir"]
