"""Cluster-side remote payloads for the amorphous-matrix embedding sampler.

``AmorphousEmbeddingSampler`` ships two Python payloads to the cluster and
runs each there via a ``remote_handler`` ``SingleScriptTask``:
``extract.py`` (region-of-interest extraction, box carving, and cleanup) and
``embedding_anneal.py`` (the multi-stage LAMMPS anneal). They live here
rather than in ``remote_handler`` -- the orchestrator must not know about the
amorphous-embedding method.

The payloads are packaged as package-data (see
``[tool.setuptools.package-data]`` in ``pyproject.toml``) so :func:`script_dir`
locates them from a non-editable install too. They are *not* meant to be
imported as modules (each runs top-level code from ``sys.argv``); the sampler
passes their filesystem paths to the tasks, which copy them to the remote
working directory. ``cleanup.py``/``geometry.py``/``surface_mesh.py`` *are*
imported (by ``extract.py``, and by this project's tests), so the remote
extraction env needs ``lammps_al`` installed.
"""

from pathlib import Path


def script_dir() -> Path:
    """Return the dir holding the embedding sampler's cluster payloads."""
    return Path(__file__).resolve().parent


__all__ = ["script_dir"]
