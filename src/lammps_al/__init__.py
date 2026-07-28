"""lammps_active_learning -- composition root for LAMMPS active-learning runs.

A thin *project* layer over ``remote_handler``: it owns the concrete LAMMPS
MD-uncertainty sampler and the cluster-side payload it ships, while
``remote_handler`` provides the orchestration loop, the ``Sampler`` /
``Evaluator`` / ``Trainer`` interfaces, and the remote-execution machinery.

The concrete ``LammpsSampler`` is intentionally **not** imported here: importing
it pulls in ``remote_handler`` (the ``[run]`` extra), which lightweight,
cluster-free consumers should not need. Import it via
``lammps_al.run.sampler``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is the git tag (via setuptools-scm); report the
    # installed package version.
    __version__ = version("lammps_active_learning")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
