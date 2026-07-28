"""Input schema for the LAMMPS sampling remote payload.

``LammpsSamplingInput`` is the JSON contract between the host-side
:class:`lammps_al.run.sampler.LammpsSampler` (which writes it) and the
cluster-side payload ``scripts/lammps_sampling.py`` (which loads and validates
it). It subclasses ``remote_handler``'s ``ScriptInput`` so it inherits the
shared ``from_file`` / ``from_dict`` / ``validate`` machinery -- exactly as the
payload used when this schema lived in ``remote_handler.scripts.input_models``.

Kept in this package (not in ``remote_handler``) because the sampler and its
payload are a *project* concern: ``remote_handler`` owns the ``Sampler``
interface and the remote-execution machinery, while the concrete LAMMPS
MD-uncertainty sampler -- and the input contract it speaks -- belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import ScriptInput, _require_path


@dataclass
class LammpsSamplingInput(ScriptInput):
    """Input configuration for ``scripts/lammps_sampling.py``.

    Attributes:
        lammps_path: Path to LAMMPS executable
        template_file: Name of LAMMPS input template file
        data_file: Name of initial structure data file
        model_files: List of paths to MACE model files (for committee)
        element_mapping: Dict mapping atom numbers to element symbols
        cutoff_radius: Cutoff radius for local uncertainty aggregation
            (Angstrom)
        bin_count: Number of trajectory segments for sampling
        seed: Random seed for LAMMPS
        lammps_config_name: Name for generated LAMMPS input file
            (default: "lammps.in")

    Example:
        >>> config = LammpsSamplingInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    _path_fields: ClassVar[tuple[str, ...]] = ("lammps_path",)

    lammps_path: Path
    template_file: str
    data_file: str
    model_files: list[str]
    element_mapping: dict[str, str]
    cutoff_radius: float
    bin_count: int
    seed: int
    lammps_config_name: str = "lammps.in"
    # LAMMPS accelerator/package flags inserted between ``-in`` and ``-log``.
    # Default is KOKKOS-on-GPU (production); pass a CPU variant (or "") to run
    # a non-GPU build, e.g. in CI.
    lammps_args: str = "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    # torch device for the MACE committee scoring ("cuda" or "cpu").
    device: str = "cuda"
    # Convert the MACE model to cuequivariance before compiling it for LAMMPS
    # (production default). False compiles the plain e3nn model and drops the
    # cuequivariance requirement, so a CPU-only build (e.g. CI) can run.
    enable_cueq: bool = True

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.lammps_path.exists():
            raise ScriptInputError(
                f"LAMMPS executable not found: {self.lammps_path}"
            )

        if not self.model_files:
            raise ScriptInputError("model_files cannot be empty")

        if self.cutoff_radius <= 0:
            raise ScriptInputError(
                f"cutoff_radius must be positive, got {self.cutoff_radius}"
            )

        if self.bin_count <= 0:
            raise ScriptInputError(
                f"bin_count must be positive, got {self.bin_count}"
            )

        if self.device not in ("cpu", "cuda"):
            raise ScriptInputError(
                f"device must be 'cpu' or 'cuda', got {self.device!r}"
            )

        if root is not None:
            _require_path(root, self.data_file, "data_file")
            _require_path(root, self.template_file, "template_file")
            for m in self.model_files:
                _require_path(root, m, "model file")
