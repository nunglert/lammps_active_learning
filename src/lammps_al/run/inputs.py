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
        model_files: List of paths to the committee's model files. Format
            depends on ``backend``: MACE ``.model`` files, an ACE
            pacemaker ``.yaml``/``.yace`` file per member, or a GRACE
            gracemaker model directory per member.
        element_mapping: Dict mapping atom numbers to element symbols
        cutoff_radius: Cutoff radius for local uncertainty aggregation
            (Angstrom)
        bin_count: Number of trajectory segments for sampling
        seed: Random seed for LAMMPS
        lammps_config_name: Name for generated LAMMPS input file
            (default: "lammps.in")
        dump_frequency: Steps between dumped MD frames (rendered into the
            template as ``{{ dump_frequency }}``). ``None`` (default) falls
            back to every step (dump_frequency=1) at render time -- fine
            for small systems, but for large ones (hundreds of thousands of
            atoms) this can produce a huge dump file in minutes; set
            explicitly for those. Deliberately a distinct name from
            ``LammpsSampler``'s pre-existing ``sampling_frequency`` (a
            different concept for ``lammps_dump_sampling.py``'s payload,
            mutually exclusive with ``num_samples`` there) -- reusing that
            name here collided with that exclusion logic and got silently
            dropped whenever ``num_samples`` was also set, confirmed as the
            cause of an unexpectedly large dump file, 2026-09-11.

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
    # Default is KOKKOS-on-GPU (production, paired with the default
    # lammps_pair_style="mliap"); pass "" (or a CPU-appropriate variant) to
    # run a CPU-only build, e.g. in CI -- lammps_pair_style="mace" needs no
    # flags at all on CPU.
    lammps_args: str = "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    # torch device for the MACE committee scoring ("cuda" or "cpu").
    device: str = "cuda"
    # Which LAMMPS/MACE coupling to compile the driving model for and render
    # into the template (as {{ lammps_pair_style }}):
    #   "mliap" (default, production): the ML-IAP Python bridge
    #     (`pair_style mliap unified`), matching the KOKKOS-on-GPU production
    #     setup. Needs `enable_cueq` below; LAMMPS's ghost-atom message
    #     passing for multi-layer models is Kokkos-CUDA-only, so this route
    #     does not run correctly CPU-only.
    #   "mace": LAMMPS's native `pair_style mace` (the `ML-MACE` package,
    #     LibTorch directly). Works CPU-only -- this is what the CI
    #     `lammps-smoke` job and local Docker-based testing use
    #     (docker/Dockerfile.lammps-mace), not a production default.
    lammps_pair_style: str = "mliap"
    # Convert the MACE model to cuequivariance before compiling it for LAMMPS
    # (production default). Only used when backend="mace" and
    # lammps_pair_style="mliap". Set False to compile the plain e3nn model
    # without the cuequivariance requirement (CPU builds); unused by
    # backend="ace"/"grace".
    enable_cueq: bool = True
    # Which committee-potential family drives both the MD and the
    # uncertainty scoring:
    #   "mace" (default): see lammps_pair_style above for the mliap/mace
    #     compile-route choice.
    #   "ace": pacemaker-fit ACE potentials (.yaml/.yace). LAMMPS's native
    #     `pair_style pace` reads the potential file directly -- no compile
    #     step. `lammps_pair_style` becomes the literal pair_style line
    #     rendered into the template (e.g. "pace").
    #   "grace": gracemaker-fit GRACE potentials (a model directory per
    #     member). LAMMPS's native `pair_style grace` reads the directory
    #     directly -- no compile step. `lammps_pair_style` becomes the
    #     literal pair_style line (e.g. "grace pad_verbose padding 0.1").
    # Committee uncertainty scoring dispatches per backend in
    # lammps_sampling.py: MACE's `MACECalculator` (forces_comm, stdev
    # computed here), ACE's `PyACEEnsembleCalculator`, or GRACE's
    # `TPCalculator` (both already return an aggregated `forces_std`).
    backend: str = "mace"
    dump_frequency: Optional[int] = None

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

        if self.dump_frequency is not None and self.dump_frequency <= 0:
            raise ScriptInputError(
                f"dump_frequency must be positive, got {self.dump_frequency}"
            )

        if self.device not in ("cpu", "cuda"):
            raise ScriptInputError(
                f"device must be 'cpu' or 'cuda', got {self.device!r}"
            )

        if self.backend not in ("mace", "ace", "grace"):
            raise ScriptInputError(
                "backend must be 'mace', 'ace', or 'grace', got "
                f"{self.backend!r}"
            )

        # The mliap/mace compile-route fork only exists for the MACE
        # backend; ace/grace's pair styles are native (no compile step), so
        # lammps_pair_style is just the literal pair_style line for them --
        # any non-empty string is valid.
        if self.backend == "mace" and self.lammps_pair_style not in (
            "mliap",
            "mace",
        ):
            raise ScriptInputError(
                "lammps_pair_style must be 'mliap' or 'mace' for "
                f"backend='mace', got {self.lammps_pair_style!r}"
            )
        if self.backend != "mace" and not self.lammps_pair_style:
            raise ScriptInputError(
                "lammps_pair_style must be a non-empty pair_style line for "
                f"backend={self.backend!r}"
            )

        if root is not None:
            _require_path(root, self.data_file, "data_file")
            _require_path(root, self.template_file, "template_file")
            for m in self.model_files:
                _require_path(root, m, "model file")
