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

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import ScriptInput, _require_path

# The payload's end-of-run ``write_data`` output, under ``<calcpath>/out/``;
# ``LammpsSampler(continue_from_previous=True)`` starts the next iteration's
# MD from it.
FINAL_DATA_NAME = "surface_final.data"


@dataclass
class LammpsSamplingInput(ScriptInput):
    """Input configuration for ``scripts/lammps_sampling.py``.

    This is the *MD-only* half of the sampling contract: it runs one LAMMPS
    trajectory and writes a proper ``trajectory.extxyz`` (correct chemical
    symbols applied via ``element_mapping``). Committee-uncertainty scoring
    and peak-frame selection is a separate actor/contract --
    ``lammps_al.select.selector.CommitteeUncertaintySelector`` /
    ``lammps_al.select.inputs.CommitteeSelectionInput`` -- split out so
    "generate dynamics" and "pick the most informative configs" are two
    loop roles, not one (mirroring this repo's existing
    ``ROIExtractor``/``EmbeddingAnnealer`` split).

    Attributes:
        lammps_path: Path to LAMMPS executable
        template_file: Name of LAMMPS input template file
        data_file: Name of initial structure data file
        model_files: List of paths to the committee's model files. Format
            depends on ``backend``: MACE ``.model`` files, an ACE
            pacemaker ``.yaml``/``.yace`` file per member, or a GRACE
            gracemaker model directory per member. Only ``model_files[0]``
            drives the MD here (the rest of the committee is only needed
            for scoring, downstream).
        element_mapping: Dict mapping atom numbers to element symbols,
            applied when converting the raw LAMMPS dump to
            ``out/trajectory.extxyz``.
        seed: Random seed for LAMMPS
        lammps_config_name: Name for generated LAMMPS input file
            (default: "lammps.in")
        dump_frequency: Steps between dumped MD frames (rendered into the
            template as ``{{ dump_frequency }}``). ``None`` (default) falls
            back to every step (dump_frequency=1) at render time -- fine
            for small systems, but for large ones (hundreds of thousands of
            atoms) this can produce a huge dump file in minutes; set
            explicitly for those.
        max_restarts: How often a LAMMPS run that exits non-zero is
            retried (default 0: not at all). A retry resumes from the
            newest restart file if the template writes them, else it
            starts over -- see ``scripts/lammps_sampling.py``'s module
            docstring for what the template has to do.
        restart_frequency: Steps between restart files, rendered as
            ``{{ restart_frequency }}`` (with ``{{ restart_file }}`` as the
            file pattern). ``None`` (default): no restart files, so a retry
            starts the run over.
        extra_template_kwargs: Arbitrary extra jinja2 render kwargs (e.g. a
            ``Schedule``'s ``temperature``/``timestep``/``n_run``),
            forwarded to the template verbatim alongside the fixed ones
            ``main()`` always sets. Populated by
            ``LammpsSampler.update()`` from any ``update(**kwargs)`` key
            that isn't one of this dataclass's own named fields, instead of
            being silently dropped as an "unknown key" (the same class of
            bug that once ate a ``dump_frequency`` override under the old,
            colliding ``sampling_frequency`` name).

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
    seed: int
    lammps_config_name: str = "lammps.in"
    # LAMMPS accelerator/package flags inserted between ``-in`` and ``-log``.
    # Default is KOKKOS-on-GPU (production, paired with the default
    # lammps_pair_style="mliap"); pass "" (or a CPU-appropriate variant) to
    # run a CPU-only build, e.g. in CI -- lammps_pair_style="mace" needs no
    # flags at all on CPU.
    lammps_args: str = "-k on g 1 -sf kk -pk kokkos newton on neigh half"
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
    # Which committee-potential family drives the MD:
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
    # Only affects the MD's driving potential here; committee-uncertainty
    # scoring (which also dispatches on this field) lives in
    # lammps_al.select.scripts.committee_selection.
    backend: str = "mace"
    dump_frequency: Optional[int] = None
    max_restarts: int = 0
    restart_frequency: Optional[int] = None
    extra_template_kwargs: dict = field(default_factory=dict)

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

        if self.dump_frequency is not None and self.dump_frequency <= 0:
            raise ScriptInputError(
                f"dump_frequency must be positive, got {self.dump_frequency}"
            )

        if self.max_restarts < 0:
            raise ScriptInputError(
                f"max_restarts must be >= 0, got {self.max_restarts}"
            )

        if self.restart_frequency is not None and self.restart_frequency <= 0:
            raise ScriptInputError(
                "restart_frequency must be positive, got "
                f"{self.restart_frequency}"
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
