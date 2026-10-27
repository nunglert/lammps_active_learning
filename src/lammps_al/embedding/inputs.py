"""Input schemas for the amorphous-matrix embedding remote payloads.

``MatrixEmbeddingInput`` and ``EmbeddingAnnealInput`` are the JSON contracts
between the host-side
:class:`lammps_al.embedding.extractor.ROIExtractor` /
:class:`lammps_al.embedding.annealer.EmbeddingAnnealer` (which write them)
and the cluster-side payloads ``scripts/extract.py`` and
``scripts/embedding_anneal.py`` (which load and validate them). They
subclass ``remote_handler``'s ``ScriptInput`` so they inherit the shared
``from_file`` / ``from_dict`` / ``validate`` machinery -- exactly as the
schemas used when they lived in ``remote_handler.scripts.input_models``.

Kept in this package (not in ``remote_handler``) because these actors and
their payloads are a *project* concern: ``remote_handler`` owns the
``Selector``/``Sampler`` interfaces and the remote-execution machinery,
while the concrete amorphous-matrix embedding actors -- and the input
contracts they speak -- belong here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import ScriptInput, _require_path

logger = logging.getLogger(__name__)


@dataclass
class MatrixEmbeddingInput(ScriptInput):
    """Input configuration for ``scripts/extract.py``.

    Attributes:
        trajectory: Filename of the uncertainty-tagged trajectory to extract
            regions of interest (ROIs) from (must carry a per-atom
            ``uncertainty`` array).
        threshold: Minimum per-atom uncertainty for an atom to seed a ROI.
        cluster_cutoff: Distance (Angstrom) under which two high-uncertainty
            atoms are grouped into the same ROI.
        box_x: Half-width of the carved box along x.
        box_y: Half-width of the carved box along y.
        box_z_below: Half-width of the carved box below the ROI along z.
        protect_radius: Radius around each ROI that cleanup never modifies.
        box_z_above: Half-width above the ROI along z (e.g. extra solvent
            headspace); defaults to ``box_z_below`` (a symmetric box).
        buffer: Extra padding added to each carved-cell dimension.
        topology_mode: ``"bulk"`` (no normal/rotation), ``"surface"`` (OVITO
            surface-mesh normal, rotated onto z), or ``"interface"`` (two
            condensed phases -- ``remove_species`` is stripped on an internal
            OVITO copy to expose a mesh at the interface plane).
        remove_species: Phase to strip before meshing. Required when
            ``topology_mode="interface"``.
        phase_species: Species defining the condensed phase. Used by the
            species-COM fallback heuristic if OVITO mesh construction
            fails, and (together with ``phase_cutoff``) to restrict ROI
            candidates to high-uncertainty atoms actually near this phase.
        phase_cutoff: Distance (Angstrom) for the ``phase_species``
            proximity filter in :func:`find_rois` -- keeps ROIs out of a
            bulk single-phase region with no condensed-phase core for
            ``EmbeddingAnnealer``'s frozen-sphere anneal to work with.
            Required together with ``phase_species`` for the filter to
            apply; ``None`` (with ``phase_species=None``) applies no
            filter, the right choice for ``topology_mode="bulk"``.
        cleanup_mode: ``"none"`` (raw cut box), ``"molecule"`` (reform
            molecules broken by the cut using a solvent template), or
            ``"stoichiometry"`` (trim the matrix to a target ratio).
        solvent_label: An ``ase.build.molecule`` g2 database label (e.g.
            ``"H2O"``). Exactly one of ``solvent_label``/``solvent_file`` is
            required when ``cleanup_mode="molecule"``.
        solvent_file: Path to a staged ASE-readable file holding a custom
            solvent template ``Atoms``.
        stoichiometry: Target ratio dict (e.g. ``{"H": 2, "O": 1}``).
            Required when ``cleanup_mode="stoichiometry"``.
        specorder: Species order for ``lammps-data`` output; defaults to the
            order species are encountered.
        smoothing_level: OVITO ``ConstructSurfaceMeshModifier`` smoothing.
        mesh_radius: OVITO probe-sphere radius; ``None`` uses OVITO's
            default.
        normal_patch_radius: Approximate geodesic radius over which
            edge-connected surface-mesh facet normals are area-averaged
            after anchoring at the closest surface point (see
            ``surface_mesh.detect_normal``). This smooths local atomic-scale
            roughness without mixing in a disconnected nearby surface.
            ``None`` (default) uses ``protect_radius``.
        max_num_structures: Caps the total number of carved boxes across the
            whole trajectory. When the number of ROIs :func:`find_rois`
            turns up (summed over every frame) exceeds this, only the
            ``max_num_structures`` with the highest per-atom uncertainty are
            actually carved/cleaned/written -- the rest are dropped before
            any of that per-ROI work runs. An uncalibrated ``threshold``
            combined with a spatially fragmented interface can otherwise
            produce far more ROIs than intended, each one becoming its own
            downstream ``EmbeddingAnnealer`` SLURM job. Ranking by
            uncertainty (not e.g. frame order) keeps the genuinely most
            informative configs regardless of where in the trajectory they
            fall. ``None`` (default) applies no cap.

    Example:
        >>> config = MatrixEmbeddingInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    _path_fields: ClassVar[tuple[str, ...]] = ("trajectory", "solvent_file")

    trajectory: str
    threshold: float
    cluster_cutoff: float
    box_x: float
    box_y: float
    box_z_below: float
    protect_radius: float
    box_z_above: Optional[float] = None
    buffer: float = 2.0
    topology_mode: str = "bulk"
    remove_species: Optional[list[str]] = None
    phase_species: Optional[list[str]] = None
    phase_cutoff: Optional[float] = None
    cleanup_mode: str = "none"
    solvent_label: Optional[str] = None
    solvent_file: Optional[Path] = None
    stoichiometry: Optional[dict[str, int]] = None
    specorder: Optional[list[str]] = None
    smoothing_level: int = 8
    mesh_radius: Optional[float] = None
    normal_patch_radius: Optional[float] = None
    max_num_structures: Optional[int] = None

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        valid_topology = {"bulk", "surface", "interface"}
        if self.topology_mode not in valid_topology:
            raise ScriptInputError(
                f"topology_mode must be one of {valid_topology}, got "
                f"'{self.topology_mode}'"
            )

        valid_cleanup = {"none", "molecule", "stoichiometry"}
        if self.cleanup_mode not in valid_cleanup:
            raise ScriptInputError(
                f"cleanup_mode must be one of {valid_cleanup}, got "
                f"'{self.cleanup_mode}'"
            )

        if self.threshold < 0:
            raise ScriptInputError("threshold must be non-negative")

        if self.cluster_cutoff <= 0:
            raise ScriptInputError("cluster_cutoff must be positive")

        if self.protect_radius <= 0:
            raise ScriptInputError("protect_radius must be positive")

        if self.box_x <= 0 or self.box_y <= 0 or self.box_z_below <= 0:
            raise ScriptInputError(
                "box_x, box_y and box_z_below must all be positive"
            )

        if self.box_z_above is not None and self.box_z_above <= 0:
            raise ScriptInputError("box_z_above must be positive if given")

        if (
            self.normal_patch_radius is not None
            and self.normal_patch_radius <= 0
        ):
            raise ScriptInputError(
                "normal_patch_radius must be positive if given"
            )

        if self.phase_cutoff is not None:
            if self.phase_cutoff <= 0:
                raise ScriptInputError(
                    "phase_cutoff must be positive if given"
                )
            if not self.phase_species:
                raise ScriptInputError(
                    "phase_cutoff requires phase_species to also be given"
                )

        if (
            self.max_num_structures is not None
            and self.max_num_structures <= 0
        ):
            raise ScriptInputError(
                "max_num_structures must be positive if given"
            )

        if self.buffer < 0:
            raise ScriptInputError("buffer must be non-negative")

        if self.topology_mode == "interface" and not self.remove_species:
            raise ScriptInputError(
                "remove_species is required when topology_mode='interface'"
            )

        if self.cleanup_mode == "molecule":
            if bool(self.solvent_label) == bool(self.solvent_file):
                raise ScriptInputError(
                    "cleanup_mode='molecule' requires exactly one of "
                    "solvent_label or solvent_file"
                )

        if self.cleanup_mode == "stoichiometry":
            if not self.stoichiometry:
                raise ScriptInputError(
                    "cleanup_mode='stoichiometry' requires a non-empty "
                    "stoichiometry dict"
                )
            if any(v <= 0 for v in self.stoichiometry.values()):
                raise ScriptInputError(
                    "stoichiometry ratios must all be positive"
                )

        if root is not None:
            _require_path(root, self.trajectory, "trajectory")
            if self.solvent_file is not None:
                _require_path(root, self.solvent_file, "solvent_file")


@dataclass
class EmbeddingAnnealInput(ScriptInput):
    """Input configuration for ``scripts/embedding_anneal.py``.

    Attributes:
        data_file: Filename of the carved/cleaned-up structure (LAMMPS data
            format) to anneal.
        lammps_path: Path to the LAMMPS executable.
        template_file: Name of the jinja2 LAMMPS deck template to render.
        cutoff: Radius of the ``fixSphere`` region kept frozen around the
            region of interest (ROI).
        roi_x: ROI position, x.
        roi_y: ROI position, y.
        roi_z: ROI position, z.
        pair_style: LAMMPS ``pair_style`` line contents (e.g.
            ``"grace pad_verbose padding 0.1"``).
        species: Element symbols in LAMMPS type order (type 1, 2, ...).
        masses: Atomic masses aligned with ``species``.
        pair_coeff_path: Path to the potential file or directory (e.g. a
            GRACE model, normally saved as a directory) for
            ``pair_coeff * * <path> <species...>``.
        stages: One dict per anneal stage, run in order. Each stage is
            frozen (``setforce 0``) while every *other* stage's atoms are
            annealed; the ``fixSphere`` ROI is frozen throughout every
            stage regardless. A single-entry list (e.g. one stage covering
            all species) reproduces the old ``bulk``/``surface``
            single-stage anneal; two or more reproduce (and generalize) the
            old ``interface`` two-stage anneal. Species must not repeat
            across stages. Each dict has keys:

            * ``species`` (list[str]): this stage's species group.
            * ``temperatures`` (list[float]): temperature waypoints (K),
              e.g. ``[300, 600, 300]`` for a ramp-up-then-down; a flat
              ``[300, 300]`` for one constant-temperature segment. Must
              have at least 2 entries.
            * ``run_steps`` (list[int]): steps for each segment between
              consecutive waypoints -- ``len(run_steps) ==
              len(temperatures) - 1``.
            * ``pressure_xy`` (float, optional): target in-plane pressure
              (bar) for NPT dilation for this stage; omit/``None`` leaves
              x/y uncoupled.
            * ``pressure_z`` (float, optional): target out-of-plane
              pressure (bar); omit/``None`` leaves z uncoupled. A stage
              with both pressures unset runs NVT (thermostat only, no
              barostat) instead of NPT.
            * ``z_wall_margin``/``z_wall_epsilon``/``z_wall_cutoff``
              (float, optional): per-stage override of the deck-wide
              ``z_wall_margin``/``z_wall_epsilon``/``z_wall_cutoff``
              below -- e.g. a stiffer or more relaxed z-confinement wall
              for this stage's species than the deck default. Only
              relevant when this stage has no ``pressure_z`` (a
              barostatted stage never gets a z-wall, regardless of these).
              (There is deliberately no equivalent per-stage override for
              the ROI repulsive wall below -- that one protects the frozen
              core itself and stays a single, deck-wide, always-active
              wall shared by every stage's mobile atoms.)
        timestep: MD timestep (ps).
        seed: Random seed for velocity initialization.
        dump_every: Dump-output frequency (steps).
        lammps_flags: Extra command-line flags passed verbatim to ``lmp``
            (inserted between ``-in`` and ``-log``). The correct acceleration
            flags depend on the LAMMPS build and force field, so they are not
            hard-coded; e.g. ``"-k on g 1 -sf kk -pk kokkos newton on neigh
            half"`` for a single-GPU Kokkos build. Empty runs a plain,
            unaccelerated ``lmp``.
        n_processors: MPI ranks to run ``lmp`` under (``mpirun -np N``).
            ``1`` (default) runs ``lmp`` directly, unchanged from a bare
            invocation. Only effective if the ``lmp`` binary itself was
            built with MPI enabled -- LAMMPS silently runs single-rank
            under a non-MPI build regardless of this value.
        lammps_config_name: Name for the rendered LAMMPS input file.
        roi_repel_margin: Gap (Angstrom) subtracted from ``cutoff`` to place
            a harmonic repulsive wall (``region ... side out`` + ``fix
            wall/region``) that keeps every mobile atom from approaching
            closer than ``cutoff - roi_repel_margin`` to the ROI. Without
            this, nothing stops the annealed shell from diffusing into (or
            through) the frozen ``fixSphere`` core -- only the pair
            potential's own short-range repulsion would resist an overlap,
            which is not a reliable safeguard this close to a rigid,
            non-relaxing boundary. Set to ``0.0`` to place the wall exactly
            at ``cutoff`` (no buffer); the wall itself can be disabled by
            setting ``roi_repel_epsilon=0.0``. Deliberately a single,
            deck-wide setting shared by every stage's mobile atoms (unlike
            ``z_wall_margin`` below, which is tunable per stage): this
            protects the frozen core itself and should stay uniformly
            active regardless of which species is currently mobile, rather
            than trusting the pair potential's own short-range repulsion to
            exclude some phases on its own. Created once (right after the
            initial settling ``run 0``) and never removed, so it stays
            active for the entire rest of the anneal, whichever stage is
            currently being annealed.
        roi_repel_epsilon: Harmonic force constant for the ROI repulsive
            wall. ``0.0`` disables it.
        roi_repel_wall_cutoff: Interaction range (Angstrom) of the ROI
            repulsive wall -- the harmonic force is only felt within this
            distance of the wall surface.
        halt_max_kinetic_energy: Stop the anneal (``fix halt``, checked
            every step) once any mobile atom's kinetic energy exceeds
            this (eV) -- the sign of an atom launched by a hole in the
            potential. The box then returns the last dumped frame before
            the halt as its sample (see ``scripts/halted.py``). ``None``
            drops this criterion.
        halt_max_force: Same, for the largest force on a mobile atom
            (eV/Angstrom). ``None`` (default) drops this criterion; with
            both ``None`` there is no halt check. Off by default because
            an under-trained potential can give one-step force spikes
            (over 100 eV/Angstrom) that a heavy atom shrugs off -- an atom
            that is really launched shows up in its kinetic energy.
        z_wall_margin: Deck-wide default padding (Angstrom) added around
            each stage's mobile atoms' initial z-extent (measured fresh at
            the start of that stage) to build a pair of ``wall/harmonic``
            z-confinement walls. Only applied to stages where
            ``pressure_z`` is unset -- when a stage already barostats z, a
            static wall would fight the box rescaling, so it is skipped
            there (mirroring why the z-wall reference this was ported from
            only ever enabled it for the non-``pressure_z`` stage). Without
            this, an unbarostatted stage has nothing stopping its mobile
            atoms drifting into vacuum/headspace above an interface box.
            Unlike ``roi_repel_margin`` above (created once, permanent for
            the whole run), this wall is created fresh and torn down within
            each stage's own block -- it exists only while that stage's own
            segments are running. Any stage may override this with its own
            ``z_wall_margin`` key
            (see ``stages`` above); disable per-stage by setting that
            stage's ``z_wall_epsilon=0.0``, or deck-wide with this
            ``z_wall_epsilon=0.0``.
        z_wall_epsilon: Deck-wide default harmonic force constant for the
            z-confinement walls. ``0.0`` disables them for every stage that
            doesn't set its own ``z_wall_epsilon``.
        z_wall_cutoff: Deck-wide default interaction range (Angstrom) of the
            z-confinement walls.

    Example:
        >>> config = EmbeddingAnnealInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    _path_fields: ClassVar[tuple[str, ...]] = ("lammps_path",)

    data_file: str
    lammps_path: Path
    template_file: str
    cutoff: float
    roi_x: float
    roi_y: float
    roi_z: float
    pair_style: str
    pair_coeff_path: str
    species: list[str]
    masses: list[float]
    stages: list[dict]
    timestep: float = 0.0005
    seed: int = 42
    dump_every: int = 1000
    lammps_flags: str = ""
    n_processors: int = 1
    lammps_config_name: str = "lammps.in"
    roi_repel_margin: float = 0.5
    roi_repel_epsilon: float = 10.0
    roi_repel_wall_cutoff: float = 1.0
    halt_max_kinetic_energy: Optional[float] = 5.0
    halt_max_force: Optional[float] = None
    z_wall_margin: float = 1.0
    z_wall_epsilon: float = 10.0
    z_wall_cutoff: float = 1.0

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if self.cutoff <= 0:
            raise ScriptInputError("cutoff must be positive")

        if self.n_processors <= 0:
            raise ScriptInputError("n_processors must be positive")

        if not self.species:
            raise ScriptInputError("species cannot be empty")

        if len(self.species) != len(self.masses):
            raise ScriptInputError(
                "species and masses must have the same length"
            )

        if not self.stages:
            raise ScriptInputError("stages cannot be empty")

        seen_species: set[str] = set()
        for i, stage in enumerate(self.stages):
            missing = {"species", "temperatures", "run_steps"} - set(stage)
            if missing:
                raise ScriptInputError(
                    f"stages[{i}] is missing required key(s): "
                    f"{sorted(missing)}"
                )

            phase = stage["species"]
            if not phase:
                raise ScriptInputError(
                    f"stages[{i}]['species'] cannot be empty"
                )
            unknown = set(phase) - set(self.species)
            if unknown:
                raise ScriptInputError(
                    f"stages[{i}]['species'] contains species not in "
                    f"'species': {sorted(unknown)}"
                )
            overlap = seen_species & set(phase)
            if overlap:
                raise ScriptInputError(
                    "stages must not share species across entries; "
                    f"{sorted(overlap)} appear in more than one stage"
                )
            seen_species |= set(phase)

            temperatures = stage["temperatures"]
            run_steps = stage["run_steps"]
            if len(temperatures) < 2:
                raise ScriptInputError(
                    f"stages[{i}]['temperatures'] must have at least 2 "
                    f"waypoints, got {len(temperatures)}"
                )
            if len(run_steps) != len(temperatures) - 1:
                raise ScriptInputError(
                    f"stages[{i}]['run_steps'] must have one entry per "
                    f"segment (len(temperatures) - 1 = "
                    f"{len(temperatures) - 1}), got {len(run_steps)}"
                )
            if any(steps <= 0 for steps in run_steps):
                raise ScriptInputError(
                    f"stages[{i}]['run_steps'] must all be positive"
                )
            if stage.get("pressure_xy") is None and (
                stage.get("pressure_z") is None
            ):
                logger.info(
                    "stages[%d] has no pressure_xy/pressure_z -- runs NVT "
                    "(no barostat) for this stage.",
                    i,
                )

            # Optional per-stage overrides of the deck-wide z-wall strength
            # (falls back to z_wall_margin/epsilon/cutoff when absent).
            if "z_wall_margin" in stage and stage["z_wall_margin"] < 0:
                raise ScriptInputError(
                    f"stages[{i}]['z_wall_margin'] must be non-negative"
                )
            if "z_wall_epsilon" in stage and stage["z_wall_epsilon"] < 0:
                raise ScriptInputError(
                    f"stages[{i}]['z_wall_epsilon'] must be non-negative"
                )
            if "z_wall_cutoff" in stage and stage["z_wall_cutoff"] <= 0:
                raise ScriptInputError(
                    f"stages[{i}]['z_wall_cutoff'] must be positive"
                )

        if not self.pair_coeff_path:
            raise ScriptInputError("pair_coeff_path cannot be empty")

        if not self.lammps_path.exists():
            raise ScriptInputError(
                f"LAMMPS executable not found: {self.lammps_path}"
            )

        if self.roi_repel_margin < 0:
            raise ScriptInputError("roi_repel_margin must be non-negative")
        if self.roi_repel_margin >= self.cutoff:
            raise ScriptInputError(
                "roi_repel_margin must be smaller than cutoff (got "
                f"roi_repel_margin={self.roi_repel_margin} >= "
                f"cutoff={self.cutoff})"
            )
        if self.roi_repel_epsilon < 0:
            raise ScriptInputError("roi_repel_epsilon must be non-negative")
        if self.roi_repel_wall_cutoff <= 0:
            raise ScriptInputError("roi_repel_wall_cutoff must be positive")
        for name in ("halt_max_kinetic_energy", "halt_max_force"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ScriptInputError(f"{name} must be positive or None")

        if self.z_wall_margin < 0:
            raise ScriptInputError("z_wall_margin must be non-negative")
        if self.z_wall_epsilon < 0:
            raise ScriptInputError("z_wall_epsilon must be non-negative")
        if self.z_wall_cutoff <= 0:
            raise ScriptInputError("z_wall_cutoff must be positive")

        if root is not None:
            _require_path(root, self.data_file, "data_file")
            _require_path(root, self.template_file, "template_file")
            _require_path(root, self.pair_coeff_path, "pair_coeff_path")
