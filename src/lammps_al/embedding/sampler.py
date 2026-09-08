"""The amorphous-matrix embedding Sampler for this project.

In the layered architecture, concrete samplers belong to the *project*, not
to the orchestrator: ``remote_handler`` owns the ``Sampler`` interface and
the generic remote-execution machinery (``SingleScriptTask``,
``ClusterConfig``, the SLURM templating), while the application picks and
wires the actual sampling method.

``AmorphousEmbeddingSampler`` is a **native** implementation of that
interface. Per AL iteration it:

    1. extracts regions of interest (ROIs) from an uncertainty-tagged
       trajectory, carves/orients/cleans up an amorphous-matrix box around
       each (``scripts/extract.py``, run via a ``SingleScriptTask``, needs
       the ``embedding`` extra remotely when ``topology_mode`` is
       ``"surface"``/``"interface"``);
    2. anneals every carved box with a multi-stage LAMMPS deck around a
       frozen ROI core (``scripts/embedding_anneal.py``, one
       ``SingleScriptTask`` per box, needs a LAMMPS build).

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler).
The cluster env that runs ``scripts/extract.py`` needs ``lammps_al`` (for the
input schema and the geometry/cleanup/surface-mesh helpers it imports) plus
``ovito`` when ``topology_mode`` is ``"surface"``/``"interface"``; the env
that runs ``scripts/embedding_anneal.py`` needs ``lammps_al`` and a LAMMPS
build. The host side needs just ``remote_handler`` + ``ase`` + ``numpy``.

``stub=True`` dry-runs the loop (mock boxes, no cluster) for testing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import ase.io
import numpy as np
from ase import Atoms
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.exceptions import WorkflowError
from remote_handler.workflows.active_learning_base import Sampler

from lammps_al.embedding.scripts import script_dir

logger = logging.getLogger(__name__)


class AmorphousEmbeddingSampler(Sampler):
    """
    Extracts regions of interest (ROIs) from an uncertainty-tagged
    trajectory, carves/orients/cleans up an amorphous-matrix box around
    each, and anneals every box with a multi-stage LAMMPS deck.

    Two remote stages, each a separate ``SingleScriptTask`` (potentially on
    different machines/envs -- see ``anneal_cluster_config``/
    ``anneal_remote_fs``): extraction
    (``scripts/extract.py``, needs the ``embedding`` extra when
    ``topology_mode`` is ``"surface"``/``"interface"``) and, one task per
    extracted box, annealing (``scripts/embedding_anneal.py``, needs a
    LAMMPS build).

    Args:
        remote_fs: FileSystemHandlerBase for the extraction stage
        cluster_config: ClusterConfig for the extraction stage
        trajectory_path: Path to the uncertainty-tagged trajectory (must
            carry a per-atom ``uncertainty`` array)
        threshold: Minimum per-atom uncertainty for an atom to seed a ROI
        cluster_cutoff: Distance (Angstrom) grouping high-uncertainty atoms
            into the same ROI
        box_x: Half-width of the carved box along x
        box_y: Half-width of the carved box along y
        box_z_below: Half-width of the carved box below the ROI along z
        protect_radius: Radius around each ROI that cleanup never modifies
        lammps_path: Path to the LAMMPS executable on the anneal machine
        pair_style: LAMMPS ``pair_style`` line contents
        pair_coeff_path: Path to the potential file or directory (e.g. a
            GRACE model, normally saved as a directory)
        species: Element symbols in LAMMPS type order
        masses: Atomic masses aligned with ``species``
        anneal_stages: One dict per anneal stage, run in order; each stage
            is frozen while every other stage anneals (the ROI stays frozen
            throughout every stage). A single-entry list is a single-stage
            anneal; two or more chain stages, each annealed while the
            others are held still. Species must not repeat across stages.
            Each dict has keys:

            * ``species`` (list[str]): this stage's species group.
            * ``temperatures`` (list[float]): temperature waypoints (K),
              e.g. ``[300, 600, 300]`` for a ramp-up-then-down, or any
              longer sequence for more segments; a flat ``[300, 300]`` for
              one constant-temperature segment. At least 2 entries.
            * ``run_steps`` (list[int]): steps for each segment between
              consecutive waypoints -- ``len(run_steps) ==
              len(temperatures) - 1``.
            * ``pressure_xy`` (float, optional): target in-plane pressure
              (bar) for NPT dilation for this stage; omit/``None`` leaves
              x/y uncoupled.
            * ``pressure_z`` (float, optional): target out-of-plane
              pressure (bar); omit/``None`` leaves z uncoupled. A stage
              with both pressures unset runs NVT instead of NPT.

            Example, a two-stage metal-then-solvent anneal::

                anneal_stages=[
                    {
                        "species": ["Cu"],
                        "temperatures": [300, 600, 300],
                        "run_steps": [50000, 100000],
                        "pressure_xy": 1.0,
                    },
                    {
                        "species": ["H", "O"],
                        "temperatures": [300, 300],
                        "run_steps": [50000],
                        "pressure_z": 1.0,
                    },
                ]
        box_z_above: Half-width above the ROI along z; defaults to
            ``box_z_below`` (symmetric box)
        buffer: Extra padding added to each carved-cell dimension
        topology_mode: ``"bulk"``, ``"surface"``, or ``"interface"`` --
            see ``MatrixEmbeddingInput``/``EmbeddingAnnealInput``
        remove_species: Phase stripped (on an internal OVITO copy) before
            meshing for the interface normal. Required for ``"interface"``.
        phase_species: Species defining the condensed phase, used only by
            the normal-detection fallback heuristic
        cleanup_mode: ``"none"``, ``"molecule"``, or ``"stoichiometry"``
        solvent: For ``cleanup_mode="molecule"``: either an ASE ``Atoms``
            solvent template (staged and sent as a file) or an
            ``ase.build.molecule`` g2 label string (e.g. ``"H2O"``)
        stoichiometry: Target ratio dict for ``cleanup_mode="stoichiometry"``
        specorder: Species order for ``lammps-data`` output
        smoothing_level: OVITO ``ConstructSurfaceMeshModifier`` smoothing
        mesh_radius: OVITO probe-sphere radius; ``None`` uses OVITO's default
        normal_patch_radius: Approximate geodesic radius over which
            edge-connected surface-mesh facet normals are area-averaged
            after anchoring at the closest surface point. This smooths local
            atomic-scale roughness without mixing in a disconnected nearby
            surface. ``None`` (default) uses ``protect_radius``.
        anneal_cutoff: Radius of the anneal's frozen ``fixSphere``; defaults
            to ``protect_radius``
        anneal_template_path: Jinja2 LAMMPS deck template; defaults to the
            shipped ``scripts/templates/embedding_anneal.in.j2``
        roi_repel_margin: See ``EmbeddingAnnealInput.roi_repel_margin`` --
            gap subtracted from ``anneal_cutoff`` for the ROI's repulsive
            wall. A single setting shared by every stage's mobile atoms
            (unlike ``z_wall_margin`` below, which is per-stage): the wall
            protects the frozen core itself, so it stays uniformly active
            regardless of which species is currently mobile. Disable the
            wall with ``roi_repel_epsilon=0.0``.
        roi_repel_epsilon: See ``EmbeddingAnnealInput.roi_repel_epsilon``.
        roi_repel_wall_cutoff: See
            ``EmbeddingAnnealInput.roi_repel_wall_cutoff``.
        z_wall_margin: See ``EmbeddingAnnealInput.z_wall_margin`` -- deck-wide
            default padding around each stage's mobile atoms' initial
            z-extent, used by any stage in ``anneal_stages`` that doesn't
            set its own ``z_wall_margin``/``z_wall_epsilon``/
            ``z_wall_cutoff`` keys (e.g. to tune the wall differently for a
            rigid metal-lattice stage vs. a mobile solvent stage). Disable
            deck-wide with ``z_wall_epsilon=0.0``.
        z_wall_epsilon: See ``EmbeddingAnnealInput.z_wall_epsilon``. ``None``
            (default) auto-decides from ``topology_mode``: ``10.0`` for
            ``"surface"``/``"interface"`` (where a free surface means there
            may be vacuum/solvent headspace along z for atoms to drift
            into), ``0.0`` (disabled) for ``"bulk"`` (fully periodic in z --
            there is no vacuum to escape into, so the wall would be inert
            overhead). Pass an explicit value to override either way.
        z_wall_cutoff: See ``EmbeddingAnnealInput.z_wall_cutoff``.
        timestep: MD timestep (ps)
        seed: Random seed for velocity initialization
        dump_every: Dump-output frequency (steps)
        lammps_flags: Extra ``lmp`` command-line flags (e.g. Kokkos/GPU
            acceleration) passed verbatim to the anneal; depends on the
            LAMMPS build and force field, empty runs a plain ``lmp``
        extract_script: Remote payload; defaults to the shipped
            ``scripts/extract.py``
        anneal_script: Remote payload; defaults to the shipped
            ``scripts/embedding_anneal.py``
        anneal_cluster_config: ClusterConfig for the anneal stage; defaults
            to ``cluster_config``
        anneal_remote_fs: FileSystemHandlerBase for the anneal stage;
            defaults to ``remote_fs``
        remote_savepath: Path on the extraction machine for calculation files
        anneal_remote_savepath: Path on the anneal machine for calculation
            files; defaults to ``remote_savepath``
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, skip remote execution and produce mock outputs
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        trajectory_path: os.PathLike,
        threshold: float,
        cluster_cutoff: float,
        box_x: float,
        box_y: float,
        box_z_below: float,
        protect_radius: float,
        lammps_path: os.PathLike,
        pair_style: str,
        pair_coeff_path: os.PathLike,
        species: list[str],
        masses: list[float],
        anneal_stages: list[dict],
        box_z_above: Optional[float] = None,
        buffer: float = 2.0,
        topology_mode: str = "bulk",
        remove_species: Optional[list[str]] = None,
        phase_species: Optional[list[str]] = None,
        cleanup_mode: str = "none",
        solvent: Optional[Union[Atoms, str]] = None,
        stoichiometry: Optional[dict[str, int]] = None,
        specorder: Optional[list[str]] = None,
        smoothing_level: int = 8,
        mesh_radius: Optional[float] = None,
        normal_patch_radius: Optional[float] = None,
        anneal_cutoff: Optional[float] = None,
        anneal_template_path: Optional[os.PathLike] = None,
        roi_repel_margin: float = 0.5,
        roi_repel_epsilon: float = 10.0,
        roi_repel_wall_cutoff: float = 1.0,
        z_wall_margin: float = 1.0,
        z_wall_epsilon: Optional[float] = None,
        z_wall_cutoff: float = 1.0,
        timestep: float = 0.0005,
        seed: int = 42,
        dump_every: int = 1000,
        lammps_flags: str = "",
        extract_script: Optional[os.PathLike] = None,
        anneal_script: Optional[os.PathLike] = None,
        anneal_cluster_config: Optional[ClusterConfig] = None,
        anneal_remote_fs: Optional[FileSystemHandlerBase] = None,
        remote_savepath: Optional[os.PathLike] = None,
        anneal_remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__()
        if extract_script is None:
            extract_script = script_dir() / "extract.py"
        if anneal_script is None:
            anneal_script = script_dir() / "embedding_anneal.py"
        if anneal_template_path is None:
            anneal_template_path = (
                script_dir() / "templates" / "embedding_anneal.in.j2"
            )

        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.anneal_cluster_config = anneal_cluster_config or cluster_config
        self.anneal_remote_fs = anneal_remote_fs or remote_fs
        self.extract_script = extract_script
        self.anneal_script = anneal_script
        self.anneal_template_path = anneal_template_path

        self.trajectory_path = trajectory_path
        self.threshold = threshold
        self.cluster_cutoff = cluster_cutoff
        self.box_x = box_x
        self.box_y = box_y
        self.box_z_below = box_z_below
        self.box_z_above = box_z_above
        self.buffer = buffer
        self.protect_radius = protect_radius
        self.topology_mode = topology_mode
        self.remove_species = remove_species
        self.phase_species = phase_species
        self.cleanup_mode = cleanup_mode
        self.solvent = solvent
        self.stoichiometry = stoichiometry
        self.specorder = specorder
        self.smoothing_level = smoothing_level
        self.mesh_radius = mesh_radius
        self.normal_patch_radius = normal_patch_radius

        self.lammps_path = lammps_path
        self.pair_style = pair_style
        self.pair_coeff_path = pair_coeff_path
        self.species = species
        self.masses = masses
        self.anneal_stages = anneal_stages
        self.anneal_cutoff = (
            anneal_cutoff if anneal_cutoff is not None else protect_radius
        )
        self.roi_repel_margin = roi_repel_margin
        self.roi_repel_epsilon = roi_repel_epsilon
        self.roi_repel_wall_cutoff = roi_repel_wall_cutoff
        self.z_wall_margin = z_wall_margin
        self.z_wall_epsilon = (
            z_wall_epsilon
            if z_wall_epsilon is not None
            else (10.0 if topology_mode in ("surface", "interface") else 0.0)
        )
        self.z_wall_cutoff = z_wall_cutoff
        self.timestep = timestep
        self.seed = seed
        self.dump_every = dump_every
        self.lammps_flags = lammps_flags

        self.remote_savepath = remote_savepath
        self.anneal_remote_savepath = anneal_remote_savepath or remote_savepath
        self.loop_time = loop_time
        self.stub = stub

    def update(self, calcdir: Path, sampler_input=None, **kwargs) -> None:
        """
        Prepare the extraction task for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: Unused (this sampler reads directly from
                ``trajectory_path``, which does not change per iteration)
            **kwargs: Additional parameters
        """
        self.sample_dir = calcdir / "sample"
        self.extract_dir = self.sample_dir / "extract"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)

        if self.stub:
            return

        solvent_label = None
        solvent_file = None
        if self.cleanup_mode == "molecule":
            if isinstance(self.solvent, Atoms):
                solvent_file = self.extract_dir / "solvent_template.extxyz"
                ase.io.write(solvent_file, self.solvent, format="extxyz")
            elif isinstance(self.solvent, str):
                solvent_label = self.solvent
            else:
                raise WorkflowError(
                    "cleanup_mode='molecule' requires 'solvent' to be an "
                    "ase.Atoms object or an ase.build.molecule label "
                    f"string, got {type(self.solvent)}"
                )

        script_input = {
            "trajectory": Path(self.trajectory_path).name,
            "threshold": self.threshold,
            "cluster_cutoff": self.cluster_cutoff,
            "box_x": self.box_x,
            "box_y": self.box_y,
            "box_z_below": self.box_z_below,
            "box_z_above": self.box_z_above,
            "buffer": self.buffer,
            "protect_radius": self.protect_radius,
            "topology_mode": self.topology_mode,
            "remove_species": self.remove_species,
            "phase_species": self.phase_species,
            "cleanup_mode": self.cleanup_mode,
            "solvent_label": solvent_label,
            "solvent_file": (
                solvent_file.name if solvent_file is not None else None
            ),
            "stoichiometry": self.stoichiometry,
            # The anneal deck fixes LAMMPS type order to ``species`` (mass /
            # pair_coeff / Z_of_type lines), so every carved box must be
            # written with that same full ordering. Without an explicit
            # specorder ASE numbers types per-box from the symbols it happens
            # to contain, which breaks boxes missing a species. Default to
            # ``species`` unless the caller overrides.
            "specorder": (
                self.specorder if self.specorder is not None else self.species
            ),
            "smoothing_level": self.smoothing_level,
            "mesh_radius": self.mesh_radius,
            "normal_patch_radius": self.normal_patch_radius,
        }

        self.extract_input_file = self.extract_dir / "input_embedding.json"
        with open(self.extract_input_file, "w") as f:
            f.write(json.dumps(script_input, indent=2))

        self.extract_task = SingleScriptTask(
            local_savepath=self.extract_dir,
            input_file=self.extract_input_file,
            cluster_config=self.cluster_config,
            remote_fs=self.remote_fs,
            script_file=self.extract_script,
            remote_savepath=self.remote_savepath,
            jobfile_name="run_embed_extract.sh",
            jobname="embed_extract",
            logfile="embed_extract.log",
            n_processors=1,
            loop_time=self.loop_time,
            collect_only=False,
        )

        self.extract_task.allocate_data()
        self.extract_task.copy_file(self.trajectory_path)
        if solvent_file is not None:
            self.extract_task.copy_file(solvent_file)

    def _create_stub_outputs(self, num_boxes: int = 3) -> list[Atoms]:
        """
        Create mock embedded/annealed outputs for testing.

        Args:
            num_boxes: Number of mock boxes to generate

        Returns:
            List of Atoms objects mimicking annealed embedding boxes
        """
        logger.info(f"Creating {num_boxes} stub amorphous embedding boxes")

        mock_boxes = []
        for i in range(num_boxes):
            symbols = self.species * 4
            positions = np.random.uniform(0, 2 * self.box_x, (len(symbols), 3))
            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[
                    2 * self.box_x,
                    2 * self.box_y,
                    self.box_z_below + (self.box_z_above or self.box_z_below),
                ],
                pbc=True,
            )
            atoms.info["box_index"] = i
            atoms.info["topology_mode"] = self.topology_mode
            mock_boxes.append(atoms)
        return mock_boxes

    def get_sampled_configs(self) -> Path:
        """
        Run extraction, then anneal every extracted box.

        Returns:
            Path to extxyz file containing the annealed configurations
        """
        if self.stub:
            logger.info(
                "Running in stub mode - creating mock amorphous embedding "
                "boxes"
            )
            sampled_configs = self._create_stub_outputs()
            output_dir = self.sample_dir / "out"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "samples.extxyz"
            ase.io.write(output_path, sampled_configs)
            return output_path

        self.extract_task.submit()
        if not self.extract_task.finish_check():
            raise WorkflowError(
                "Amorphous matrix embedding extraction did not finish "
                "successfully (no is_finished.info marker). Check the job "
                f"logs in {self.extract_task.calcdir}."
            )
        self.extract_task.collect_results()

        extract_out = Path(self.extract_task.local_savepath) / "out"
        manifest = json.loads((extract_out / "manifest.json").read_text())

        if not manifest:
            logger.warning(
                "Amorphous matrix embedding extraction produced no regions "
                "of interest (no atom exceeded threshold=%s). No boxes will "
                "be annealed.",
                self.threshold,
            )

        anneal_dir = self.sample_dir / "anneal"
        os.makedirs(anneal_dir, exist_ok=True)

        anneal_tasks = []
        for entry in manifest:
            box_dir = anneal_dir / f"box_{entry['index']}"
            os.makedirs(box_dir, exist_ok=True)

            anneal_input = {
                "data_file": Path(entry["data_file"]).name,
                "lammps_path": Path(self.lammps_path).as_posix(),
                "template_file": Path(self.anneal_template_path).name,
                "cutoff": self.anneal_cutoff,
                "roi_x": entry["roi_position"][0],
                "roi_y": entry["roi_position"][1],
                "roi_z": entry["roi_position"][2],
                "pair_style": self.pair_style,
                "pair_coeff_path": Path(self.pair_coeff_path).name,
                "species": self.species,
                "masses": self.masses,
                "stages": self.anneal_stages,
                "roi_repel_margin": self.roi_repel_margin,
                "roi_repel_epsilon": self.roi_repel_epsilon,
                "roi_repel_wall_cutoff": self.roi_repel_wall_cutoff,
                "z_wall_margin": self.z_wall_margin,
                "z_wall_epsilon": self.z_wall_epsilon,
                "z_wall_cutoff": self.z_wall_cutoff,
                "timestep": self.timestep,
                "seed": self.seed,
                "dump_every": self.dump_every,
                "lammps_flags": self.lammps_flags,
            }
            anneal_input_file = box_dir / "input_anneal.json"
            with open(anneal_input_file, "w") as f:
                f.write(json.dumps(anneal_input, indent=2))

            task = SingleScriptTask(
                local_savepath=box_dir,
                input_file=anneal_input_file,
                cluster_config=self.anneal_cluster_config,
                remote_fs=self.anneal_remote_fs,
                script_file=self.anneal_script,
                remote_savepath=self.anneal_remote_savepath,
                jobfile_name="run_embed_anneal.sh",
                jobname=f"embed_anneal_{entry['index']}",
                logfile="embed_anneal.log",
                n_processors=1,
                loop_time=self.loop_time,
                collect_only=False,
            )
            task.allocate_data()
            task.copy_file(extract_out / entry["data_file"])
            task.copy_file(self.anneal_template_path)
            # copy_path (not copy_file): potentials aren't always a single
            # file -- e.g. a GRACE model is normally saved as a directory.
            task.copy_path(self.pair_coeff_path)
            anneal_tasks.append((entry["index"], task))

        annealed = []
        for index, task in anneal_tasks:
            task.submit()
            if not task.finish_check():
                raise WorkflowError(
                    f"Embedding anneal for box {index} did not finish "
                    "successfully (no is_finished.info marker). Check the "
                    f"job logs in {task.calcdir}."
                )
            task.collect_results()
            out_extxyz = Path(task.local_savepath) / "out" / "samples.extxyz"
            annealed.append(ase.io.read(out_extxyz))

        # Deliberately under sample_dir, not extract_dir: this is the
        # pipeline's final combined output, not an extraction artifact --
        # keeping it out of extract_dir avoids it looking like one.
        output_path = self.sample_dir / "out" / "samples.extxyz"
        os.makedirs(output_path.parent, exist_ok=True)
        ase.io.write(output_path, annealed)

        logger.info(
            f"Amorphous matrix embedding produced {len(annealed)} annealed "
            f"configurations, written to {output_path}"
        )
        return output_path
