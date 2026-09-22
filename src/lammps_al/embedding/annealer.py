"""The frozen-core anneal Sampler for amorphous-matrix embedding.

In the layered architecture, concrete actors belong to the *project*, not to
the orchestrator: ``remote_handler`` owns the ``Sampler`` interface and the
generic remote-execution machinery (``SingleScriptTask``, ``ClusterConfig``,
the SLURM templating), while the application picks and wires the actual
sampling method.

``EmbeddingAnnealer`` is a **native** implementation of that interface. Per
AL iteration it anneals every carved box handed to it by
:class:`lammps_al.embedding.extractor.ROIExtractor` with a multi-stage
LAMMPS deck around a frozen ROI core (``scripts/embedding_anneal.py``, one
``SingleScriptTask`` per box, needs a LAMMPS build). It runs genuine new MD
and is exactly where new atomic configurations get generated in this
pipeline -- but it does not decide *where* to anneal; that ROI-finding step
belongs to ``ROIExtractor``, whose output this class's ``sampler_input``
must be.

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler).
The cluster env that runs ``scripts/embedding_anneal.py`` needs
``lammps_al`` and a LAMMPS build. The host side needs just
``remote_handler`` + ``ase`` + ``numpy``.

``stub=True`` dry-runs the loop (mock annealed boxes, no cluster) for
testing -- it still reads the real manifest handed to it by
``ROIExtractor``, so the number of stub boxes tracks whatever was
extracted.
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


class EmbeddingAnnealer(Sampler):
    """
    Anneals every carved box produced by
    :class:`lammps_al.embedding.extractor.ROIExtractor` with a multi-stage
    LAMMPS deck around a frozen ROI core.

    One ``SingleScriptTask`` per extracted box, running
    ``scripts/embedding_anneal.py``.

    Args:
        remote_fs: FileSystemHandlerBase for the anneal stage
        cluster_config: ClusterConfig for the anneal stage
        lammps_path: Path to the LAMMPS executable
        pair_style: LAMMPS ``pair_style`` line contents
        pair_coeff_path: Path to the potential file or directory (e.g. a
            GRACE model, normally saved as a directory), or ``None`` to
            require it be supplied per-iteration via ``update()``'s
            ``pair_coeff_path`` kwarg instead (wired from a ``Stage``'s
            ``extra_inputs``, the same mechanism
            ``CommitteeUncertaintySelector`` uses for its committee -- see
            that class's module docstring). A per-call value always
            overrides this constructor default, not just fills a gap.
        species: Element symbols in LAMMPS type order. Must match the
            ``specorder`` given to the upstream ``ROIExtractor`` exactly --
            the anneal deck fixes LAMMPS type order to this list (mass /
            pair_coeff / Z_of_type lines), so every carved box needs the
            same ordering it was written with.
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
        anneal_cutoff: Radius of the anneal's frozen ``fixSphere`` --
            typically the same value passed as the upstream
            ``ROIExtractor.protect_radius``, since both describe the same
            physical frozen-core radius.
        topology_mode: ``"bulk"``, ``"surface"``, or ``"interface"`` -- used
            here only to auto-decide ``z_wall_epsilon`` when it is left
            unset (see below). Should match the upstream ``ROIExtractor``'s
            ``topology_mode``.
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
        anneal_script: Remote payload; defaults to the shipped
            ``scripts/embedding_anneal.py``
        remote_savepath: Path on the anneal machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, skip remote execution and produce mock outputs
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        lammps_path: os.PathLike,
        pair_style: str,
        species: list[str],
        masses: list[float],
        anneal_stages: list[dict],
        anneal_cutoff: float,
        pair_coeff_path: Optional[os.PathLike] = None,
        topology_mode: str = "bulk",
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
        anneal_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__()
        if anneal_script is None:
            anneal_script = script_dir() / "embedding_anneal.py"
        if anneal_template_path is None:
            anneal_template_path = (
                script_dir() / "templates" / "embedding_anneal.in.j2"
            )

        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.anneal_script = anneal_script
        self.anneal_template_path = anneal_template_path

        self.lammps_path = lammps_path
        self.pair_style = pair_style
        self.pair_coeff_path = pair_coeff_path
        self.species = species
        self.masses = masses
        self.anneal_stages = anneal_stages
        self.anneal_cutoff = anneal_cutoff
        self.topology_mode = topology_mode
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
        self.loop_time = loop_time
        self.stub = stub

    def update(
        self,
        calcdir: Path,
        sampler_input: Path | list[Path],
        pair_coeff_path: Optional[Path | list[Path]] = None,
        **kwargs,
    ) -> None:
        """
        Stage one anneal task per box in the extractor's manifest.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: Directory produced by
                ``ROIExtractor.get_selected_configs()`` -- must contain
                ``manifest.json`` plus every carved-box data file it lists.
            pair_coeff_path: The potential to anneal with, resolved via a
                ``Stage``'s ``extra_inputs`` (same mechanism
                ``CommitteeUncertaintySelector`` uses for its committee) --
                not this stage's primary ring input. Overrides the
                constructor default for this iteration; a list (e.g. every
                committee member from the "trainer" stage) picks its first
                entry, since annealing drives a single ``pair_style``/
                ``pair_coeff``, not a committee.
            **kwargs: Additional parameters
        """
        if pair_coeff_path is not None:
            self.pair_coeff_path = (
                pair_coeff_path[0]
                if isinstance(pair_coeff_path, list)
                else pair_coeff_path
            )
        if self.pair_coeff_path is None:
            raise WorkflowError(
                "EmbeddingAnnealer.update() needs a 'pair_coeff_path' -- "
                "either at construction time or wired via the Stage's "
                "extra_inputs."
            )

        extract_out = Path(
            sampler_input
            if isinstance(sampler_input, (str, Path))
            else sampler_input[0]
        )
        manifest = json.loads((extract_out / "manifest.json").read_text())

        self.anneal_dir = Path(calcdir) / "anneal"
        os.makedirs(self.anneal_dir, exist_ok=True)
        self._num_boxes = len(manifest)

        if self.stub:
            return

        self.anneal_tasks: list[tuple[int, SingleScriptTask]] = []
        for entry in manifest:
            box_dir = self.anneal_dir / f"box_{entry['index']}"
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
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                script_file=self.anneal_script,
                remote_savepath=self.remote_savepath,
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
            self.anneal_tasks.append((entry["index"], task))

    def _create_stub_outputs(self) -> list[Atoms]:
        """Mock annealed outputs, one per box the extractor reported."""
        logger.info(
            "Creating %d stub annealed embedding boxes", self._num_boxes
        )
        mock_boxes = []
        for i in range(self._num_boxes):
            symbols = self.species * 4
            positions = np.random.uniform(
                0, 2 * self.anneal_cutoff, (len(symbols), 3)
            )
            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[2 * self.anneal_cutoff] * 3,
                pbc=True,
            )
            atoms.info["box_index"] = i
            atoms.info["topology_mode"] = self.topology_mode
            mock_boxes.append(atoms)
        return mock_boxes

    def get_sampled_configs(self) -> Path:
        """
        Run every box's anneal task and combine the results.

        Returns:
            Path to extxyz file containing the annealed configurations
        """
        out_dir = self.anneal_dir / "out"
        os.makedirs(out_dir, exist_ok=True)
        output_path = out_dir / "samples.extxyz"

        if self.stub:
            ase.io.write(output_path, self._create_stub_outputs())
            return output_path

        annealed = []
        for index, task in self.anneal_tasks:
            task.submit()
            if not task.finish_check():
                raise WorkflowError(
                    f"Embedding anneal for box {index} did not finish "
                    "successfully (no is_finished.info marker). Check the "
                    f"job logs in {task.calcdir}."
                )
            task.collect_results()
            box_out = Path(task.local_savepath) / "out" / "samples.extxyz"
            annealed.append(ase.io.read(box_out))

        ase.io.write(output_path, annealed)
        logger.info(
            "Embedding anneal produced %d annealed configurations, "
            "written to %s",
            len(annealed),
            output_path,
        )
        return output_path
