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
from remote_handler.core.remote_env import CALCDIRS_DIRNAME
from remote_handler.core.tasks import (
    ArrayScriptTask,
    DetachableTaskMixin,
    DictJSON,
    SingleScriptTask,
    Writeable,
)
from remote_handler.exceptions import WorkflowError
from remote_handler.workflows.active_learning_base import Sampler

from lammps_al.embedding.scripts import script_dir

logger = logging.getLogger(__name__)


class _CopyFileResource(Writeable):
    """An ``ArrayScriptTask`` per-box resource that uploads ``content`` (a
    local file path) verbatim.

    Deliberately *not* ``remote_handler``'s own ``AtomsBatch`` ``Writeable``:
    ``AtomsBatch.write`` funnels every box through ``FileSystemHandlerBase.
    write_ase``, which always writes its local staging copy to a hardcoded
    ``temp_ase_file.extxyz`` (extxyz format) regardless of the *target*
    filename, then uploads those bytes verbatim -- confirmed directly against
    the installed ``remote_handler``. Handed a box's already-correctly-written
    ``.data`` (LAMMPS-data format) file, that would silently replace its
    content with mislabeled extxyz text. This class instead
    copies the file byte-for-byte via ``remote_fs.put``, the same primitive
    ``FileAllocator.copy_file`` uses -- no re-serialization, so no format
    mismatch is possible.
    """

    def write(self, calcpath):
        self.remote_fs.put(self.content, Path(calcpath) / self.filename)


class EmbeddingAnnealer(DetachableTaskMixin, Sampler):
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
        n_processors: MPI ranks per box (``mpirun -np N``), and the number
            of cores requested from the scheduler for each box. Only
            effective if ``lammps_path`` points at an MPI-enabled build --
            a non-MPI ``lmp`` silently runs single-rank regardless of this
            value. Each box here is typically a few hundred atoms, well
            below where LAMMPS' spatial decomposition scales well, so a
            small value (2-4) is more likely to help than a large one; this
            also trades directly against ``array_job``'s ``num_parallel``
            for the same core budget (N boxes * n_processors cores each).
        anneal_script: Remote payload; defaults to the shipped
            ``scripts/embedding_anneal.py``
        remote_savepath: Path on the anneal machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        array_job: Submit every box as one SLURM *array* job
            (``remote_handler.core.tasks.ArrayScriptTask``) instead of one
            independent ``SingleScriptTask`` per box. The extraction side
            can legitimately produce hundreds of ROIs in one iteration (a
            spatially fragmented interface, or an uncalibrated ``threshold``,
            can each drive the candidate count up sharply -- see
            ``ROIExtractor.max_num_structures`` for the complementary fix on
            that side), and queuing that many independent jobs both floods
            the scheduler and, worse, was previously fully serial here (each
            ``SingleScriptTask`` submitted and *waited on* in turn before the
            next was even queued) -- an array job submits once and lets
            ``num_parallel`` boxes actually run concurrently. Unlike the
            single-task path (which raises on the first box's failure),
            array mode tolerates individual box failures: it logs and skips
            whichever boxes never produced ``out/samples.extxyz``, raising
            only if *none* of them did -- at this scale, requiring every
            single LAMMPS run to succeed would make one crashed box discard
            every other box's already-finished work.
        num_parallel: Array jobs run concurrently at once (only relevant
            when ``array_job=True``).
        stub: If True, skip remote execution and produce mock outputs
        collect_only: If True (``array_job=True`` only), ``update``
            reconstructs the array job an earlier, ``collect_only=False``
            call already submitted (from its manifest) instead of building
            and resubmitting a new one -- see ``DetachableTaskMixin``.
            Normally left False and driven instead by
            ``StagedActiveLearningHandler``'s own automatic reattach: it
            detects a pending manifest via ``pending_manifest()`` and flips
            this on for one call on its own, so a driver restart resumes an
            already-submitted array job rather than resubmitting from
            scratch. Not supported for the per-box ``array_job=False`` path
            (one task per box, no single manifest to reattach to).
        reattach_max_wait: Only used when ``collect_only``: how long (in
            seconds) to wait for the reattached array job to finish before
            giving up. ``None`` (default) waits forever.
    """

    _iteration_subdir = "anneal"  # calcdir/anneal -- see DetachableTaskMixin
    # Matches this actor's pre-existing on-disk layout for the array job
    # (calcdir/anneal/array, not the mixin's own "results" default) --
    # keeps a manifest already written under the old layout still findable.
    _results_subdir = "array"

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
        n_processors: int = 1,
        anneal_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        array_job: bool = False,
        num_parallel: int = 10,
        stub: bool = False,
        collect_only: bool = False,
        reattach_max_wait: Optional[float] = None,
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
        self.n_processors = n_processors

        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.array_job = array_job
        self.num_parallel = num_parallel
        self.stub = stub
        self.collect_only = collect_only
        self.reattach_max_wait = reattach_max_wait

    def pending_manifest(self, calcdir: os.PathLike) -> Optional[Path]:
        """Reattach is only supported for the array-job path.

        The per-box ``array_job=False`` path builds one independent
        ``SingleScriptTask`` per box rather than a single task/manifest, so
        there is nothing here for ``DetachableTaskMixin``'s default
        (manifest-path-based) implementation to reattach to -- return
        ``None`` (reattach unsupported) in that case, same as the ABC's own
        default.
        """
        if not self.array_job:
            return None
        return super().pending_manifest(calcdir)

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

        self.anneal_dir = self.iteration_dir(calcdir)
        os.makedirs(self.anneal_dir, exist_ok=True)
        self._num_boxes = len(manifest)

        if self.stub:
            return

        if self.array_job:
            self._build_array_task(calcdir, manifest, extract_out)
        else:
            self._build_single_tasks(manifest, extract_out)

    def _anneal_input(self, entry: dict, *, shared_prefix: str = "") -> dict:
        """The per-box anneal-payload JSON, shared by both submission modes.

        ``shared_prefix`` is ``""`` for a ``SingleScriptTask`` (the template/
        potential are copied into this box's own calcdir, so a bare filename
        resolves directly) or ``"../../"`` for an array job: ``ArrayScriptTask``
        places every per-box calcdir at ``rootpath/calcdirs/<i>`` (see
        ``remote_handler.core.remote_env.CALCDIRS_DIRNAME``), while shared
        resources land in ``rootpath`` itself via ``copy_path`` -- two levels
        up from a per-box calcdir, not one (verified directly against
        ``ArrayScriptTask``'s real remote layout, not just inferred from its
        docstring). See ``embedding_anneal.py``'s ``_template_location`` for
        how the remote payload resolves this back (it strips however many
        leading ``"../"`` segments are given, rather than assuming a fixed
        depth, so this stays correct even if that nesting ever changes).
        """
        return {
            "data_file": Path(entry["data_file"]).name,
            "lammps_path": Path(self.lammps_path).as_posix(),
            "template_file": shared_prefix
            + Path(self.anneal_template_path).name,
            "cutoff": self.anneal_cutoff,
            "roi_x": entry["roi_position"][0],
            "roi_y": entry["roi_position"][1],
            "roi_z": entry["roi_position"][2],
            "pair_style": self.pair_style,
            "pair_coeff_path": shared_prefix + Path(self.pair_coeff_path).name,
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
            "n_processors": self.n_processors,
        }

    def _build_single_tasks(
        self, manifest: list[dict], extract_out: Path
    ) -> None:
        """One independent ``SingleScriptTask`` per box (the default)."""
        self.anneal_tasks: list[tuple[int, SingleScriptTask]] = []
        for entry in manifest:
            box_dir = self.anneal_dir / f"box_{entry['index']}"
            os.makedirs(box_dir, exist_ok=True)

            anneal_input = self._anneal_input(entry)
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
                n_processors=self.n_processors,
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

    def _build_array_task(
        self, calcdir: Path, manifest: list[dict], extract_out: Path
    ) -> None:
        """One SLURM array job covering every box (``array_job=True``).

        The deck template and potential are genuinely identical across every
        box, so they're uploaded once as shared resources (``copy_path``,
        landing in the array's root directory) rather than duplicated into
        every one of potentially hundreds of per-box calcdirs. Each box still
        gets its own config JSON + structure file via ``individual_ressources``
        (``ArrayScriptTask`` writes ``individual_ressources[k][i]`` into
        ``calcdir/i`` for every resource kind ``k`` and box ``i`` -- the two
        lists built below must stay index-aligned with ``manifest``, which
        ``self._array_manifest`` preserves for :meth:`get_sampled_configs` to
        map results back by the same position).

        Goes through ``DetachableTaskMixin.build_or_resume_array_task``
        rather than constructing+allocating ``ArrayScriptTask`` directly, so
        a ``collect_only`` call (driven by ``StagedActiveLearningHandler``'s
        automatic reattach, via :meth:`pending_manifest`) reconstructs the
        already-submitted job from its manifest instead of resubmitting --
        this is what makes a driver restart pick up an anneal array job
        that's already running/finished rather than launching a duplicate.
        The shared-resource upload (template + potential) only makes sense
        for a freshly built job, so it's skipped entirely when reattaching.
        """
        self._array_manifest = manifest
        self.results_dir = self.results_dir_for(calcdir)

        def _build_task() -> ArrayScriptTask:
            configs: list[DictJSON] = []
            structures: list[_CopyFileResource] = []
            for entry in manifest:
                anneal_input = self._anneal_input(
                    entry, shared_prefix="../../"
                )
                configs.append(
                    DictJSON(
                        content=anneal_input,
                        filename="input_anneal.json",
                        remote_fs=self.remote_fs,
                    )
                )
                structures.append(
                    _CopyFileResource(
                        content=extract_out / entry["data_file"],
                        filename=Path(entry["data_file"]).name,
                        remote_fs=self.remote_fs,
                    )
                )
            return ArrayScriptTask(
                local_savepath=self.results_dir,
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                python_script_file=self.anneal_script,
                individual_ressources=[configs, structures],
                remote_savepath=self.remote_savepath,
                jobfile_name="run_embed_anneal_array.sh",
                jobname="embed_anneal",
                logfile="embed_anneal.log",
                n_processors=self.n_processors,
                loop_time=self.loop_time,
                num_parallel=self.num_parallel,
                collect_only=False,
            )

        self.array_task = self.build_or_resume_array_task(
            self.results_dir,
            _build_task,
            python_script_file=self.anneal_script,
        )
        if not self.collect_only:
            # copy_path (not copy_file): potentials aren't always a single
            # file -- e.g. a GRACE model is normally saved as a directory.
            self.array_task.copy_path(self.anneal_template_path)
            self.array_task.copy_path(self.pair_coeff_path)

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

        if self.array_job:
            return self._get_sampled_configs_array(output_path)

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

    def _get_sampled_configs_array(self, output_path: Path) -> Path:
        """``get_sampled_configs``'s array-job path.

        One blocking call submits and waits for the *whole* array (up to
        ``num_parallel`` boxes running at once), unlike the single-task path
        above, which waits on each box in strict sequence before the next is
        even queued. Tolerant of partial failure: a box missing
        ``out/samples.extxyz`` (a crashed/non-converged anneal -- not
        unrealistic across a few hundred independent MD runs) is logged and
        skipped rather than discarding every other box's already-finished
        work; this only raises if *none* of them produced output.

        When ``collect_only`` (a reattached array -- see
        :meth:`_build_array_task`), there is no fresh ``submit()`` call to
        block on, so this polls ``wait_if_needed`` instead (bounded by
        ``reattach_max_wait``) until every box's own ``is_finished.info``
        marker appears remotely -- the same strict, whole-array wait
        ``remote_handler``'s other array-job actors (e.g. ``ModelTrainer``)
        use for reattach, even though, like here, their own downstream
        collection step already tolerates individual box failures.
        """
        if self.collect_only:
            self.wait_if_needed(
                self.array_task,
                max_wait=self.reattach_max_wait,
                error_message=(
                    "Reattached embedding anneal (array job) did not "
                    "finish."
                ),
            )
        else:
            self.array_task.submit()
        self.array_task.collect_results()

        annealed = []
        failed = []
        for i, entry in enumerate(self._array_manifest):
            box_out = (
                Path(self.array_task.local_savepath)
                / CALCDIRS_DIRNAME
                / str(i)
                / "out"
                / "samples.extxyz"
            )
            if not box_out.exists():
                failed.append(entry["index"])
                continue
            annealed.append(ase.io.read(box_out))

        if failed:
            logger.warning(
                "Embedding anneal (array job): %d/%d box(es) produced no "
                "output (no out/samples.extxyz) and were skipped -- box "
                "indices: %s. Check job logs under %s/%s/<array index>/.",
                len(failed),
                len(self._array_manifest),
                failed,
                self.array_task.local_savepath,
                CALCDIRS_DIRNAME,
            )
        if not annealed:
            raise WorkflowError(
                "Embedding anneal (array job): none of the "
                f"{len(self._array_manifest)} box(es) produced output "
                f"(no out/samples.extxyz). Check job logs under "
                f"{self.array_task.local_savepath}/{CALCDIRS_DIRNAME}"
                "/<array index>/."
            )

        ase.io.write(output_path, annealed)
        logger.info(
            "Embedding anneal (array job) produced %d/%d annealed "
            "configurations, written to %s",
            len(annealed),
            len(self._array_manifest),
            output_path,
        )
        return output_path
