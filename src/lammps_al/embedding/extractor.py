"""The region-of-interest extractor for amorphous-matrix embedding.

In the layered architecture, concrete actors belong to the *project*, not to
the orchestrator: ``remote_handler`` owns the ``Selector``/``Sampler``
interfaces and the generic remote-execution machinery (``SingleScriptTask``,
``ClusterConfig``, the SLURM templating), while the application picks and
wires the actual selection/sampling methods.

``ROIExtractor`` is a **native** implementation of the ``Selector`` interface
(a peer to ``Sampler``, not a sub-step of it -- see ``remote_handler``'s
``active_learning_base.Selector`` docstring). Per AL iteration it extracts
regions of interest (ROIs) from an uncertainty-tagged trajectory and
carves/orients/cleans an amorphous-matrix box around each
(``scripts/extract.py``, run via a ``SingleScriptTask``, needs the
``embedding`` extra remotely when ``topology_mode`` is
``"surface"``/``"interface"``). It does not run any dynamics itself --
that is :class:`lammps_al.embedding.annealer.EmbeddingAnnealer`'s job, fed
by this class's output.

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler).
The cluster env that runs ``scripts/extract.py`` needs ``lammps_al`` (for the
input schema and the geometry/cleanup/surface-mesh helpers it imports) plus
``ovito`` when ``topology_mode`` is ``"surface"``/``"interface"``. The host
side needs just ``remote_handler`` + ``ase`` + ``numpy``.

``stub=True`` dry-runs the loop (mock manifest, no cluster) for testing.
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
from remote_handler.workflows.active_learning_base import Selector

from lammps_al.embedding.scripts import script_dir

logger = logging.getLogger(__name__)


class ROIExtractor(Selector):
    """
    Extracts regions of interest (ROIs) from an uncertainty-tagged
    trajectory and carves/orients/cleans an amorphous-matrix box around
    each, ready to be annealed by
    :class:`lammps_al.embedding.annealer.EmbeddingAnnealer`.

    One remote stage, a single ``SingleScriptTask`` running
    ``scripts/extract.py`` (needs the ``embedding`` extra when
    ``topology_mode`` is ``"surface"``/``"interface"``).

    Args:
        remote_fs: FileSystemHandlerBase for the extraction stage
        cluster_config: ClusterConfig for the extraction stage
        threshold: Minimum per-atom uncertainty for an atom to seed a ROI
        cluster_cutoff: Distance (Angstrom) grouping high-uncertainty atoms
            into the same ROI
        box_x: Half-width of the carved box along x
        box_y: Half-width of the carved box along y
        box_z_below: Half-width of the carved box below the ROI along z
        protect_radius: Radius around each ROI that cleanup never modifies.
            Pick this together with the downstream
            ``EmbeddingAnnealer.anneal_cutoff`` -- they describe the same
            physical frozen-core radius from two sides of the pipeline.
        specorder: Species order for ``lammps-data`` output. The anneal deck
            fixes LAMMPS type order to ``EmbeddingAnnealer.species`` (mass /
            pair_coeff / Z_of_type lines), so every carved box must be
            written with that *same* full ordering -- pass the identical
            list you pass as ``EmbeddingAnnealer(species=...)``. Without an
            explicit ``specorder``, ASE numbers each box's types from
            whichever symbols it happens to contain, which breaks any box
            missing a species.
        box_z_above: Half-width above the ROI along z; defaults to
            ``box_z_below`` (symmetric box)
        buffer: Extra padding added to each carved-cell dimension
        topology_mode: ``"bulk"``, ``"surface"``, or ``"interface"`` --
            see ``MatrixEmbeddingInput``
        remove_species: Phase stripped (on an internal OVITO copy) before
            meshing for the interface normal. Required for ``"interface"``.
        phase_species: Species defining the condensed phase. Used by the
            normal-detection fallback heuristic, and (together with
            ``phase_cutoff``) to restrict ROI candidates to high-uncertainty
            atoms actually near this phase.
        phase_cutoff: Distance (Angstrom) for the ``phase_species``
            proximity filter (see ``geometry.find_rois``) -- keeps ROIs out
            of a bulk single-phase region with no condensed-phase core for
            ``EmbeddingAnnealer``'s frozen-sphere anneal to work with.
            Required together with ``phase_species`` for the filter to
            apply; ``None`` (default) applies no filter.
        cleanup_mode: ``"none"``, ``"molecule"``, or ``"stoichiometry"``
        solvent: For ``cleanup_mode="molecule"``: either an ASE ``Atoms``
            solvent template (staged and sent as a file) or an
            ``ase.build.molecule`` g2 label string (e.g. ``"H2O"``)
        stoichiometry: Target ratio dict for ``cleanup_mode="stoichiometry"``
        smoothing_level: OVITO ``ConstructSurfaceMeshModifier`` smoothing
        mesh_radius: OVITO probe-sphere radius; ``None`` uses OVITO's default
        normal_patch_radius: Approximate geodesic radius over which
            edge-connected surface-mesh facet normals are area-averaged
            after anchoring at the closest surface point. This smooths local
            atomic-scale roughness without mixing in a disconnected nearby
            surface. ``None`` (default) uses ``protect_radius``.
        max_num_structures: Caps the total number of carved boxes across the
            whole trajectory, keeping only the highest-uncertainty ROIs --
            see ``MatrixEmbeddingInput.max_num_structures``. ``None``
            (default) applies no cap.
        extract_script: Remote payload; defaults to the shipped
            ``scripts/extract.py``
        remote_savepath: Path on the extraction machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, skip remote execution and produce a mock manifest
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        threshold: float,
        cluster_cutoff: float,
        box_x: float,
        box_y: float,
        box_z_below: float,
        protect_radius: float,
        specorder: list[str],
        box_z_above: Optional[float] = None,
        buffer: float = 2.0,
        topology_mode: str = "bulk",
        remove_species: Optional[list[str]] = None,
        phase_species: Optional[list[str]] = None,
        phase_cutoff: Optional[float] = None,
        cleanup_mode: str = "none",
        solvent: Optional[Union[Atoms, str]] = None,
        stoichiometry: Optional[dict[str, int]] = None,
        smoothing_level: int = 8,
        mesh_radius: Optional[float] = None,
        normal_patch_radius: Optional[float] = None,
        max_num_structures: Optional[int] = None,
        extract_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__()
        if extract_script is None:
            extract_script = script_dir() / "extract.py"

        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.extract_script = extract_script

        self.threshold = threshold
        self.cluster_cutoff = cluster_cutoff
        self.box_x = box_x
        self.box_y = box_y
        self.box_z_below = box_z_below
        self.box_z_above = box_z_above
        self.buffer = buffer
        self.protect_radius = protect_radius
        self.specorder = specorder
        self.topology_mode = topology_mode
        self.remove_species = remove_species
        self.phase_species = phase_species
        self.phase_cutoff = phase_cutoff
        self.cleanup_mode = cleanup_mode
        self.solvent = solvent
        self.stoichiometry = stoichiometry
        self.smoothing_level = smoothing_level
        self.mesh_radius = mesh_radius
        self.normal_patch_radius = normal_patch_radius
        self.max_num_structures = max_num_structures

        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub

    def _resolve_trajectory(
        self, selector_input: Union[Path, list[Path]]
    ) -> Path:
        """Normalize the ring input to a single trajectory file.

        Usually a single upstream path; if the producer stage handed us
        several (e.g. ``combine=True``), concatenate them into one
        trajectory so ``scripts/extract.py`` still sees a single file.
        """
        paths = (
            [selector_input]
            if isinstance(selector_input, (str, Path))
            else list(selector_input)
        )
        if len(paths) == 1:
            return Path(paths[0])

        frames: list[Atoms] = []
        for p in paths:
            read = ase.io.read(p, index=":")
            frames.extend(read if isinstance(read, list) else [read])
        combined_path = self.extract_dir / "trajectory.extxyz"
        ase.io.write(combined_path, frames, format="extxyz")
        return combined_path

    def update(
        self, calcdir: Path, selector_input: Path | list[Path], **kwargs
    ) -> None:
        """
        Prepare the extraction task for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            selector_input: Path (or paths) to the uncertainty-tagged
                trajectory to extract ROIs from -- typically the previous
                stage's (e.g. a ``Sampler``'s) output.
            **kwargs: Additional parameters
        """
        self.extract_dir = Path(calcdir) / "extract"
        os.makedirs(self.extract_dir, exist_ok=True)

        if self.stub:
            return

        trajectory_path = self._resolve_trajectory(selector_input)

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
            "trajectory": trajectory_path.name,
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
            "phase_cutoff": self.phase_cutoff,
            "cleanup_mode": self.cleanup_mode,
            "solvent_label": solvent_label,
            "stoichiometry": self.stoichiometry,
            "specorder": self.specorder,
            "smoothing_level": self.smoothing_level,
            "mesh_radius": self.mesh_radius,
            "normal_patch_radius": self.normal_patch_radius,
            "max_num_structures": self.max_num_structures,
        }
        # Omit rather than write null: remote_handler's ScriptInput.from_dict
        # Path()-converts every _path_fields key present in the JSON
        # unconditionally, even when its value is null, raising a TypeError.
        # solvent_file is declared Optional specifically for callers that use
        # solvent_label instead, so it must stay absent from the JSON when
        # unset, not present-as-null.
        if solvent_file is not None:
            script_input["solvent_file"] = solvent_file.name

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
        self.extract_task.copy_file(trajectory_path)
        if solvent_file is not None:
            self.extract_task.copy_file(solvent_file)

    def _create_stub_manifest(self, num_boxes: int = 3) -> Path:
        """Write a mock manifest + placeholder box files for testing."""
        logger.info(
            "Creating a stub manifest with %d regions of interest", num_boxes
        )
        out_dir = self.extract_dir / "out"
        os.makedirs(out_dir, exist_ok=True)

        manifest = []
        for i in range(num_boxes):
            symbols = self.specorder * 4
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
            data_name = f"embedded_{i}.data"
            ase.io.write(
                out_dir / data_name,
                atoms,
                format="lammps-data",
                specorder=self.specorder,
            )
            manifest.append(
                {
                    "index": i,
                    "frame": 0,
                    "roi_index": 0,
                    "roi_position": [self.box_x, self.box_y, self.box_z_below],
                    # Mocked, spread out (unlike "roi_position" above, which
                    # is deliberately constant -- see extract.py's own
                    # comment on this field): stub-mode consumers exercising
                    # a spatial spread over ROIs (e.g. the dashboard's
                    # roi_spatial panel) shouldn't see every stub ROI at an
                    # identical point.
                    "roi_position_orig": (
                        np.random.uniform(0, 2 * self.box_x, 3)
                    ).tolist(),
                    "normal": None,
                    "data_file": data_name,
                    "extxyz_file": None,
                }
            )

        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return out_dir

    def get_selected_configs(self) -> Path:
        """
        Run extraction and return the directory holding its output.

        Returns:
            Path to the directory containing ``manifest.json`` and every
            carved-box data/extxyz file -- the input
            :class:`lammps_al.embedding.annealer.EmbeddingAnnealer` expects.
        """
        if self.stub:
            return self._create_stub_manifest()

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
                "of interest (no atom exceeded threshold=%s). Nothing will "
                "be annealed downstream.",
                self.threshold,
            )
        return extract_out
