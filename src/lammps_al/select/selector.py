"""The committee-uncertainty annotator, and an optional peak-frame
selector, for this project.

In the layered architecture, concrete selectors belong to the *project*, not
to the orchestrator: ``remote_handler`` owns the ``Selector`` interface (and,
for host-side strategies, the ``ConfigSelector``/``SelectorActor`` machinery)
while the application picks and wires the actual selection method.

``CommitteeUncertaintySelector`` is a **native** implementation of the
``Selector`` interface: it scores every frame of an MD trajectory (typically
:class:`lammps_al.run.sampler.LammpsSampler`'s output) against a training
committee and returns the *whole* trajectory annotated with a per-atom
``uncertainty`` array -- it does not select/reduce anything. Despite sitting
in the "selector" slot of a ``Stage`` ring (a peer to ``Sampler``, not a
sub-step of it -- see ``remote_handler``'s ``active_learning_base.Selector``
docstring), its job here is purely to *annotate*, mirroring this repo's
``ROIExtractor``/``EmbeddingAnnealer`` split in spirit (dynamics vs.
selection are different loop roles) while further separating "compute the
signal" from "act on the signal": committee-uncertainty scoring always
needs a cluster job (the committee), but what happens with the resulting
per-atom uncertainties differs by campaign -- ``ROIExtractor`` searches the
whole annotated trajectory spatially for regions of interest;
``mace_md_AL``-style campaigns instead want a handful of whole peak-
uncertainty configs, via :class:`PeakUncertaintySelector` below.

Per AL iteration, ``CommitteeUncertaintySelector``:

    1. takes the sampler's trajectory as its primary (ring) input, and the
       full committee's model files as a secondary input (via a ``Stage``'s
       ``extra_inputs`` -- the committee is not the sampler's output, it's
       whatever already feeds the sampler stage's own primary ring input;
       see ``lammps_al.select.scripts.committee_selection``'s docstring and
       the call sites for the exact ``InputSource`` wiring);
    2. scores every frame's local force-uncertainty against the committee,
       generic over the committee's potential family via ``backend``
       (``"mace"`` default, ``"ace"``, or ``"grace"``) via
       ``_build_committee_calculator`` in the payload;
    3. returns the whole trajectory, annotated.

``PeakUncertaintySelector`` is a pure host-side ``ConfigSelector`` strategy
(no cluster job, no committee -- it only needs the ``uncertainty`` arrays
``CommitteeUncertaintySelector`` already wrote): it splits an already-
annotated trajectory into ``bin_count`` contiguous segments and picks the
peak-uncertainty frame from each. Wrap it in ``remote_handler``'s
``SelectorActor`` to use as a ``Stage`` (the same adapter STO's
``RandomSelector`` already uses).

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler + a
committee stack). The cluster conda env that runs the payload needs
``lammps_al`` (for the input schema) plus whatever ``backend`` needs (MACE:
``mace-torch``+``torch``; ACE: ``pyace``; GRACE: ``tensorpotential``); the
host side needs just ``remote_handler`` + ``ase``.

``stub=True`` dry-runs the loop (mock annotated trajectory, no cluster) for
testing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
from ase import Atoms
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.exceptions import WorkflowError
from remote_handler.workflows.active_learning_base import Selector
from remote_handler.workflows.selectors import ConfigSelector

from lammps_al.select.scripts import script_dir
from lammps_al.select.scripts.committee_selection import (
    select_max_uncertainty_frames,
)

logger = logging.getLogger(__name__)


class CommitteeUncertaintySelector(Selector):
    """Annotates every frame of an MD trajectory with committee uncertainty.

    Runs the committee-uncertainty annotation payload on a cluster, given a
    trajectory (a ``Sampler``'s output) and the committee's model files (a
    secondary input -- see the module docstring). Does not reduce the
    trajectory -- every frame that went in comes back out, annotated.
    Supports stub mode for testing without actual cluster execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where scoring runs
        cutoff_radius: Cutoff radius (Angstrom) for local uncertainty
            aggregation
        device: compute device for the committee calculator, where the
            backend's calculator takes one ("cuda" or "cpu"; MACE only).
        backend: Which committee-potential family scores the trajectory:
            ``"mace"`` (default), ``"ace"`` (pacemaker), or ``"grace"``
            (gracemaker) -- see ``committee_selection.py``'s module
            docstring for the calculator dispatch.
        selection_script: Remote payload; defaults to the shipped
            ``committee_selection.py``.
        remote_savepath: Path on the remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: Dry-run without a cluster; emit a mock annotated trajectory
        cluster_config: ClusterConfig for the scoring stage
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cutoff_radius: float,
        device: str = "cuda",
        backend: str = "mace",
        selection_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        cluster_config: Optional[ClusterConfig] = None,
    ):
        super().__init__()
        if selection_script is None:
            selection_script = script_dir() / "committee_selection.py"
        self.remote_fs = remote_fs
        self.selection_script = selection_script
        self.cutoff_radius = cutoff_radius
        self.device = device
        self.backend = backend
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.cluster_config = cluster_config
        self.stub = stub

    def update(
        self,
        calcdir: Path,
        selector_input: Path | list[Path],
        model_files: Optional[Path | list[Path]] = None,
        **kwargs,
    ) -> None:
        """Prepare (and stage) the committee-scoring job for this iteration.

        Args:
            calcdir: Directory for this AL iteration
            selector_input: Trajectory to score (typically a ``Sampler``'s
                output)
            model_files: The committee's model file paths, resolved via a
                ``Stage``'s ``extra_inputs`` (see the module docstring) --
                not this stage's primary ring input.
            **kwargs: Unused; accepted for interface consistency
        """
        self.select_dir = calcdir / "committee_select"
        self.results_dir = self.select_dir / "results"
        os.makedirs(self.select_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        if isinstance(selector_input, list):
            self.trajectory_path = Path(selector_input[0])
        else:
            self.trajectory_path = Path(selector_input)

        if model_files is None:
            raise WorkflowError(
                "CommitteeUncertaintySelector.update() needs 'model_files' "
                "(the committee) -- wire it via the Stage's extra_inputs."
            )
        self.model_files = (
            model_files if isinstance(model_files, list) else [model_files]
        )

        if self.stub:
            return

        script_input = {
            "trajectory": self.trajectory_path.name,
            "model_files": [Path(m).name for m in self.model_files],
            "cutoff_radius": self.cutoff_radius,
            "device": self.device,
            "backend": self.backend,
        }

        self.script_input_file = self.results_dir / "input_select.json"
        with open(self.script_input_file, "w") as f:
            f.write(json.dumps(script_input, indent=2))

        self.selection_task = SingleScriptTask(
            local_savepath=self.results_dir,
            input_file=self.script_input_file,
            cluster_config=self.cluster_config,
            remote_fs=self.remote_fs,
            script_file=self.selection_script,
            remote_savepath=self.remote_savepath,
            jobfile_name="run_select.sh",
            jobname="committee_select",
            logfile="select.log",
            n_processors=1,
            loop_time=self.loop_time,
            collect_only=False,
        )

        self.selection_task.allocate_data()

        self.selection_task.copy_file(self.trajectory_path)
        for p in self.model_files:
            self.selection_task.copy_file(p)

    def _create_stub_outputs(self, num_frames: int = 10) -> list[Atoms]:
        """Create a mock annotated trajectory for testing.

        Args:
            num_frames: Number of trajectory frames to generate

        Returns:
            List of Atoms objects, each carrying a mock per-atom
            ``uncertainty`` array (what real annotation produces).
        """
        logger.info(f"Creating {num_frames} stub annotated frames")

        mock_frames = []
        for i in range(num_frames):
            symbols = ["Cu"] * 20 + ["Co"] * 4 + ["H"] * 2
            positions = np.random.uniform(0, 10, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[10, 10, 15],
                pbc=True,
            )
            atoms.arrays["uncertainty"] = np.random.uniform(
                0.05, 1.0, size=len(symbols)
            )
            atoms.info["step"] = i * 100

            mock_frames.append(atoms)

        return mock_frames

    def get_selected_configs(self) -> Path:
        """Run committee-uncertainty annotation and collect the result.

        Returns:
            Path to the extxyz file containing the whole trajectory,
            annotated (not reduced).
        """
        if self.stub:
            logger.info(
                "Running in stub mode - creating a mock annotated trajectory"
            )
            annotated = self._create_stub_outputs()
            output_path = (
                self.results_dir / "out" / "trajectory_uncertainties.extxyz"
            )
            os.makedirs(output_path.parent, exist_ok=True)
            ase.io.write(output_path, annotated)
            logger.info(f"Annotated trajectory written to {output_path}")
            return output_path

        self.selection_task.submit()
        if not self.selection_task.finish_check():
            raise WorkflowError(
                "Committee uncertainty annotation did not finish "
                "successfully (no is_finished.info marker). Check the job "
                f"logs in {self.selection_task.calcdir}."
            )
        self.selection_task.collect_results()

        output_path = (
            Path(self.selection_task.local_savepath)
            / "out/trajectory_uncertainties.extxyz"
        )
        logger.info(f"Annotated trajectory written to {output_path}")
        return output_path


class PeakUncertaintySelector(ConfigSelector):
    """Picks the peak-uncertainty frame from each of ``bin_count`` segments
    of an already committee-uncertainty-annotated trajectory.

    Pure host-side selection -- no cluster job, no committee model: every
    input frame must already carry a per-atom ``uncertainty`` array (written
    by :class:`CommitteeUncertaintySelector`'s remote payload). Wrap in
    ``remote_handler.workflows.SelectorActor`` to use as a ``Stage``, e.g.
    for a campaign (like ``examples/mace_md_AL``) that hands whole configs
    straight to an evaluator with no further spatial selection step (unlike
    ``rough_cu_water_al``, whose ``ROIExtractor`` does its own spatial
    selection over the full annotated trajectory and needs no peak
    reduction first).

    Args:
        bin_count: Number of trajectory segments to pick a peak-uncertainty
            frame from.
    """

    def __init__(self, bin_count: int):
        self.bin_count = bin_count

    def select(self, configs, reference=None) -> list:
        flat = (
            [at for group in configs for at in group]
            if configs and isinstance(configs[0], list)
            else list(configs)
        )
        # select_max_uncertainty_frames needs len(flat) evenly divisible by
        # bin_count (equal contiguous segments) -- trim the tail otherwise.
        mod = len(flat) % self.bin_count
        if mod != 0:
            logger.warning(
                "PeakUncertaintySelector: trajectory size %d is not evenly "
                "divisible by %d segments; dropping the last %d frames.",
                len(flat),
                self.bin_count,
                mod,
            )
            flat = flat[:-mod]
        selected = select_max_uncertainty_frames(flat, self.bin_count)
        logger.info(
            "PeakUncertaintySelector: selected %d peak-uncertainty frames "
            "from %d",
            len(selected),
            len(flat),
        )
        return selected
