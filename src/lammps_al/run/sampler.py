"""The LAMMPS MD-uncertainty Sampler for this project.

In the layered architecture, concrete samplers belong to the *project*, not to
the orchestrator: ``remote_handler`` owns the ``Sampler`` *interface* and the
generic remote-execution machinery (``SingleScriptTask``, ``ClusterConfig``,
the SLURM templating), while the application picks and wires the actual
sampling method.

``LammpsSampler`` is a **native** implementation of that interface for LAMMPS
molecular dynamics with committee-uncertainty sampling. Per AL iteration it:

    1. compiles the committee's first MACE model for LAMMPS (MLIAP) and renders
       the user's ``lammps.in`` jinja2 template;
    2. runs a LAMMPS MD trajectory on the cluster via a ``SingleScriptTask``
       (shipping the payload ``scripts/lammps_sampling.py``, located by
       :func:`lammps_al.run.scripts.script_dir`);
    3. scores every frame's local force-uncertainty against the committee,
       splits the trajectory into ``num_samples`` segments, and returns the
       most-uncertain frame from each.

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler + a
MACE stack). The cluster conda env that runs the payload needs ``lammps_al``
(for the input schema), ``mace-torch``, ``torch`` and a LAMMPS build with the
MLIAP/MACE interface; the host side needs just ``remote_handler`` + ``ase``.

``stub=True`` dry-runs the loop (mock samples, no cluster) for testing.
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
from remote_handler.workflows.active_learning_base import Sampler

from lammps_al.run.scripts import script_dir

logger = logging.getLogger(__name__)


def remap_elements(
    structs: list, element_mapping: Optional[dict] = None
) -> list:
    """Convert LAMMPS atom-type indices to element symbols, in place.

    """
    if not element_mapping:
        return structs
    for atoms in structs:
        for atom in atoms:
            symbol = element_mapping.get(str(atom.number))
            if symbol is not None:
                atom.symbol = symbol
    return structs


class LammpsSampler(Sampler):
    """MD sampler using LAMMPS with committee-uncertainty-based sampling.

    Runs LAMMPS molecular dynamics and selects configurations based on model
    uncertainty. Supports stub mode for testing without actual LAMMPS
    execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        template_path: Path to jinja2 template for the lammps.in file
        data_path: Path to lammps.data file with the starting configuration
        element_mapping: Dict mapping atom type numbers to element symbols
        num_samples: Number of configurations to sample per generation
        uncertainty_cutoff_radius: Cutoff radius (Angstrom) for local
            uncertainty aggregation
        remote_lammps_executable_path: Path to the LAMMPS executable on the
            remote machine
        sampling_script: Remote payload; defaults to the shipped
            ``lammps_sampling.py``.
        remote_savepath: Path on the remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        lammps_args: LAMMPS accelerator/package flags for the ``lmp`` command
            (inserted between ``-in`` and ``-log``). Defaults to KOKKOS-on-GPU;
            pass a CPU variant (or "") for a non-GPU build.
        device: torch device for the MACE committee scoring ("cuda" or "cpu").
        enable_cueq: Convert the MACE model to cuequivariance before compiling
            it for LAMMPS (production default). Set False to compile the plain
            e3nn model without the cuequivariance requirement (CPU builds).
        stub: Dry-run without a cluster; emit mock samples for testing
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        template_path: os.PathLike,
        data_path: os.PathLike,
        element_mapping: Optional[dict] = None,
        num_samples: Optional[int] = None,
        uncertainty_cutoff_radius: Optional[float] = None,
        remote_lammps_executable_path: os.PathLike = "lmp",
        sampling_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        lammps_args: str = "-k on g 1 -sf kk -pk kokkos newton on neigh half",
        device: str = "cuda",
        enable_cueq: bool = True,
        stub: bool = False,
        cluster_config: Optional[ClusterConfig] = None,
        sampling_frequency: Optional[int] = None,
    ):
        super().__init__()
        if sampling_script is None:
            sampling_script = script_dir() / "lammps_sampling.py"
        self.remote_fs = remote_fs
        self.sampling_script = sampling_script
        self.template_path = template_path
        self.data_path = data_path
        self.element_mapping = element_mapping
        self.num_samples = num_samples
        self.sampling_frequency = sampling_frequency
        self.uncertainty_cutoff_radius = uncertainty_cutoff_radius
        self.remote_lammps_executable_path = remote_lammps_executable_path
        self.lammps_args = lammps_args
        self.device = device
        self.enable_cueq = enable_cueq
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.cluster_config = cluster_config
        self.stub = stub

    def _load_default_settings(self) -> dict:
        """Default script-input settings. Schedule ``**kwargs`` override these
        in :meth:`update`

        """
        return {
            "lammps_path": str(Path(self.remote_lammps_executable_path)),
            "template_file": Path(self.template_path).name,
            "data_file": Path(self.data_path).name,
            "model_files": [Path(m).name for m in self.model_files],
            "lammps_config_name": "lammps.in",
            "seed": int(np.random.randint(1e6)),
            "lammps_args": self.lammps_args,
            "device": self.device,
            "enable_cueq": self.enable_cueq,
            "bin_count": self.num_samples,
            "cutoff_radius": self.uncertainty_cutoff_radius,
            "element_mapping": self.element_mapping,
            "sampling_frequency": self.sampling_frequency,
        }

    def update(
        self, calcdir: Path, sampler_input: Path | list[Path], **kwargs
    ) -> None:
        """Prepare (and stage) the LAMMPS run for this AL iteration.

        Schedule ``**kwargs`` are merged into the payload's input JSON,
        overriding the defaults. Keys such as
        ``temperature``/``timestep``/``n_run`` are forwarded verbatim so the
        payload can fill the LAMMPS template with them.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: List of model file paths (the committee)
            **kwargs: Schedule parameters merged into the sampling input
        """
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        if type(sampler_input) == Path:
            sampler_input = [sampler_input]
        else:
            self.model_files = sampler_input

        if "struct" in kwargs:
            self.data_path = kwargs["struct"]

        # In stub mode we don't stage a remote task.
        if self.stub:
            return

        script_input = self._load_default_settings()
        for key in kwargs.keys():
            script_input[key] = kwargs[key]
        for key in list(script_input.keys()):
            if isinstance(script_input[key], Path):
                script_input[key] = str(script_input[key].name)

        # num_samples (bin_count) takes precedence over sampling_frequency, as
        # in the previous behaviour. Warn so the user knows which one won.
        if (
            script_input.get("bin_count") is not None
            and script_input.get("sampling_frequency") is not None
        ):
            logger.warning(
                "Both num_samples (%s) and sampling_frequency (%s) were "
                "specified; using num_samples (previous behavior) and "
                "ignoring sampling_frequency.",
                script_input["bin_count"],
                script_input["sampling_frequency"],
            )
            script_input.pop("sampling_frequency", None)

        self.script_input_file = self.results_dir / "input_sample.json"
        with open(self.script_input_file, "w") as f:
            f.write(json.dumps(script_input, indent=2))

        self.sampling_task = SingleScriptTask(
            local_savepath=self.results_dir,
            input_file=self.script_input_file,
            cluster_config=self.cluster_config,
            remote_fs=self.remote_fs,
            script_file=self.sampling_script,
            remote_savepath=self.remote_savepath,
            jobfile_name="run_sample.sh",
            jobname="sample",
            logfile="sample.log",
            n_processors=1,
            loop_time=self.loop_time,
            collect_only=False,
        )

        self.sampling_task.allocate_data()

        # Copy required files to remote
        self.sampling_task.copy_file(self.template_path)
        self.sampling_task.copy_file(self.data_path)
        for p in self.model_files:
            self.sampling_task.copy_file(p)

    def _create_stub_outputs(self, num_samples: int = None) -> list[Atoms]:
        """Create mock LAMMPS sampling outputs for testing.

        Generates a fake MD trajectory with realistic uncertainty scores.

        Args:
            num_samples: Number of samples to generate (uses self.num_samples
                if None)

        Returns:
            List of Atoms objects with mock uncertainties
        """
        if num_samples is None:
            num_samples = self.num_samples

        logger.info(f"Creating {num_samples} stub LAMMPS samples")

        mock_samples = []
        for i in range(num_samples):
            # Create mock Cu/Co surface structure
            symbols = ["Cu"] * 20 + ["Co"] * 4 + ["H"] * 2
            positions = np.random.uniform(0, 10, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[10, 10, 15],
                pbc=True,
            )

            # Add mock uncertainty scores
            atoms.info["max_uncertainty"] = np.random.uniform(0.1, 2.0)
            atoms.info["mean_uncertainty"] = np.random.uniform(0.05, 1.0)
            atoms.info["step"] = i * 100

            mock_samples.append(atoms)

        return mock_samples

    def get_sampled_configs(self) -> Path:
        """Run LAMMPS sampling and collect configurations.

        Returns:
            Path to the XYZ file containing sampled configurations
        """
        if self.stub:
            # In stub mode, create mock outputs directly
            logger.info("Running in stub mode - creating mock LAMMPS samples")
            sampled_configs = self._create_stub_outputs()
            output_path = self.results_dir / "out" / "samples.xyz"
            os.makedirs(output_path.parent, exist_ok=True)
            ase.io.write(output_path, sampled_configs)
            logger.info(
                f"Sampled configurations written to {output_path}"
            )
            return output_path

        # Run actual LAMMPS sampling
        self.sampling_task.submit()
        if not self.sampling_task.finish_check():
            raise WorkflowError(
                "LAMMPS sampling did not finish successfully "
                "(no is_finished.info marker). Check the job logs in "
                f"{self.sampling_task.calcdir}."
            )
        self.sampling_task.collect_results()

        results_path = Path(self.sampling_task.local_savepath) / "out"

        # Read the full trajectory the payload wrote and fix up the element
        # symbols (the dump only stores atom-type indices).
        traj_paths = sorted(results_path.glob("md_sampling.dump"))
        frames: list[Atoms] = []
        for tp in traj_paths:
            frames.extend(
                ase.io.read(str(tp), format="lammps-dump-text", index=":")
            )
        remap_elements(frames, self.element_mapping)

        # Selection is handled by a dedicated SelectorActor stage in the
        # StagedActiveLearningHandler, so the sampler just emits the frames.
        output_path = results_path / "samples.xyz"
        ase.io.write(output_path, frames)

        logger.info(f"Sampled configurations written to {output_path}")
        return output_path
