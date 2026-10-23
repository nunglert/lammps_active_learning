"""The LAMMPS MD Sampler for this project.

In the layered architecture, concrete samplers belong to the *project*, not to
the orchestrator: ``remote_handler`` owns the ``Sampler`` *interface* and the
generic remote-execution machinery (``SingleScriptTask``, ``ClusterConfig``,
the SLURM templating), while the application picks and wires the actual
sampling method.

``LammpsSampler`` is a **native** implementation of that interface for LAMMPS
molecular dynamics, generic over the committee's potential family
(``backend``: ``"mace"`` default, or ``"ace"``/``"grace"`` -- see
``lammps_sampling.py``'s module docstring). Per AL iteration it:

    1. compiles the committee's first MACE model for LAMMPS (backend="mace"
       only -- ACE/GRACE's native pair styles need no compile step) and
       renders the user's ``lammps.in`` jinja2 template;
    2. runs a LAMMPS MD trajectory on the cluster via a ``SingleScriptTask``
       (shipping the payload ``scripts/lammps_sampling.py``, located by
       :func:`lammps_al.run.scripts.script_dir`);
    3. converts the raw dump to a proper ``trajectory.extxyz`` (correct
       chemical symbols via ``element_mapping``) and returns it.

This sampler only generates dynamics -- it does not score or select
candidates. Committee-uncertainty scoring and peak-frame selection is a
separate actor/loop-role,
:class:`lammps_al.select.selector.CommitteeUncertaintySelector` (a
``Selector``, consuming this sampler's trajectory output plus the
committee's model files), mirroring this repo's existing
``ROIExtractor``/``EmbeddingAnnealer`` split (a ``Sampler`` doing dynamics
and a peer ``Selector`` doing selection, rather than one actor doing both).

For ``backend="mace"``, step 1 targets one of two LAMMPS/MACE coupling
routes, chosen per run via ``lammps_pair_style`` (default ``"mliap"``,
unchanged production behaviour):

* ``"mliap"`` (**default, production**) -- the ML-IAP Python bridge
  (``pair_style mliap unified``), the KOKKOS-on-GPU setup this sampler has
  always targeted.
* ``"mace"`` (**CPU-testable, opt-in**) -- LAMMPS's native ``pair_style mace``
  (the ``ML-MACE`` package, built against LibTorch directly). For any MACE
  model with more than one interaction layer (i.e. essentially all real
  models), the ML-IAP route needs LAMMPS's ghost-atom message-passing hooks,
  which are Kokkos-CUDA-only -- confirmed by building both the ACEsuit
  LAMMPS fork and vanilla upstream ``develop`` and hitting ``AttributeError:
  ... no attribute 'forward_exchange'`` on CPU either way. ``"mace"`` has no
  such limitation and is what the CPU-only ``lammps-smoke`` CI job (and
  local Docker-based testing, see ``docker/Dockerfile.lammps-mace``) use --
  it exists purely to make this sampler's payload testable without a GPU,
  not to replace the production route.

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler + a
MACE stack). The cluster conda env that runs the payload needs ``lammps_al``
(for the input schema), ``mace-torch``, ``torch`` and a LAMMPS build with
either package (``mliap`` needs ML-IAP + Kokkos-CUDA; ``mace`` needs the
``ML-MACE`` package, CPU or GPU); the host side needs just
``remote_handler`` + ``ase``.

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

from lammps_al.run.inputs import FINAL_DATA_NAME
from lammps_al.run.scripts import script_dir

logger = logging.getLogger(__name__)

# Keys LammpsSampler itself recognizes and sets in script_input (mirrors
# LammpsSamplingInput's own field names, minus extra_template_kwargs itself).
# Any update(**kwargs) key outside this set is a schedule/template parameter
# (e.g. temperature/timestep/n_run) forwarded via extra_template_kwargs
# instead of being written as a top-level key -- a top-level unknown key
# would otherwise be silently dropped as an "unknown key" by
# LammpsSamplingInput.from_file (the same class of bug that once ate a
# dump_frequency override under a colliding field name).
_KNOWN_SCRIPT_INPUT_KEYS = {
    "lammps_path",
    "template_file",
    "data_file",
    "model_files",
    "lammps_config_name",
    "seed",
    "lammps_args",
    "lammps_pair_style",
    "enable_cueq",
    "backend",
    "element_mapping",
    "dump_frequency",
    "max_restarts",
    "restart_frequency",
}


def remap_elements(
    structs: list, element_mapping: Optional[dict] = None
) -> list:
    """Convert LAMMPS atom-type indices to element symbols, in place."""
    if not element_mapping:
        return structs
    for atoms in structs:
        for atom in atoms:
            symbol = element_mapping.get(str(atom.number))
            if symbol is not None:
                atom.symbol = symbol
    return structs


class LammpsSampler(Sampler):
    """MD sampler using LAMMPS: runs dynamics only, no scoring/selection.

    Runs LAMMPS molecular dynamics and returns the resulting trajectory as a
    proper extxyz file. Supports stub mode for testing without actual LAMMPS
    execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        template_path: Path to jinja2 template for the lammps.in file
        data_path: Path to lammps.data file with the starting configuration
        element_mapping: Dict mapping atom type numbers to element symbols,
            applied when converting the raw LAMMPS dump to
            ``out/trajectory.extxyz``.
        remote_lammps_executable_path: Path to the LAMMPS executable on the
            remote machine
        sampling_script: Remote payload; defaults to the shipped
            ``lammps_sampling.py``.
        remote_savepath: Path on the remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        lammps_args: LAMMPS accelerator/package flags for the ``lmp`` command
            (inserted between ``-in`` and ``-log``). Defaults to KOKKOS-on-GPU
            (paired with the default ``lammps_pair_style="mliap"``);
            ``lammps_pair_style="mace"`` needs none of these on a CPU-only
            build, so pass "" there.
        lammps_pair_style: For ``backend="mace"``: ``"mliap"`` (default,
            production KOKKOS-on-GPU) or ``"mace"`` (CPU-testable, opt-in --
            see the module docstring). For ``backend="ace"``/``"grace"``:
            the literal LAMMPS pair_style line rendered into the template
            (e.g. ``"pace"`` or ``"grace pad_verbose padding 0.1"`` -- these
            need no compile step, unlike MACE).
        enable_cueq: Convert the MACE model to cuequivariance before compiling
            it for LAMMPS (production default). Only used when
            ``backend="mace"`` and ``lammps_pair_style="mliap"``.
        backend: Which committee-potential family drives the MD's first
            (driving) member: ``"mace"`` (default), ``"ace"`` (pacemaker),
            or ``"grace"`` (gracemaker) -- see ``lammps_sampling.py``'s
            module docstring.
        dump_frequency: Steps between dumped MD frames, rendered into the
            LAMMPS template as ``{{ dump_frequency }}``. ``None`` (default)
            falls back to every step at render time -- fine for small
            systems, but for large ones (hundreds of thousands of atoms)
            this can produce a huge dump file in minutes; set explicitly for
            those.
        max_restarts: Retries of a LAMMPS run that exits non-zero, e.g.
            after a sporadic GPU fault (default 0). With
            ``restart_frequency`` set and a template that supports it, a
            retry resumes from the newest restart file instead of starting
            over -- see ``lammps_sampling.py``'s module docstring.
        restart_frequency: Steps between LAMMPS restart files, rendered
            into the template as ``{{ restart_frequency }}``.
        continue_from_previous: Start each iteration's MD from the final
            frame of the most recent earlier iteration's run (the
            template's ``write_data {{ final_data_file }}`` output, collected
            under ``<iteration>/sample/results/out/``) instead of always
            restarting from ``data_path``. Falls back to ``data_path`` when
            no earlier iteration has one (e.g. the first sampling
            iteration). An explicit ``struct`` schedule kwarg still wins.
        stub: Dry-run without a cluster; emit a mock trajectory for testing
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        template_path: os.PathLike,
        data_path: os.PathLike,
        element_mapping: Optional[dict] = None,
        remote_lammps_executable_path: os.PathLike = "lmp",
        sampling_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        lammps_args: str = "-k on g 1 -sf kk -pk kokkos newton on neigh half",
        lammps_pair_style: str = "mliap",
        enable_cueq: bool = True,
        backend: str = "mace",
        stub: bool = False,
        cluster_config: Optional[ClusterConfig] = None,
        dump_frequency: Optional[int] = None,
        continue_from_previous: bool = False,
        max_restarts: int = 0,
        restart_frequency: Optional[int] = None,
    ):
        super().__init__()
        if sampling_script is None:
            sampling_script = script_dir() / "lammps_sampling.py"
        self.remote_fs = remote_fs
        self.sampling_script = sampling_script
        self.template_path = template_path
        self.data_path = data_path
        self.initial_data_path = data_path
        self.continue_from_previous = continue_from_previous
        self.element_mapping = element_mapping
        self.dump_frequency = dump_frequency
        self.max_restarts = max_restarts
        self.restart_frequency = restart_frequency
        self.remote_lammps_executable_path = remote_lammps_executable_path
        self.lammps_args = lammps_args
        self.lammps_pair_style = lammps_pair_style
        self.enable_cueq = enable_cueq
        self.backend = backend
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
            "lammps_pair_style": self.lammps_pair_style,
            "enable_cueq": self.enable_cueq,
            "backend": self.backend,
            "element_mapping": self.element_mapping,
            "dump_frequency": self.dump_frequency,
            "max_restarts": self.max_restarts,
            "restart_frequency": self.restart_frequency,
        }

    def update(
        self, calcdir: Path, sampler_input: Path | list[Path], **kwargs
    ) -> None:
        """Prepare (and stage) the LAMMPS run for this AL iteration.

        Schedule ``**kwargs`` are merged into the payload's input JSON,
        overriding the defaults. Keys matching one of ``LammpsSampler``'s
        own script-input fields override that field; any other key (e.g. a
        ``Schedule``'s ``temperature``/``timestep``/``n_run``) is forwarded
        via ``extra_template_kwargs`` so the payload can fill the LAMMPS
        template with it.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: List of model file paths (the committee)
            **kwargs: Schedule parameters merged into the sampling input
        """
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        if isinstance(sampler_input, list):
            self.model_files = sampler_input
        else:
            self.model_files = [sampler_input]

        if "struct" in kwargs:
            self.data_path = kwargs["struct"]
        elif self.continue_from_previous:
            self.data_path = self._previous_final_data(calcdir)

        # In stub mode we don't stage a remote task.
        if self.stub:
            return

        script_input = self._load_default_settings()
        extra_template_kwargs = {}
        for key, value in kwargs.items():
            if key == "struct":
                continue
            if key in _KNOWN_SCRIPT_INPUT_KEYS:
                script_input[key] = value
            else:
                extra_template_kwargs[key] = value
        script_input["extra_template_kwargs"] = extra_template_kwargs
        for key in list(script_input.keys()):
            if isinstance(script_input[key], Path):
                script_input[key] = str(script_input[key].name)

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

    def _previous_final_data(self, calcdir: Path) -> os.PathLike:
        """The final-frame data file of the most recent earlier iteration
        (sibling calcdirs named by iteration number), or ``data_path`` if
        there is none."""
        calcdir = Path(calcdir)
        if not calcdir.name.isdigit():
            return self.initial_data_path
        current = int(calcdir.name)
        earlier = sorted(
            (
                int(d.name)
                for d in calcdir.parent.iterdir()
                if d.is_dir() and d.name.isdigit() and int(d.name) < current
            ),
            reverse=True,
        )
        for iteration in earlier:
            path = (
                calcdir.parent
                / str(iteration)
                / "sample"
                / "results"
                / "out"
                / FINAL_DATA_NAME
            )
            if path.is_file():
                logger.info(
                    "Continuing MD from iteration %d's final frame: %s",
                    iteration,
                    path,
                )
                return path
        logger.info(
            "No earlier iteration's final frame found; starting MD from %s",
            self.initial_data_path,
        )
        return self.initial_data_path

    def _create_stub_outputs(self, num_frames: int = 10) -> list[Atoms]:
        """Create a mock LAMMPS trajectory for testing.

        Args:
            num_frames: Number of trajectory frames to generate

        Returns:
            List of Atoms objects (no uncertainty -- this sampler no longer
            scores anything; see
            ``lammps_al.select.selector.CommitteeUncertaintySelector``)
        """
        logger.info(f"Creating {num_frames} stub LAMMPS trajectory frames")

        mock_frames = []
        for i in range(num_frames):
            # Create mock Cu/Co surface structure
            symbols = ["Cu"] * 20 + ["Co"] * 4 + ["H"] * 2
            positions = np.random.uniform(0, 10, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[10, 10, 15],
                pbc=True,
            )
            atoms.info["step"] = i * 100

            mock_frames.append(atoms)

        return mock_frames

    def get_sampled_configs(self) -> Path:
        """Run LAMMPS sampling and collect the resulting trajectory.

        Returns:
            Path to the extxyz file containing the MD trajectory
        """
        if self.stub:
            # In stub mode, create a mock trajectory directly
            logger.info(
                "Running in stub mode - creating mock LAMMPS trajectory"
            )
            trajectory = self._create_stub_outputs()
            output_path = self.results_dir / "out" / "trajectory.extxyz"
            os.makedirs(output_path.parent, exist_ok=True)
            ase.io.write(output_path, trajectory)
            logger.info(f"Trajectory written to {output_path}")
            return output_path
        else:
            # Run actual LAMMPS sampling
            self.sampling_task.submit()
            if not self.sampling_task.finish_check():
                raise WorkflowError(
                    "LAMMPS sampling did not finish successfully "
                    "(no is_finished.info marker). Check the job logs in "
                    f"{self.sampling_task.calcdir}."
                )
            self.sampling_task.collect_results()

            output_path = (
                Path(self.sampling_task.local_savepath)
                / "out/trajectory.extxyz"
            )

        logger.info(f"Trajectory written to {output_path}")
        return output_path
