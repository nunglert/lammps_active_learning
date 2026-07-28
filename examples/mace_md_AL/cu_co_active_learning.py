import json
import logging
import os
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import (
    ArrayScriptTask,
    DictJSON,
    SingleScriptTask,
)
from remote_handler.core.utils import grep
from remote_handler.dft_runner import VASPHandler
from remote_handler.workflows.active_learning_base import (
    GroundTruthEvaluator,
    ModelTrainer,
    Sampler,
)

logger = logging.getLogger(__name__)


class VASPEvaluator(GroundTruthEvaluator):
    def __init__(
        self,
        singlepoint_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        num_parallel: int,
        incar_kwargs: dict,
        remote_savepath: Optional[os.PathLike] = None,
        launcher: str = "mpirun -n",
        stub: bool = False,  # just for testing
        loop_time: float = 30,
        required_fraction: float = 0.5,
        n_processors: int = 2,
    ):
        """
        Args:
            singlepoint_script: Path to dft_singlepoint.py or another compatible
                evaluation script
            remote_fs: `FileSystemHandlerBase` for the machine on which the evaluation
                should be run
            num_parallel: number of parallel slurm jobs to queue
            incar_kwargs: VASP incar tags
            remote_savepath: Path in which file for remote calculations are stored
            launcher: launch command used for starting VASP including flag for setting
                number of processors
            stub: if True, only writes VASP input files without running calculations
            loop_time: interval (in seconds) at which to poll if the evaluation slurm
                jobs have finished
            required_fraction: fraction of samples for which DFT must succeed to
                progress to the next step
            n_processors: number of CPU cores to run each VASP job on
        """
        super().__init__()
        self.singlepoint_script = singlepoint_script
        self.incar_kwargs = incar_kwargs
        self.remote_fs = remote_fs
        self.num_parallel = num_parallel
        self.stub = stub
        self.launcher = launcher
        self.loop_time = loop_time
        self.required_fraction = required_fraction
        self.remote_savepath = Path(remote_savepath)
        self.n_processors = n_processors

    def update(
        self, calcdir: Path, evaluator_input: Path | list[Path], **kwargs
    ) -> None:
        self.dft_dir = Path(calcdir) / "dft"
        self.results_dir = self.dft_dir / "results"
        os.makedirs(self.dft_dir, exist_ok=True)
        it = str(calcdir).split("/")[-1]

        traj_files = (
            [str(evaluator_input)]
            if isinstance(evaluator_input, Path)
            else [str(p) for p in evaluator_input]
        )

        self.vasp_handler = VASPHandler(
            loop_time=self.loop_time,
            jobname="dft",
            every=1,
            start=None,
            end=None,
            logfile="dft.log",
            tag="dft",
            stub=self.stub,
            collect_only=False,
            remote_fs=self.remote_fs,
            savepath=self.remote_fs.slurm_config["savepath"],
            local_savepath=self.results_dir,
            traj_files=traj_files,
            python_script_path=self.singlepoint_script,
            num_parallel=self.num_parallel,
            n_processors=self.n_processors,
            incar_kwargs=self.incar_kwargs,
            launcher=self.launcher,
        )

    def get_evaluated_configs(self) -> Path:
        self.vasp_handler.run()

        if not self.vasp_handler.finish_check(self.required_fraction):
            raise Exception(
                f"Less than {self.required_fraction * 100}% ran successfully. "
                "Something probably went wrong. Otherwise, lower the "
                "required_fraction parameter."
            )

        all_ats = []
        for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
            try:
                at = ase.io.read(Path(calcpath) / "vasprun.xml")
                all_ats.append(at)
            except:
                logger.info(f"VASP run failed for {calcpath=}")

        for s in all_ats:
            s.info["REF_energy"] = s.get_potential_energy()
            s.arrays["REF_forces"] = s.get_forces()
            s.calc = None

        path = self.results_dir / "evaluated_configs.extxyz"
        ase.io.write(path, all_ats)
        return path


class LammpsSampler(Sampler):
    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        sampling_script: os.PathLike,  # lammps_sampling.py
        template_path: os.PathLike,  # lammps.in.template
        data_path: os.PathLike,  # lammps .data file
        element_mapping: dict,
        num_samples: int,
        uncertainty_cutoff_radius: float,
        remote_lammps_executable_path: os.PathLike,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
    ):
        """
        Args:
            remote_fs: `FileSystemHandlerBase` for the machine on which the sampling
                should be run
            sampling_script: Path to lammps_sampling.py or another compatible sampling
                script
            template_path: Path to a jinja-template of the lammps.in file which should
                be used for sampling
            data_path: Path to a lammps.data file which contains the starting configuration
                for all sampling MD runs
            element_mapping: Dict that maps the atom species number from the given
                data file to the corresponding element symbol
            num_samples: number of configurations to sample each generation
            uncertainty_cutoff_radius: cutoff radius (in Angstrom) for the calculation
                of local uncertainties
            remote_lammps_executable_path: Path to lammps executable on the machine on
                which the sampling script is run
            remote_savepath: Path in which files for the remote calculations are stored
            loop_time: interval (in seconds) at which to poll if the sampling slurm job
                has finished
        """
        super().__init__()
        self.remote_fs = remote_fs
        self.sampling_script = sampling_script
        self.template_path = template_path
        self.data_path = data_path
        self.element_mapping = element_mapping
        self.num_samples = num_samples
        self.uncertainty_cutoff_radius = uncertainty_cutoff_radius
        self.remote_lammps_executable_path = remote_lammps_executable_path
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time

    def update(
        self, calcdir: Path, sampler_input: list[Path], **kwargs
    ) -> None:
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.model_files = sampler_input

        script_input = {
            "lammps_path": Path(self.remote_lammps_executable_path).as_posix(),
            "template_file": Path(self.template_path).name,
            "data_file": Path(self.data_path).name,
            "model_files": [p.name for p in self.model_files],
            "lammps_config_name": "lammps.in",
            "element_mapping": self.element_mapping,
            "cutoff_radius": self.uncertainty_cutoff_radius,
            "bin_count": self.num_samples,
            "seed": np.random.randint(1e6),
        }

        self.script_input_file = (
            self.results_dir / "input_sample.json"
        )  # script input file
        with open(self.script_input_file, "w") as f:
            f.write(json.dumps(script_input))

        self.sampling_task = SingleScriptTask(
            local_savepath=self.results_dir,
            input_file=self.script_input_file,
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

        self.sampling_task.allocate_data()  # moves script and input files

        self.sampling_task.copy_file(self.template_path)
        self.sampling_task.copy_file(self.data_path)
        for p in self.model_files:
            self.sampling_task.copy_file(p)

    def get_sampled_configs(self) -> Path:
        self.sampling_task.submit()
        self.sampling_task.collect_results()

        return Path(self.sampling_task.local_savepath) / "out/samples.xyz"


class MaceModelTrainer(ModelTrainer):
    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        training_script: os.PathLike,
        template_path: os.PathLike,
        remote_mace_train_runner: os.PathLike,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
    ):
        """
        Args:
            remote_fs: `FileSystemHandlerBase` for the machine on which the training
                should be run
            training_script: Path to mace_training.py or another compatible training
                script.
            template_path: Path to a jinja2-template of the mace_config.yaml file which
                should be used for training
            num_models: Number of models to train in each generation
            validation_fraction: fraction for train/validation split
            remote_mace_train_runner: Path to MACE run_train.py script on the machine
                on which the sampling script is run
            remote_savepath: Path in which files for the remote calculations are stored
            loop_time: interval (in seconds) at which to poll if all training slurm jobs
                have finished
        """
        super().__init__()
        self.training_script = training_script
        self.remote_fs = remote_fs
        self.template_path = template_path
        self.num_models = num_models
        self.validation_fraction = validation_fraction
        self.remote_mace_train_runner = remote_mace_train_runner
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time

    def update(
        self, calcdir: Path, trainer_input: Path | list[Path], **kwargs
    ) -> None:
        self.train_dir = calcdir / "train"
        self.results_dir = self.train_dir / "results"
        os.makedirs(self.train_dir, exist_ok=True)

        db_files = (
            [trainer_input]
            if isinstance(trainer_input, Path)
            else trainer_input
        )

        script_input_shared = {
            "database_files": [Path(p).name for p in db_files],
            "template_file": Path(self.template_path).name,
            "model_name": "Model",
            "validation_fraction": self.validation_fraction,
            "train_runner": Path(self.remote_mace_train_runner).as_posix(),
        }

        individual_resources = [
            [
                DictJSON(
                    content=script_input_shared
                    | {"seed": np.random.randint(1e6)},
                    filename="input_train.json",
                    remote_fs=self.remote_fs,
                )
                for i in range(self.num_models)
            ]
        ]

        self.training_task = ArrayScriptTask(
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            python_script_file=self.training_script,
            individual_ressources=individual_resources,
            shared_ressources=[self.template_path] + db_files,
            logfile="train.log",
            remote_savepath=self.remote_savepath,
            collect_only=False,
            num_parallel=5,
            loop_time=self.loop_time,
            jobname="train",
            n_processors=1,
            jobfile_name="run_train.sh",
        )

        self.training_task.allocate_data()

    def get_trained_model(self) -> list[Path]:
        self.training_task.submit()
        self.training_task.collect_results()

        model_paths = []
        for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
            model_paths.append(
                Path(calcpath) / f"models/{self.model_name}_stagetwo.model"
            )

        return model_paths
