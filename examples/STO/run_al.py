"""
Active learning workflow using SSPD with MACE models.

This example demonstrates a three-step active learning loop:
  1. Initial sampling: TorchSimSampler runs NPT parallel tempering with the
                       MACE-MP foundation model to collect seed configurations
                       across a temperature ladder.
  2. Evaluation:       VASPEvaluator runs DFT single-point calculations on the
                       sampled structures to obtain ground-truth energies and forces.
  3. Training:         MaceFinetuningTrainer finetunes a committee of MACE models
                       on the accumulated DFT database.

After the initial data collection, the cycle repeats:
  sampling → evaluation → training → sampling → ...
with finetuned models replacing the foundation model for increasingly
accurate parallel tempering exploration.

Usage:
    Adjust the paths and parameters below to match your cluster setup,
    then run:
        python run_al.py
"""

import os

# Ensure writable TMPDIR (Slurm nodes may restrict /tmp)
_tmpdir = os.path.expanduser("~/tmp")
os.makedirs(_tmpdir, exist_ok=True)
os.environ["TMPDIR"] = _tmpdir

import logging
import sys
from pathlib import Path

import ase.io

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem, LocalFileSystem
from remote_handler.scripts import input_models
from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.staged_active_learning import StagedActiveLearningHandler
from remote_handler.workflows.trainers import MaceModelTrainer
from remote_handler.workflows import active_learning_base
from remote_handler.workflows.selectors import PipelineSelector, RandomSelector
from swieser_scripts.utils.dft_utils import get_settings_from_structure

from lammps_al.run.sampler import LammpsSampler
from lammps_al.run.scripts import script_dir

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ---------------------------------------------------------------------------
# Paths  (adjust to your setup)
# ---------------------------------------------------------------------------
calchome = Path(__file__).parent.resolve()
rh_scripts_dir = Path(input_models.__file__).parent
lammps_scripts_dir = script_dir()
cluster_config_dir = Path(os.environ["REMOTE_HANDLER_CONFIG_DIR"])
template_dir = Path(os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"])
inputs_dir = calchome / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)

# Cluster filesystems
gpu_fs = RemoteFileSystem(
    remote_alias="datalab",
    local_temp_path=_tmpdir,
    identityfile="/home/swieser/.ssh/datalab_key",
)  # GPU cluster for sampling + training

dft_fs = LocalFileSystem()  # CPU cluster for DFT

DATALAB_CONDA = "/home/sandro.wieser/miniconda3/"
DIRONA_CONDA = "/home/swieser/micromamba/"

EVALUATOR_DIR = "/home/swieser/cellar/STO_AL_production/STO_eval_dir"
TRAIN_DIR = "/share/theochem/sandro.wieser/STO_training/MD_train_dir"
SAMPLING_DIR = "/share/theochem/sandro.wieser/STO_training/MD_sample_dir"
MACE_TRAIN_SCRIPT = "mace_run_train" 

# number of structures to sample each iteration
NUM_SAMPLE = 20

# number of ensemble members
NUM_ENSEMBLE = 20

# number of AL iterations
num_iterations = 7

# for sampling
gpu_cluster_config_a100s = ClusterConfig.from_file(
    cluster_config_dir / "datalab_a100s_lammps.json"
)
gpu_cluster_config_a100s.savepath = SAMPLING_DIR

# for training
gpu_cluster_config_a40 = ClusterConfig.from_file(
    cluster_config_dir / "datalab_a100s.json"
)
gpu_cluster_config_a40.savepath = TRAIN_DIR

dft_cluster_config = ClusterConfig.from_file(cluster_config_dir / "dirona.json")
dft_cluster_config.savepath = EVALUATOR_DIR


# ---------------------------------------------------------------------------
# Inputs  (adjust to your system)
# ---------------------------------------------------------------------------
mace_template = template_dir / "mace_config_ft_omat.yaml.template"
foundation_model = Path(os.environ["OMAT0"])
initial_config = calchome / "inputs" / "config" / "poscar.lmp"

# ---------------------------------------------------------------------------
# 1. Sampler: LAMMPS Sampler
# ---------------------------------------------------------------------------


selector = PipelineSelector(
    [
        RandomSelector(n_configs=NUM_SAMPLE,),
    ]
)

lammps_sampler = LammpsSampler(
    remote_fs=gpu_fs,
    template_path=calchome / "config" / "templates" / "lammps_STO.in.template",
    cluster_config=gpu_cluster_config_a100s,
    element_mapping={1: "O", 2: "Sr", 3: "Ti"},
    data_path=initial_config,
    sampling_script=lammps_scripts_dir / "lammps_dump_sampling.py",
    remote_savepath=Path(SAMPLING_DIR),
    loop_time=30,
)

# ---------------------------------------------------------------------------
# 2. Evaluator: VASPEvaluator
# ---------------------------------------------------------------------------
atoms = ase.io.read(initial_config, format="lammps-data")
settings = get_settings_from_structure(atoms, kpts=True)
for key in settings:
    print(key, settings[key])
# add parallelization
settings["nsim"] = 2
settings["ncore"] = 14

vasp_evaluator = VASPEvaluator(
    singlepoint_script=rh_scripts_dir / "dft_singlepoint.py",
    remote_fs=dft_fs,
    cluster_config=dft_cluster_config,
    num_parallel=10,
    remote_savepath=Path(EVALUATOR_DIR),
    n_processors=28,
    loop_time=30,
    required_fraction=0.5,
    incar_kwargs=settings,
    slurm_header_add="", # I put everything in the template
)


# ---------------------------------------------------------------------------
# 3. Trainer: MaceModelTrainer
# ---------------------------------------------------------------------------

mace_trainer = MaceModelTrainer(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config_a40,
    training_script=rh_scripts_dir / "mace" / "mace_training.py",
    config_file=mace_template,
    remote_savepath=Path(TRAIN_DIR),
    remote_mace_train_runner=MACE_TRAIN_SCRIPT,
    loop_time=30,
    num_models=1,
)


# ---------------------------------------------------------------------------
# Active Learning Loop
# ---------------------------------------------------------------------------



class SamplerSchedule(active_learning_base.Schedule):

    def __init__(
        self, num_iterations: int, starting_temperature: float, temperature_step: float
    ):
        self.temperature = starting_temperature
        self.temperature_step = temperature_step
        super().__init__(num_iterations)

    def get_kwargs(self, iteration_idx):
        temperature = self.temperature + self.temperature_step * iteration_idx
        kwargs = {
            "temperature": temperature,
            "timestep": 0.001,
            "n_run": 10000,
            "sampling_frequency": 1,
        }
        return kwargs


sampler_schedule = SamplerSchedule(num_iterations, 1000, 250)


al_handler = StagedActiveLearningHandler.from_actors(
    sampler=lammps_sampler,
    evaluator=vasp_evaluator,
    trainer=mace_trainer,
    calchome=calchome,
    selector=selector,
    selector_reference_from="evaluator",
    initial_selector_reference=None,
    sampler_schedule=sampler_schedule,
    num_iterations=num_iterations,
    cycle_entry_point="sampler",
    cycle_exit_point="training",
    initial_sampler_out=None,
    initial_evaluator_out=None,
    initial_trainer_out=foundation_model,
    combine_evaluator_outputs=True,
    sampler_output_extension="extxyz",
    trainer_output_extension="model",
)

al_handler.run()
