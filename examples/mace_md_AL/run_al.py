import logging
import os
import sys
from pathlib import Path

import cu_co_active_learning
from remote_handler import configure_logging
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.active_learning_base import ActiveLearningHandler

configure_logging()  # timestamped logs (level=INFO -> stdout)

example_dir = Path(__file__).parent.resolve()
input_dir = example_dir / "inputs"

test_files_dir = Path(
    "/home/ffitsch/Documents/university/bsc_thesis/remote_handler/test_files/outputs"
)

dirona_fs = RemoteFileSystem(slurm_config_file="dirona.json")

datalab_a100s_fs = RemoteFileSystem(slurm_config_file="datalab_a100s.json")

datalab_l40s_fs = RemoteFileSystem(slurm_config_file="datalab_l40s.json")

vasp_evaluator = cu_co_active_learning.VASPEvaluator(
    singlepoint_script=example_dir / "dft_singlepoint.py",
    remote_fs=dirona_fs,
    num_parallel=25,
    remote_savepath=Path("/home/ffitsch/dump"),
    incar_kwargs={
        "ncore": 8,
        "algo": "Normal",
        # "prec": "Accurate",
        "xc": "pbe",
        "gga": "RP",
        # "setups": {"H": "_h", "O": "_h"},
        "encut": 300,
        # "encut": 850,
        "ediff": 1e-3,
        "kpts": [3, 3, 1],
        "gamma": True,
        "ismear": 0,
        "sigma": 0.05,
        # "ispin": 2,
        # "ivdw": 11,
        "lwave": False,
        "lcharg": False,
        "lreal": False,
    },
    n_processors=16,
    stub=False,
    loop_time=30,
    required_fraction=1,
)

mace_trainer = cu_co_active_learning.MaceModelTrainer(
    remote_fs=datalab_l40s_fs,
    training_script=example_dir / "mace_training.py",
    template_path=example_dir
    / "config/templates/mace_config_l2.yaml.template",
    remote_savepath=Path("/home/e12218283/dump/dump"),
    remote_mace_train_runner=Path(
        "/home/e12218283/miniconda3/envs/bsc_base/lib/python3.10/site-packages/mace/cli/run_train.py"
    ),
    loop_time=30,
)

lammps_sampler = cu_co_active_learning.LammpsSampler(
    remote_fs=datalab_a100s_fs,
    sampling_script=example_dir / "lammps_sampling.py",
    template_path=example_dir / "config/templates/lammps.in.template",
    data_path=input_dir / "test_fcc100_hollow_2x2.data",
    element_mapping={1: "Co", 2: "Cu", 3: "H", 4: "O"},
    num_samples=2,
    uncertainty_cutoff_radius=4,
    remote_lammps_executable_path=Path(
        "/home/e12218283/software/lammps/build-mliap/lmp"
    ),
    remote_savepath=Path("/home/e12218283/dump/dump"),
    loop_time=30,
)

db_file = input_dir / "cu+cu_co_db_example.xyz"
model_files = [
    input_dir / "models/81_v8_Model_stagetwo.model",
    input_dir / "models/82_v8_Model_stagetwo.model",
    input_dir / "models/83_v8_Model_stagetwo.model",
    input_dir / "models/84_v8_Model_stagetwo.model",
    input_dir / "models/85_v8_Model_stagetwo.model",
]

al_handler = ActiveLearningHandler(
    sampler=lammps_sampler,
    evaluator=vasp_evaluator,
    trainer=mace_trainer,
    calchome=test_files_dir,
    num_iterations=5,
    cycle_entry_point="sampler",
    cycle_exit_point="training",
    initial_evaluator_out=db_file,
    initial_trainer_out=model_files,
    trainer_output_extension="model",
)

al_handler.run()
