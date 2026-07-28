"""Active learning workflow: LAMMPS MD sampling with a MACE committee.

A Cu/Co/H/O surface active-learning loop with three actors:
  1. Sampling:   ``LammpsSampler`` (this package) runs LAMMPS MD driven by the
                 first committee model (compiled to the MLIAP interface), scores
                 every frame's local force-uncertainty across the committee, and
                 returns the most-uncertain frame of each trajectory segment.
  2. Evaluation: ``VASPEvaluator`` (remote_handler) runs DFT single-points on
                 the sampled structures for ground-truth energies/forces.
  3. Training:   ``MaceModelTrainer`` (remote_handler) retrains the MACE
                 committee on the accumulated DFT database.

The loop seeds from an existing committee + DFT database and then repeats
  sampling -> evaluation -> training -> sampling -> ...
with the freshly trained committee driving the next round of MD.

Concrete samplers live in the *project*, not the orchestrator: the sampler
comes from ``lammps_al``; ``remote_handler`` provides the loop, the evaluator,
the trainer, and the remote-execution machinery. The shipped remote payloads
(the LAMMPS sampling script, ``dft_singlepoint.py``, ``mace_training.py``) are
used by default -- no per-example copies needed.

Usage:
    Adjust the cluster configs, SSH aliases, and paths below to your setup
    (they are placeholders from a prior user), point
    ``REMOTE_HANDLER_SLURM_TEMPLATES`` at ``config/templates`` for the SLURM
    header + body templates, then run:
        python run_al.py
"""

import os

# Ensure a writable TMPDIR (Slurm nodes may restrict /tmp).
_tmpdir = os.path.expanduser("~/tmp")
os.makedirs(_tmpdir, exist_ok=True)
os.environ["TMPDIR"] = _tmpdir

from pathlib import Path

from remote_handler import configure_logging
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows import Stage, StagedActiveLearningHandler
from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.trainers import MaceModelTrainer

from lammps_al.run.sampler import LammpsSampler

configure_logging()  # timestamped logs (level=INFO -> stdout)

# ---------------------------------------------------------------------------
# Paths  (adjust to your setup)
# ---------------------------------------------------------------------------
calchome = Path(__file__).parent.resolve()
inputs_dir = calchome / "inputs"
config_dir = calchome / "config"

# ---------------------------------------------------------------------------
# Clusters  (SSH aliases resolved via ~/.ssh/config; adjust to your setup)
# ---------------------------------------------------------------------------
# GPU cluster: LAMMPS MD sampling + MACE training.
gpu_fs = RemoteFileSystem(remote_alias="datalab", local_temp_path=_tmpdir)
gpu_cluster_config = ClusterConfig(
    remote_machine="datalab",
    partition="GPU-a100s",
    conda_env="bsc_base",
    conda_source="/home/e12218283/miniconda3",
    savepath="/home/e12218283/dump/remote_handler",
    template="datalab.sh.j2",
)

# CPU cluster: DFT single-points.
dft_fs = RemoteFileSystem(remote_alias="dirona", local_temp_path=_tmpdir)
dft_cluster_config = ClusterConfig(
    remote_machine="dirona",
    partition="normal",
    conda_env="bsc_base",
    conda_source="/home/ffitsch/miniconda3",
    savepath="/home/ffitsch/dump/remote_handler",
    template="dirona.sh.j2",
)

# ---------------------------------------------------------------------------
# Inputs  (Cu/Co/H/O surface; adjust to your system)
# ---------------------------------------------------------------------------
db_file = inputs_dir / "cu+cu_co_db_example.xyz"  # seed DFT database
lammps_data = inputs_dir / "test_fcc100_hollow_2x2.data"  # MD start structure
lammps_template = config_dir / "templates" / "lammps.in.template"
mace_template = config_dir / "templates" / "mace_config_l2.yaml.template"
model_files = [
    inputs_dir / "models" / "81_v8_Model_stagetwo.model",
    inputs_dir / "models" / "82_v8_Model_stagetwo.model",
    inputs_dir / "models" / "83_v8_Model_stagetwo.model",
    inputs_dir / "models" / "84_v8_Model_stagetwo.model",
    inputs_dir / "models" / "85_v8_Model_stagetwo.model",
]

# ---------------------------------------------------------------------------
# 1. Sampler: LammpsSampler (MACE committee-uncertainty MD)
# ---------------------------------------------------------------------------
# ``sampling_script`` defaults to the payload shipped by lammps_al; the LAMMPS
# accelerator flags / MACE device default to KOKKOS-on-GPU.
lammps_sampler = LammpsSampler(
    remote_fs=gpu_fs,
    template_path=lammps_template,
    data_path=lammps_data,
    element_mapping={1: "Co", 2: "Cu", 3: "H", 4: "O"},
    num_samples=2,
    uncertainty_cutoff_radius=4.0,
    remote_lammps_executable_path=Path(
        "/home/e12218283/software/lammps/build-mliap/lmp"
    ),
    remote_savepath=Path("/home/e12218283/dump/remote_handler"),
    loop_time=30,
)

# ---------------------------------------------------------------------------
# 2. Evaluator: VASPEvaluator  (singlepoint_script defaults to the shipped one)
# ---------------------------------------------------------------------------
vasp_evaluator = VASPEvaluator(
    remote_fs=dft_fs,
    cluster_config=dft_cluster_config,
    num_parallel=25,
    n_processors=16,
    remote_savepath=Path("/home/ffitsch/dump/dft"),
    incar_kwargs={
        "ncore": 8,
        "algo": "Normal",
        "xc": "pbe",
        "gga": "RP",
        "encut": 300,
        "ediff": 1e-3,
        "kpts": [3, 3, 1],
        "gamma": True,
        "ismear": 0,
        "sigma": 0.05,
        "lwave": False,
        "lcharg": False,
        "lreal": False,
    },
    required_fraction=1.0,
    loop_time=30,
)

# ---------------------------------------------------------------------------
# 3. Trainer: MaceModelTrainer  (training_script defaults to the shipped one)
# ---------------------------------------------------------------------------
mace_trainer = MaceModelTrainer(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config,
    config_file=mace_template,
    remote_mace_train_runner=Path(
        "/home/e12218283/miniconda3/envs/bsc_base/lib/python3.10/"
        "site-packages/mace/cli/run_train.py"
    ),
    num_models=len(model_files),
    remote_savepath=Path("/home/e12218283/dump/remote_handler"),
    loop_time=30,
)

# ---------------------------------------------------------------------------
# Active Learning Loop
# ---------------------------------------------------------------------------
# Build the stage list explicitly and instantiate StagedActiveLearningHandler
# directly. Each stage consumes the previous stage's output; ``initial_input``
# seeds its *first* iteration, and ``combine=True`` accumulates a stage's
# outputs across iterations before the next stage consumes them.
#
# Chain (cyclic): sampler -> evaluator -> trainer -> (next iter) sampler.
#   - sampler seed = the starting committee (it needs a model to run MD).
#   - evaluator combines its outputs, so the trainer trains on the *whole*
#     accumulated DFT set, not just the latest batch.
#   - trainer seed = the starting DFT database; its outputs are ``.model``
#     committees that drive the next round of MD.
#
# To thin the candidate pool, insert a Stage("selector", SelectorActor(...))
# between sampler and evaluator; here the LammpsSampler already emits
# ``num_samples`` candidates, so no selector stage is used.
stages = [
    Stage(
        name="sampler",
        actor=lammps_sampler,
        combine=False,
        output_extension="extxyz",
        initial_input=model_files,
    ),
    Stage(
        name="evaluator",
        actor=vasp_evaluator,
        combine=True,
        output_extension="extxyz",
    ),
    Stage(
        name="trainer",
        actor=mace_trainer,
        combine=False,
        output_extension="model",
        initial_input=db_file,
    ),
]

al_handler = StagedActiveLearningHandler(
    stages=stages,
    calchome=calchome / "calchome",
    num_iterations=5,
    entry_stage="sampler",
    exit_stage="trainer",
)

al_handler.run()
