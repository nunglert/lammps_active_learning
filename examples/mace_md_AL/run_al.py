"""Active learning workflow: LAMMPS MD sampling with a MACE committee.

A Cu/Co/H/O surface active-learning loop with five actors:
  1. Sampling:    ``LammpsSampler`` (this package) runs LAMMPS MD driven by
                  the first committee model (compiled to the MLIAP
                  interface) and returns the resulting trajectory. No
                  scoring/selection.
  2. Annotation:  ``CommitteeUncertaintySelector`` (this package) scores
                  every frame's local force-uncertainty across the
                  committee and returns the whole trajectory annotated --
                  no reduction. Split out of the sampler; see
                  ``lammps_al.select.selector``'s module docstring.
  3. Selection:   ``PeakUncertaintySelector`` (this package, host-side, no
                  cluster job) picks the peak-uncertainty frame from each
                  of a few trajectory segments.
  4. Evaluation:  ``VASPEvaluator`` (remote_handler) runs DFT single-points
                  on the sampled structures for ground-truth
                  energies/forces.
  5. Training:    ``MaceModelTrainer`` (remote_handler) retrains the MACE
                  committee on the accumulated DFT database.

The loop seeds from an existing committee + DFT database and then repeats
  sampling -> annotation -> selection -> evaluation -> training -> sampling
  -> ...
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
from remote_handler.workflows import (
    InputSource,
    SelectorActor,
    Stage,
    StagedActiveLearningHandler,
)
from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.trainers import MaceModelTrainer

from lammps_al.run.sampler import LammpsSampler
from lammps_al.select.selector import (
    CommitteeUncertaintySelector,
    PeakUncertaintySelector,
)

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
    remote_lammps_executable_path=Path(
        "/home/e12218283/software/lammps/build-mliap/lmp"
    ),
    remote_savepath=Path("/home/e12218283/dump/remote_handler"),
    loop_time=30,
)

# ---------------------------------------------------------------------------
# 1b. Selector: CommitteeUncertaintySelector (annotate, no reduction)
# ---------------------------------------------------------------------------
# Split out of LammpsSampler -- see lammps_al.select.selector's module
# docstring. Its committee (model_files, via extra_inputs below) comes from
# "trainer", the same producer that already feeds the sampler stage's own
# primary ring input (initial_input=model_files at iteration 0). Returns the
# whole annotated trajectory -- peak-frame reduction is a separate,
# host-side step next (this campaign, unlike rough_cu_water_al, has no
# further spatial-selection stage, so it needs that reduction explicitly).
committee_selector = CommitteeUncertaintySelector(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config,
    cutoff_radius=4.0,
    remote_savepath=Path("/home/e12218283/dump/remote_handler"),
    loop_time=30,
)

# ---------------------------------------------------------------------------
# 1c. Selector: PeakUncertaintySelector (host-side peak-frame reduction)
# ---------------------------------------------------------------------------
# Pure numpy over already-annotated frames -- no cluster job, no committee.
# Wrapped in remote_handler's SelectorActor (the same adapter STO's
# RandomSelector uses) to slot in as a Stage.
peak_selector = SelectorActor(
    PeakUncertaintySelector(bin_count=2),
    output_filename="peak_samples.extxyz",
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
# Chain (cyclic): sampler -> uncertainty_select -> peak_select -> evaluator
# -> trainer -> (next iter) sampler.
#   - sampler seed = the starting committee (it needs a model to run MD).
#   - uncertainty_select scores the sampler's trajectory against the same
#     committee (pulled via extra_inputs from "trainer", the same producer
#     that seeds the sampler stage itself) and returns it fully annotated
#     -- no reduction. See lammps_al.select.selector's module docstring for
#     why annotation and reduction are two separate stages now (this used
#     to be folded into LammpsSampler alone).
#   - peak_select (host-side, no cluster) picks the peak-uncertainty frame
#     from each of ``bin_count`` segments -- this campaign hands whole
#     configs straight to DFT with no further spatial-selection stage, so
#     it needs that reduction explicitly (unlike rough_cu_water_al's
#     ROIExtractor, which consumes the full annotated trajectory directly).
#   - evaluator combines its outputs, so the trainer trains on the *whole*
#     accumulated DFT set, not just the latest batch.
#   - trainer seed = the starting DFT database; its outputs are ``.model``
#     committees that drive the next round of MD.
stages = [
    Stage(
        name="sampler",
        actor=lammps_sampler,
        combine=False,
        output_extension="extxyz",
        initial_input=model_files,
    ),
    Stage(
        name="uncertainty_select",
        actor=committee_selector,
        combine=False,
        output_extension="extxyz",
        extra_inputs={
            "model_files": InputSource(
                stage="trainer",
                accumulate=False,
                include_current=False,
                initial=model_files,
            )
        },
    ),
    Stage(
        name="peak_select",
        actor=peak_selector,
        combine=False,
        output_extension="extxyz",
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
