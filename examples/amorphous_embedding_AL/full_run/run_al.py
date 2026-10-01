"""Active-learning workflow: rough Cu-water interfaces, ACE committee.

Forked from ``experiments/rough_cu_water_al`` (2026-09-25) with four
parameter changes, informed by that campaign's own results/dashboard:
sampler MD length 2000 -> 20000 steps (``config/templates/
lammps_ace.in.template``, ``lammps_sampler.dump_frequency`` scaled to
match), ``roi_extractor.max_num_structures`` 100 -> 10, ``vasp_evaluator.
num_parallel`` 1 -> 3, and the annealer's water stage (``embedding_
annealer.anneal_stages[1]["run_steps"]``) 25000 -> 50000 steps (real
per-box thermo from rough_cu_water_al's completed anneal showed this
stage starting from a genuine ~500-540K thermal transient at the stage
handoff, leaving little post-transient time at its own 25000-step budget
even though it did fully settle by the end -- see that setting's own
comment). Everything else (cluster assignment, INCAR settings, etc.) is
unchanged -- see each setting's own comment below for why the *original*
value was chosen; only the values noted here changed.

Skeleton for the campaign described in
``/cellar/nunglert/projects/remote_handler/lammps/PLAN.md``: reproduce
erhard2026's amorphous-matrix-embedding AL workflow (rough Cu(100/110/111)
surfaces + water, spatially-resolved committee uncertainty) starting from
the same seed database, but with an ACE committee instead of GRACE, and our
own iteration schedule/thresholds -- not a replication of that paper.

Six stages, cyclic (mirrors ``examples/mace_md_AL/run_al.py``'s shape, but
with extraction and annealing as separate ``Stage``s -- see
``lammps_al.embedding.{extractor,annealer}``'s module docstrings for why --
and with committee-uncertainty scoring split out of the sampler into its own
``Selector`` stage, mirroring that same "peer actor, not a sub-step" split --
see ``lammps_al.select.selector``'s module docstring). That stage only
*annotates* every frame with committee uncertainty; it does not reduce the
trajectory -- ``ROIExtractor`` needs the whole thing to search for regions
of interest, unlike ``mace_md_AL``, which adds a separate peak-frame
reduction stage after annotation since it has no spatial-selection step of
its own:

    sampler (LammpsSampler, backend="ace", MD only -- no scoring)
      -> uncertainty_select (CommitteeUncertaintySelector, annotate only)
      -> selector (ROIExtractor)
      -> annealer (EmbeddingAnnealer)
      -> evaluator (VASPEvaluator)
      -> trainer (AceModelTrainer)
      -> (next iteration) sampler ...

``entry_stage="trainer"``: no ready-made ACE committee exists for
``initial_database.xyz`` (erhard2026's own ready-made committee, ``iter10``,
is MACE-only), so iteration 0 trains the first committee from the seed
database before any sampling happens; iteration 1+ runs the full ring, each
using the previous iteration's freshly retrained committee.

STATUS: skeleton, gated by the single ``STUB`` switch below (every actor's
``stub=STUB``) -- ``STUB=True`` validates the ring mechanics (data flow
between stages, entry_stage bootstrapping) with no LAMMPS, no VASP, no
pacemaker actually invoked. Flip it to ``False`` to submit for real -- but
first read the "before flipping to a real run" list below; several of those
are not yet in place.

GPU status (2026-09-10 update, PLAN.md has the full story):
  - **Sampling: GPU, working.** LAMMPS's ``ML-PACE`` package genuinely
    supports Kokkos (``pace/kk``/``pace/extrapolation/kk``) -- an earlier
    claim in this campaign's history that it was CPU-only was wrong, and
    only checked the *fitting* library (pyace), a different codebase. Built
    from upstream LAMMPS (``build/build_lammps_pace_kokkos.sh``,
    Kokkos-CUDA + ML-PACE, Hopper/H100) after four real attempts (conda
    activation-hook bug, missing MPI, missing libjpeg -- see that script's
    inline history) -- confirmed via ``lmp -h`` listing both ``pace/kk``
    and ``pace/extrapolation/kk``. The sampler stage runs on
    ``musica_gpu_cluster`` (``--gres=gpu:1``) with Kokkos flags
    (``lammps_args``) restored.
  - **Training: GPU, via a dedicated env.** pacemaker's GPU path
    (``backend.evaluator="tensorpot"``) needs old-API symbols
    (``potentials.ace``, ``fit``, ``utils.utilities``, ...) that current
    PyPI ``tensorpotential`` (0.6.0/0.6.1, GRACE-focused) relocated/
    redesigned away from -- see build/TENSORPOT_ACE_INCOMPATIBILITY.md.
    Fixed by installing the OLD ``tensorpotential`` API
    (github.com/ICAMS/TensorPotential, pre-GRACE-rename) via a new,
    dedicated conda env, ``ace_gpu_train`` (musica, python 3.10 -- that old
    package needs ``Requires-Python: <3.11``), kept separate from
    ``lammps_al`` because [ace_tensorpot_gpu] and [grace] install the same
    top-level ``tensorpotential`` package name with incompatible APIs and
    cannot coexist. Its declared ``numpy<=1.26.4``/``pandas<=2.0`` pins are
    stale, not real incompatibilities (confirmed: all five symbols import
    cleanly against numpy 2.2.6/pandas 2.3.3) -- installed via
    ``pip install --no-deps`` in ``redistribute_lammps_al.sh`` to route
    around pip's resolver otherwise refusing the (disjoint, unsatisfiable)
    version ranges entirely. The sampler, annealer, and trainer all now run
    in this env (``config/templates/{musica,musica_gpu}.json``'s
    ``conda_env``). **Not yet verified with a real fit** -- only that the
    import chain resolves; see ``config/ace_config_gpu.yaml``'s backend
    comment and ``build/test_ace_gpu_training.py``.

Cluster assignment (2026-09-22 update -- moved off musica entirely):
sampling + committee-eval + training on datalab's GPU partition
(``ace_gpu_train`` env there -- see "GPU status" above, now stale on the
"musica" specifics but current on the GPU-need reasoning itself), extraction
+ annealing on lugus (plain ``lammps_al`` env, CPU partition -- neither
needs a GPU: extraction is geometry/OVITO work, and the annealer's carved
boxes are small, same reasoning as when it ran plain-CPU on musica), DFT on
dirona. ``lugus`` and ``dirona`` are, confirmed, the *same physical
cluster* -- ``~/.ssh/config``'s ``Host lugus`` entry hardcodes dirona's IP
(a historical alias, not a stale/wrong one), so they share one home
filesystem, one set of conda envs (``lammps_al``'s ovito-via-conda fix --
see pyproject.toml's ``[embedding]`` comment -- and the plain-CPU ML-PACE
binary from ``build/build_lammps_pace_cpu.sh`` both already cover both
aliases), and one SLURM ``normal`` partition; kept as two separate
``ClusterConfig``s purely to keep each stage's remote savepath tidy
(``lugus.json``'s ``.../train`` vs ``dirona.json``'s ``.../data``), not
because anything about the underlying machine differs. Loaded from real
``ClusterConfig`` JSONs (``config/templates/{lugus,datalab,dirona}.json``)
with SLURM submission (``executor`` defaults to ``"slurm"`` when unset in
the JSON). dirona's VASP module load lives in its ``environment.vasp``
entry (the preferred mechanism, resolved into the SLURM header's
``{{ slurm_header_add }}`` automatically; see
``remote_handler.workflows.dft_runner._resolve_environment_header``), not
hardcoded in ``dirona.sh.j2``. ``musica`` is no longer used by this campaign
-- its ``ClusterConfig``/template files stay on disk untouched, but are no
longer constructed here (mirrors how ``lugus`` was previously the one left
untouched, before this move). `EmbeddingAnnealer.pair_coeff_path` is wired
the same way `CommitteeUncertaintySelector.model_files` is -- via the
"annealer" ``Stage``'s ``extra_inputs``, pulling "trainer"'s output --
rather than a fixed constructor value, so it always anneals with the most
recently trained committee (its first member; annealing drives one
``pair_style``/``pair_coeff``, not a committee) with no manual wiring
needed once the trainer has produced real output.
"""

import os

# Ensure a writable TMPDIR (Slurm nodes may restrict /tmp).
_tmpdir = os.path.expanduser("~/tmp")
os.makedirs(_tmpdir, exist_ok=True)
os.environ["TMPDIR"] = _tmpdir

from pathlib import Path

from remote_handler import configure_logging
from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.staged_active_learning import (
    InputSource,
    Stage,
    StagedActiveLearningHandler,
)
from remote_handler.workflows.trainers import AceModelTrainer

from lammps_al.embedding.annealer import EmbeddingAnnealer
from lammps_al.embedding.extractor import ROIExtractor
from lammps_al.run.sampler import LammpsSampler
from lammps_al.select.selector import CommitteeUncertaintySelector

configure_logging()  # timestamped logs (level=INFO -> stdout)

# Global switch: True dry-runs every stage (no cluster, no LAMMPS/VASP/
# pacemaker actually invoked -- see module docstring); False submits real
# jobs to datalab/lugus/dirona for every stage. One switch, not per-actor,
# so a real run can't accidentally leave one stage stubbed.
STUB = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
calchome = Path(__file__).parent.resolve()
config_dir = calchome / "config"
data_dir = Path("/cellar/nunglert/projects/remote_handler/lammps/data")

# SLURM header templates (lugus.sh.j2/datalab.sh.j2/dirona.sh.j2) are
# referenced by bare filename in the cluster JSONs below; this is where
# ClusterConfig resolves them from.
os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = str(config_dir / "templates")

# LAMMPS type order shared across every stage that touches a LAMMPS box:
# ROIExtractor.specorder, EmbeddingAnnealer.species, and LammpsSampler's
# element_mapping must all agree on this ordering.
species = ["Cu", "H", "O"]
masses = [63.546, 1.008, 15.999]

# ---------------------------------------------------------------------------
# Clusters: sampling + committee-eval + training on datalab's GPU partition
# (ace_gpu_train env there), extraction + annealing on lugus (plain
# lammps_al env, CPU), DFT on dirona. Loaded from the real ClusterConfig
# JSONs below -- executor defaults to "slurm" (unset in all JSONs), so
# these submit real SLURM jobs once STUB=False above.
# ---------------------------------------------------------------------------
# musica no longer used by this campaign (extraction moved to lugus,
# annealing moved to lugus, sampling/committee-eval/training moved to
# datalab -- see module docstring's "GPU status"/"Cluster assignment"
# sections) -- its ClusterConfig/template files stay on disk untouched, but
# are no longer constructed here.
# musica_fs = RemoteFileSystem(
#     remote_alias="musica", local_temp_path=_tmpdir
# )
# musica_cluster = ClusterConfig.from_file(
#     str(config_dir / "templates" / "musica.json")
# )

# GPU variant of the musica cluster (zen4_0768_h100x4, --gres=gpu:1 -- see
# musica_gpu.json/musica_gpu.sh.j2), shared by the sampler and trainer
# stages: LAMMPS was built with Kokkos-CUDA + ML-PACE specifically for the
# sampler (build/build_lammps_pace_kokkos.sh, succeeded 2026-09-09 --
# pace/kk and pace/extrapolation/kk both confirmed present via `lmp -h`),
# and pacemaker's tensorpot backend needs a GPU for the trainer (see module
# docstring). The annealer (geometry/small-box work, no real GPU benefit)
# stays on the plain CPU partition above.
musica_gpu_fs = RemoteFileSystem(
    remote_alias="musica", local_temp_path=_tmpdir
)
musica_gpu_cluster = ClusterConfig.from_file(
    str(config_dir / "templates" / "musica_gpu.json")
)

datalab_gpu_fs = RemoteFileSystem(
    remote_alias="datalab", local_temp_path=_tmpdir
)
datalab_gpu_cluster = ClusterConfig.from_file(
    str(config_dir / "templates" / "datalab.json")
)

# No longer used -- EmbeddingAnnealer moved to lugus (see stage 3 below),
# so it no longer needs to piggyback on ace_gpu_train's pace-kokkos binary
# just to get a CPU partition on the same host as the sampler.
# musica_ace_gpu_train_cpu_fs = RemoteFileSystem(
#     remote_alias="musica", local_temp_path=_tmpdir
# )
# musica_ace_gpu_train_cpu_cluster = ClusterConfig.from_file(
#     str(config_dir / "templates" / "musica_ace_gpu_train_cpu.json")
# )

dirona_fs = RemoteFileSystem(remote_alias="dirona", local_temp_path=_tmpdir)
dirona_cluster = ClusterConfig.from_file(
    str(config_dir / "templates" / "dirona.json")
)
# "perform" partition (c013/c014, up to 500GB/128 real cores -- vs. "normal"'s
# 250GB/28 real cores): the VASP evaluator's boxes (~500-600 atoms, NBANDS in
# the thousands) need this scale to finish an SCF step in a reasonable time at
# all -- see vasp_evaluator's own comment. Temporary, while waiting on VASP
# license access on musica (modern Zen4 hardware, no 2FA blocker, but access
# is currently license-gated) -- musica is the better long-term home for this
# stage once that's granted; this keeps the DFT stage actually producing data
# in the meantime, on infrastructure already fully wired up here.
dirona_perform_cluster = ClusterConfig.from_file(
    str(config_dir / "templates" / "dirona_perform.json")
)

# Extraction + annealing (see stages 2/3 below): plain lammps_al env, CPU
# partition -- neither needs a GPU (extraction is geometry/OVITO work,
# annealing's boxes are small), so both share this one cluster/env instead
# of contending for musica's ace_gpu_train GPU-partition twin. Needs the
# [embedding] extra's ovito installed via conda, not pip, on lugus too --
# see pyproject.toml's [embedding] comment; confirmed necessary on dirona,
# not yet verified on lugus specifically.
lugus_fs = RemoteFileSystem(remote_alias="lugus", local_temp_path=_tmpdir)
lugus_cluster = ClusterConfig.from_file(
    str(config_dir / "templates" / "lugus.json")
)

# ---------------------------------------------------------------------------
# 1. Sampler: LammpsSampler (ACE committee-uncertainty MD)
# ---------------------------------------------------------------------------
lammps_sampler = LammpsSampler(
    remote_fs=musica_gpu_fs,
    cluster_config=musica_gpu_cluster,
    template_path=config_dir / "templates" / "lammps_ace.in.template",
    # Frame 0 of RoughSurfaces/RoughSurfaces/Cu_Water_equilibrated.xyz (46
    # distinct rough surfaces) -- the smallest by atom count (206941;
    # picked for that reason alone, not for roughness -- see
    # inputs/extract_surface.py), converted to lammps-data with velocities
    # preserved from the equilibrated trajectory's momenta.
    data_path=calchome / "inputs" / "surface_0.data",
    element_mapping={1: "Cu", 2: "H", 3: "O"},
    # Steps between dumped frames (template's "run 20000" / 4 uncertainty
    # bins downstream -> 100 total frames, 25/bin -- plenty of resolution to
    # find each bin's peak-uncertainty frame). Scaled up 10x from
    # rough_cu_water_al's dump_frequency=20 to match this run's 10x longer
    # "run" in the template, so frame count -- and downstream committee-
    # scoring cost -- stays the same even though simulated time didn't.
    # Previously (rough_cu_water_al, pre-scaling) collided with a different,
    # pre-existing field name (sampling_frequency) and got silently dropped
    # -- confirmed as the cause of a >20GB dump file in ~10 minutes on this
    # 206941-atom system, 2026-09-11. See LammpsSampler.dump_frequency's
    # docstring.
    dump_frequency=200,
    backend="ace",
    # "pace product" (not "pace/kk"): the template's pair_style line is
    # rendered verbatim from this string, and LAMMPS's `-sf kk` suffix flag
    # (below) auto-upgrades every kk-eligible style to its /kk variant at
    # runtime -- the standard, documented way to write a Kokkos-portable
    # input script rather than hardcoding the accelerated style name. The
    # "product" keyword is NOT optional here: pair_pace_kokkos.cpp requires
    # it explicitly ("Must use 'product' algorithm with pair pace/kk on the
    # GPU", confirmed via a real failed job, 2026-09-11) -- the default
    # evaluator (recursive, what a bare "pace" renders) only runs on CPU.
    # Confirmed in pair_pace.cpp's settings() parsing: "pace [recursive|
    # product] [chunksize N]". The plain-CPU annealer keeps the default
    # (recursive, its own "pace" literal below) since it never runs /kk.
    lammps_pair_style="pace product",
    # KOKKOS-on-GPU flags -- restored now that a real Kokkos-CUDA + ML-PACE
    # LAMMPS exists (build/build_lammps_pace_kokkos.sh, succeeded
    # 2026-09-09). Same shape as the class default (written for the MACE
    # mliap route) but "g 1" here maps to datalab_gpu's --gres=gpu:1.
    lammps_args="-k on g 1 -sf kk -pk kokkos newton on neigh half",
    remote_lammps_executable_path=(
        # datalab's own Kokkos-CUDA + ML-PACE build (AMPERE80), from
        # rough_cu_water_al/build/build_lammps_pace_kokkos_datalab.sh.
        "/data/fs201236/nu90319/software/lammps-pace-kokkos/src/build/lmp"
    ),
    stub=STUB,
)

# ---------------------------------------------------------------------------
# 1b. Selector: CommitteeUncertaintySelector (annotate, no reduction)
# ---------------------------------------------------------------------------
# Split out of LammpsSampler -- see lammps_al.select.selector's module
# docstring. Same GPU env as the sampler (ACE committee scoring is the same
# tensorpot/pyace stack). Its committee (`model_files`, the extra_inputs
# wiring below) comes from the "trainer" stage, exactly the same producer
# LammpsSampler's own primary ring input already comes from. Deliberately no
# peak-frame reduction here: ROIExtractor (next stage) needs the *whole*
# annotated trajectory to search for regions of interest, not a pre-picked
# handful of frames.
committee_selector = CommitteeUncertaintySelector(
    remote_fs=datalab_gpu_fs,
    cluster_config=datalab_gpu_cluster,
    cutoff_radius=4.0,  # matches the paper's aggregation radius
    backend="ace",
    stub=STUB,
)

# ---------------------------------------------------------------------------
# 2. Selector: ROIExtractor (find + carve ROIs from the tagged trajectory)
# ---------------------------------------------------------------------------
# Moved to lugus (plain lammps_al env, CPU partition -- no GPU need, no
# gres wasted; shares lugus_cluster with EmbeddingAnnealer below). Needs
# ovito (topology_mode="interface"'s surface-normal detection); ovito's
# PyPI wheel needs a working system OpenGL stack and fails to import on a
# headless node missing it (confirmed on dirona: `ImportError:
# libOpenGL.so.0`) -- fixed by installing ovito via conda instead of pip
# (see pyproject.toml's [embedding] comment). "lugus" and "dirona" are the
# same physical cluster/conda envs (see EmbeddingAnnealer's comment below),
# so that fix already covers this stage too -- no separate verification
# needed. Previously on musica (ovito confirmed working there); not
# ace_gpu_train (that env has no ovito installed at all -- only
# [ace,ace_tensorpot_gpu], no [embedding]).
roi_extractor = ROIExtractor(
    remote_fs=lugus_fs,
    cluster_config=lugus_cluster,
    threshold=0.03,  # paper's iteration-1 scale (0.02-0.03 eV/A); recalibrate
    cluster_cutoff=3.0,
    box_x=9.0,
    box_y=9.0,
    box_z_below=10.0,
    box_z_above=12.0,  # asymmetric: solvent headspace above the interface
    buffer=2.0,
    protect_radius=4.0,  # must match EmbeddingAnnealer.anneal_cutoff below
    specorder=species,
    topology_mode="interface",
    remove_species=["O", "H"],
    phase_species=["Cu"],
    # threshold=0.03 above is far too low for this system's real
    # aggregated-uncertainty scale (confirmed: the *minimum* over an entire
    # 206941-atom frame was 0.0966, i.e. every atom exceeds it), collapsing
    # find_rois' clustering to ~one giant, mostly-bulk-water cluster per
    # frame -- 98/101 extracted boxes had zero Cu, since water's
    # uncertainty tail runs higher than Cu's. This filter is a mitigation
    # (keeps ROIs near Cu, where EmbeddingAnnealer's frozen-core anneal
    # actually needs them), NOT a fix for that miscalibration -- threshold
    # itself still needs recalibrating separately (2026-09-22 investigation).
    # 6.0 (< box_z_below=10.0) is a starting guess: close enough that a
    # 10A-reach box should reliably still capture Cu even after the
    # normal-rotation in carve_box, not yet tuned against real data.
    phase_cutoff=6.0,
    # Global cap, ranked by uncertainty (not frame order) -- 913 ROIs from
    # a single 101-frame trajectory (2026-09-22, this same threshold/
    # phase_cutoff combo: the rough, multi-island interface fragments the
    # phase-filtered candidate pool into many small disconnected clusters
    # per frame instead of the one giant blob threshold=0.03 alone used to
    # collapse everything into) would otherwise mean 913 downstream anneal
    # jobs from one iteration. Lowered from rough_cu_water_al's 100 to 10
    # here -- annealing + DFT are the expensive stages (each anneal box is
    # a real MD run, each DFT box is the multi-hour-per-box VASP job this
    # campaign spent a long time getting to actually finish at all -- see
    # vasp_evaluator below), so this run trades ROI count for being able to
    # complete full AL iterations in a reasonable wall-clock time; revisit
    # once threshold itself is recalibrated (see above).
    max_num_structures=10,
    cleanup_mode="molecule",
    solvent="H2O",
    normal_patch_radius=5.0,
    stub=STUB,
)

# ---------------------------------------------------------------------------
# 3. Annealer: EmbeddingAnnealer (two-stage frozen-core anneal per box)
# ---------------------------------------------------------------------------
embedding_annealer = EmbeddingAnnealer(
    remote_fs=lugus_fs,
    cluster_config=lugus_cluster,
    # Moved to lugus, alongside ROIExtractor above -- the carved boxes are
    # small, so this stage never needed a GPU (it ran plain-CPU even when
    # it was on musica's own ace_gpu_train env, just reusing that env's
    # pace-kokkos binary because it happened to be there). "lugus" and
    # "dirona" are the same physical cluster (confirmed: both SSH aliases
    # resolve to the same host/home filesystem/conda envs -- see
    # ~/.ssh/config's "Host lugus" entry, hardcoded to dirona's IP), so
    # this reuses the plain-CPU ML-PACE binary already built there by
    # build/build_lammps_pace_cpu.sh -- no new build needed. An MPI-enabled
    # sibling build also exists (build/build_lammps_pace_cpu_mpi.sh,
    # lammps-pace-cpu-mpi/) for EmbeddingAnnealer's n_processors > 1
    # option, but real multi-rank throughput on this campaign's box sizes
    # (~500-600 atoms) isn't yet confirmed on this cluster -- several
    # attempted benchmarks landed on contended or unexpectedly stalled
    # nodes without producing a trustworthy number. Staying on n_processors
    # default (1) and this plain binary until that's actually verified.
    lammps_path=("/home/nunglert/software/lammps-pace-cpu/src/build/lmp"),
    pair_style="pace",
    # No pair_coeff_path here -- wired dynamically each iteration via the
    # Stage's extra_inputs below (same "trainer" producer as
    # uncertainty_select's model_files), not a fixed constructor value.
    species=species,
    masses=masses,
    anneal_cutoff=4.0,  # matches ROIExtractor.protect_radius above
    topology_mode="interface",
    anneal_stages=[
        {
            "species": ["Cu"],
            "temperatures": [300.0, 600.0, 300.0],
            "run_steps": [25000, 50000],
            "pressure_xy": 1.0,
        },
        {
            "species": ["H", "O"],
            "temperatures": [300.0, 300.0],
            # 50000, not rough_cu_water_al's 25000: checked real per-box
            # thermo from that campaign's completed anneal (dashboard's
            # annealer_thermo panel data) -- this stage starts from a real
            # ~500-540K thermal spike (inherited from stage 0's hot Cu via
            # nonbonded coupling while H/O sat frozen) despite its own
            # 300K flat target, and spends a good chunk of its step budget
            # just cooling back down before any genuine equilibration-at-
            # -target happens. It did fully settle by the end of 25000
            # steps (energy fluctuation dropped to ~0), but that leaves
            # little genuine post-transient time; doubling gives real
            # margin for that.
            "run_steps": [50000],
            "pressure_z": 1.0,
        },
    ],
    roi_repel_margin=0.5,
    roi_repel_epsilon=10.0,
    roi_repel_wall_cutoff=1.0,
    z_wall_margin=1.0,
    z_wall_cutoff=1.0,
    # One SLURM array job instead of one job per box -- extraction can
    # produce far more ROIs than a one-job-per-box submission can
    # reasonably queue, and the old per-box path was fully serial,
    # submitting and *waiting on* each one before the next was even
    # queued. Each anneal box stays single-core (n_processors default) for
    # now -- see lammps_path's comment above on why multi-rank isn't yet
    # verified as a real win here -- so num_parallel (more boxes running
    # concurrently, not more cores per box) is the throughput lever,
    # bounded by lugus' shared capacity (DFT jobs from the group routinely
    # occupy a large share of its cores); revisit if it's starving other
    # users or leaving the queue underfilled.
    array_job=True,
    num_parallel=20,
    stub=STUB,
)

# ---------------------------------------------------------------------------
# 2b. Evaluator: VASPEvaluator (RPBE + D3(zero-damping, all atoms))
# ---------------------------------------------------------------------------
# INCAR settings taken from the exact `calculator_parameters` recorded on
# Database.zip's frames (erhard2026's own labeling convention: encut,
# ediff, ismear/sigma, setups). k-points use KSPACING (a target reciprocal-
# space density, Å⁻¹) instead of a fixed `kpts` mesh -- this is almost
# certainly the actual mechanism behind the paper's "k-point density...
# scaled to a similar density" for their differently-sized cells (Section
# 2.2): VASP derives N_i = max(1, |b_i|/KSPACING) per cell automatically,
# so one deck-wide value keeps every differently-shaped/-sized carved box
# at consistent density with no per-cell arithmetic (and naturally relaxes
# toward Gamma-only for our large embedding boxes, ~20-30 A across).
# KSPACING_ESTIMATE below reproduces their stated "11x11x11 for a one-atom
# fcc-Cu primitive cell": |b_prim| = 2*pi*sqrt(3)/a with a=3.615 A (Cu's
# conventional lattice constant) -> |b_prim| ~ 3.01 A^-1 -> KSPACING =
# |b_prim|/11 ~ 0.274 A^-1. This is a reconstruction (the paper doesn't
# state KSPACING explicitly, and RPBE's relaxed lattice constant may not be
# exactly 3.615 A), not a verified reproduction -- check by confirming it
# yields an 11x11x11-equivalent mesh for the bulk cell before relying on it.
KSPACING_ESTIMATE = 0.274

vasp_evaluator = VASPEvaluator(
    remote_fs=dirona_fs,
    cluster_config=dirona_perform_cluster,
    # Requested as 3 (run2 vs. rough_cu_water_al's 1) -- but n_processors=128
    # (below) requests --exclusive on the "perform" partition's one
    # 128-real-core node (c013; the other, c014, only has 64 and can never
    # win this request), so at most 1 of these 3 array-job slots can
    # actually be RUNNING at once; the other 2 just sit queued behind it
    # rather than genuinely running concurrently. Left at 3 as asked --
    # harmless (SLURM just queues the rest), but if the intent was more
    # real throughput, that needs a smaller n_processors (so >1 job fits on
    # a node at once) rather than a bigger num_parallel, and n_processors=128
    # was itself a deliberate choice (see incar_kwargs's own comment: this
    # box size's NBANDS needs that scale to finish an SCF step at all).
    num_parallel=3,
    n_processors=128,
    incar_kwargs={
        "xc": "RPBE",
        "encut": 850,
        "ediff": 1e-5,
        "ismear": 0,
        "sigma": 0.05,
        "ivdw": 11,  # DFT-D3, zero-damping, applied to all atoms
        "gamma": True,
        "kspacing": KSPACING_ESTIMATE,
        "lwave": False,
        "lcharg": False,
        # LREAL=Auto (not False/reciprocal): VASP's own advice for a cell
        # this large, and -- confirmed empirically this session -- the
        # difference between a single SCF step not finishing within 90
        # minutes (LREAL=False, default ALGO) and 10 steps finishing in
        # ~1.5h (LREAL=Auto + ALGO=Fast + NCORE=64 + KPAR=1 below, on
        # perform's 128-core node). NELECT/NBANDS for these ~500-600 atom
        # boxes run into the thousands, so the default algorithm's
        # per-iteration subspace diagonalization dominates completely --
        # this is what actually fixes it, not core count alone (8 vs 28
        # ranks made no measurable difference under the default ALGO).
        "lreal": "Auto",
        "algo": "Fast",
        "ncore": 64,  # cores-per-socket on c013 -- VASP's own recommendation
        "kpar": 1,
        "prec": "Accurate",
        "setups": {"base": "minimal", "H": "_h", "O": "_h"},
    },
    # remote_savepath: AseCalculatorEvaluator otherwise defaults this to the
    # literal "/tmp" (-> "/tmp/dft"), unlike every other actor here (which
    # falls back to the cluster's own configured savepath). On this cluster
    # /tmp is per-node local disk, not shared -- the array job's tasks land
    # on compute nodes that can't see whatever dirona (the submission host)
    # wrote there, so every DFT box failed instantly with the payload script
    # itself missing. Point this at the same shared, home-directory savepath
    # every other stage already uses.
    remote_savepath=dirona_perform_cluster.savepath,
    required_fraction=1.0,
    stub=STUB,
)

# ---------------------------------------------------------------------------
# 4. Trainer: AceModelTrainer (pacemaker ensemble)
# ---------------------------------------------------------------------------
ace_trainer = AceModelTrainer(
    remote_fs=datalab_gpu_fs,
    cluster_config=datalab_gpu_cluster,
    # Our own copy of erhard2026's ACE_7_300/input_1.yaml pacemaker sweep
    # config (cutoff 7 A, 300 functions/element) with the stale
    # "test_filename" key removed (see config/ace_config.yaml's own
    # comment -- it crashed the first real run otherwise), and
    # backend.evaluator switched to tensorpot for GPU fitting (see
    # ace_config_gpu.yaml's own backend comment -- NOT yet verified with a
    # real fit, only that the import chain resolves; see
    # build/test_ace_gpu_training.py). data/seed are injected per job by
    # remote_handler, so this only supplies potential/fit/backend
    # hyperparameters -- see ace_utils.py.
    config_file=config_dir / "ace_config_gpu.yaml",
    num_models=5,
    stub=STUB,
)

# ---------------------------------------------------------------------------
# Active Learning Loop
# ---------------------------------------------------------------------------
# entry_stage="trainer": iteration 0 only trains the seed committee from
# initial_database.xyz (no sampler seed needed -- the ring's first stage
# each iteration reads the *previous* iteration's last-stage output, so
# iteration 1's sampler naturally consumes iteration 0's freshly trained
# committee). See this module's docstring.
stages = [
    Stage(
        name="sampler",
        actor=lammps_sampler,
        combine=False,
        output_extension="extxyz",
    ),
    Stage(
        name="uncertainty_select",
        actor=committee_selector,
        combine=False,
        output_extension="extxyz",
        extra_inputs={
            "model_files": InputSource(
                stage="trainer", accumulate=False, include_current=False
            )
        },
    ),
    Stage(
        name="selector",
        actor=roi_extractor,
        combine=False,
        # ROIExtractor's output is a *directory* (manifest.json + carved
        # boxes), not a single file -- the handler copies it as a whole
        # (shutil.copytree) regardless of this label.
        output_extension="dir",
    ),
    Stage(
        name="annealer",
        actor=embedding_annealer,
        combine=False,
        output_extension="extxyz",
        extra_inputs={
            "pair_coeff_path": InputSource(
                stage="trainer", accumulate=False, include_current=False
            )
        },
    ),
    Stage(
        name="evaluator",
        actor=vasp_evaluator,
        combine=True,  # trainer sees the whole accumulated DFT set
        output_extension="extxyz",
    ),
    Stage(
        name="trainer",
        actor=ace_trainer,
        combine=False,
        output_extension="yaml",
        initial_input=data_dir / "initial_database" / "initial_database.xyz",
    ),
]

al_handler = StagedActiveLearningHandler(
    stages=stages,
    calchome=calchome / "calchome",
    num_iterations=5,  # iteration 0 = bootstrap train; iteration 1 = 1 full cycle
    entry_stage="trainer",
)

al_handler.run()
