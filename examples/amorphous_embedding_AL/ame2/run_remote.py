"""Example -- amorphous-matrix embedding anneal on a bulk solid.

Demonstrates ``ROIExtractor`` + ``EmbeddingAnnealer``, the two peer actors of
the amorphous-matrix embedding pipeline: extract regions of interest (ROIs)
from an uncertainty-tagged MD trajectory and carve a box around each ROI
(``ROIExtractor``), then anneal the mobile shell around the frozen ROI with
a LAMMPS deck driven by a GRACE foundation-model potential
(``EmbeddingAnnealer``).

The system here is a bulk, fully periodic Si/SiO2 mixture with no free
surface, which is the simplest case this pipeline supports:
``topology_mode="bulk"`` skips surface-normal detection entirely (no OVITO
needed), the carved box is a plain axis-aligned cube, and a single anneal
stage covers every species since there is no substrate/solvent split to
stagger. See ``ame1/run_remote.py`` for the interface variant, where an
oriented box, a two-stage anneal and molecule-repair cleanup are all needed
because that system has a free surface and a distinct solvent phase.

This runs a single extract-then-anneal pass directly against the two
actors, not a full active-learning loop (see
``remote_handler.workflows.staged_active_learning.StagedActiveLearningHandler``
for wiring them -- plus an evaluator/trainer -- into a real campaign) --
useful for testing/tuning the pipeline in isolation first.
"""

from pathlib import Path

from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem

from lammps_al.embedding.annealer import EmbeddingAnnealer
from lammps_al.embedding.extractor import ROIExtractor

local_fs = LocalFileSystem()

# conda_source must point at your actual conda/micromamba install (a path
# ending in "miniconda3", containing "micromamba", or a "bin/activate"
# path) -- without it conda_env is silently ignored and the ambient
# `python` on PATH is used instead.
#
# Extraction and anneal are two separate remote stages and can each target a
# different cluster/env (e.g. a CPU node for extraction, a GPU node for the
# anneal) -- here both happen to run locally in the same env for simplicity.
extract_cfg = ClusterConfig(
    remote_machine="localhost",
    savepath="/tmp/embedding_test_sio2_extract",  # parent (/tmp) must exist
    partition=None,
    template="none",
    executor="direct",
    conda_source="/home/lerhard/micromamba",
    conda_env="remote_handler",  # bulk mode needs no OVITO extra
)

anneal_cfg = ClusterConfig(
    remote_machine="localhost",
    savepath="/tmp/embedding_test_sio2_anneal",  # parent (/tmp) must exist
    partition=None,
    template="none",
    executor="direct",
    conda_source="/home/lerhard/micromamba",
    conda_env="remote_handler",  # has `remote_handler` + LAMMPS on PATH
)

# Species/type order shared verbatim between the two actors: the anneal
# deck fixes LAMMPS type order to this list, so the extractor must write
# every carved box with the same ordering (via its `specorder` below).
species = ["Si", "O"]

extractor = ROIExtractor(
    remote_fs=local_fs,
    cluster_config=extract_cfg,
    # threshold/cluster_cutoff define the ROIs (regions of interest): atoms
    # with uncertainty >= threshold are seeds, and seeds within
    # cluster_cutoff of each other are merged into a single ROI. One
    # extracted/annealed box is produced per ROI. Both are dataset-specific:
    # tune threshold against this system's own uncertainty distribution
    # rather than reusing a value from a different model/system.
    threshold=0.8,  # min per-atom uncertainty to seed a ROI
    cluster_cutoff=3.0,  # (Angstrom) high-uncertainty atoms within this
    # distance of each other are grouped into one ROI
    # Carved-box half-widths around each ROI, in Angstrom. A bulk, isotropic
    # matrix has no preferred direction, so x/y/z are all equal here (unlike
    # a surface/interface box, which is typically wider in-plane and
    # asymmetric in z to leave headspace on the vacuum/solvent side).
    box_x=9.0,
    box_y=9.0,
    box_z_below=9.0,  # box_z_above defaults to box_z_below
    buffer=2.0,  # extra padding added to every carved-cell edge
    protect_radius=4.0,  # radius around the ROI that cleanup never
    # touches; pass the same value as EmbeddingAnnealer's anneal_cutoff
    # below
    specorder=species,  # must match EmbeddingAnnealer's `species` below
    # "bulk" is the default topology and the right choice for a homogeneous
    # matrix: no OVITO surface mesh, no rotation, no remove_species/
    # phase_species (those only matter for "surface"/"interface", where a
    # normal direction has to be found). The carved box keeps the founder's
    # own periodic orientation as-is.
    topology_mode="bulk",
    # Si-O is a continuous covalent network, not discrete molecules, and not
    # a fixed stoichiometric ratio (it's a genuine Si/SiO2 mixture) -- so
    # neither "molecule" nor "stoichiometry" cleanup fits. "none" keeps the
    # raw cut boundary; the frozen ROI plus the anneal's repulsive wall
    # handle the boundary instead.
    cleanup_mode="none",
    stub=False,
)

annealer = EmbeddingAnnealer(
    remote_fs=local_fs,
    cluster_config=anneal_cfg,
    # LAMMPS/potential setup for the anneal stage. GRACE is the easy
    # foundation-model route: it needs no special LAMMPS build (just the
    # `pair_style grace ...` already compiled into the `remote_handler` env's
    # `lmp`), and it ships its own foundation-model downloader. One-time
    # setup, in an env with `tensorpotential` (e.g. the `grace_dev` conda
    # env on this machine):
    #   grace_models list                  # see available foundation models
    #   grace_models download GRACE-2L-OAM # caches to ~/.cache/grace/...
    # GRACE-2L-OAM covers ~90 elements incl. Si/O (6 A cutoff, fitted on
    # OMat24 + fine-tuned on sAlex/MPTraj) -- GRACE-1L-OAM/GRACE-FS-OAM are
    # lighter/faster single-layer alternatives if 2L is too slow.
    lammps_path="/home/lerhard/micromamba/envs/remote_handler/bin/lmp",
    pair_style="grace pad_verbose padding 0.1",
    pair_coeff_path=Path.home() / ".cache/grace/GRACE-2L-OAM",
    species=species,  # LAMMPS type order -- matches extractor's specorder
    masses=[28.085, 15.999],  # aligned with species
    anneal_cutoff=4.0,  # frozen fixSphere radius -- matches extractor's
    # protect_radius above
    topology_mode="bulk",  # only used here to auto-decide z_wall_epsilon
    # below; matches the extractor's topology_mode
    # A single stage covering every species: everything outside the frozen
    # ROI moves together for the whole anneal. Multiple stages only make
    # sense when distinct phases need to equilibrate their own density
    # separately (e.g. a metal substrate vs. a solvent overlayer); a
    # homogeneous matrix like this one has no such split.
    anneal_stages=[
        {
            "species": ["Si", "O"],
            "temperatures": [300.0, 900.0, 300.0],  # ramp up then down
            "run_steps": [50000, 50000],
            "pressure_xy": 0.0,  # fully periodic in all 3 directions, so
            "pressure_z": 0.0,  # let the whole cell relax isotropically
        },
    ],
    # Repulsive wall keeping mobile atoms from drifting into (or through)
    # the frozen fixSphere core -- sits at anneal_cutoff - roi_repel_margin
    # (here 4.0 - 0.5 = 3.5 A). Without it, nothing but the pair potential's
    # own short-range repulsion resists a mobile atom approaching the rigid,
    # non-relaxing boundary.
    roi_repel_margin=0.5,
    roi_repel_epsilon=10.0,
    roi_repel_wall_cutoff=1.0,
    # z_wall_epsilon is left unset here: for topology_mode="bulk" it
    # defaults to 0.0 (disabled), since a fully periodic cell has no
    # vacuum/headspace for atoms to drift into. It only defaults on (10.0)
    # for "surface"/"interface" topologies, where that headspace exists.
    stub=False,
)

calcdir = Path("results/iteration_0")  # collected locally, next to this script
calcdir.mkdir(parents=True, exist_ok=True)

# Trajectory that seeds the ROIs; every frame's atoms.arrays must include a
# per-atom "uncertainty" array (e.g. force-committee stddev).
extractor.update(calcdir, Path("inputs/snapshot.xyz"))
extracted = extractor.get_selected_configs()

annealer.update(calcdir, extracted)
output_path = annealer.get_sampled_configs()
print(f"Annealed configurations written to {output_path}")
