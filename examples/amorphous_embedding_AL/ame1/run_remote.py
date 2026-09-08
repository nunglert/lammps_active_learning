"""Example -- amorphous-matrix embedding anneal on a metal/solvent interface.

Demonstrates ``ROIExtractor`` + ``EmbeddingAnnealer``, the two peer actors of
the amorphous-matrix embedding pipeline: extract regions of interest (ROIs)
from an uncertainty-tagged MD trajectory, carve an oriented box around each
ROI, repair the cut boundary (``ROIExtractor``), then anneal the mobile
shell around the frozen ROI with a multi-stage LAMMPS deck driven by a GRACE
foundation-model potential (``EmbeddingAnnealer``).

The system here is a Cu slab with a water overlayer, so the carved box is
oriented to the copper surface normal (``topology_mode="interface"``) and the
anneal is split into two stages so the metal and the solvent each equilibrate
their own density independently. See ``ame2/run_remote.py`` for the same
pipeline on a homogeneous bulk solid, where several of these interface-only
pieces (surface-normal detection, an oriented/asymmetric box, molecule-repair
cleanup, the two-stage split) fall away.

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
extract_cfg = ClusterConfig(
    remote_machine="localhost",
    savepath="/tmp/embedding_test/extract",
    partition=None,
    template="none",
    executor="direct",
    conda_source="/home/lerhard/micromamba",
    conda_env="remote_handler",  # has `remote_handler[embedding]` (ovito) installed
)

anneal_cfg = ClusterConfig(
    remote_machine="localhost",
    savepath="/tmp/embedding_test/anneal",
    partition=None,
    template="none",
    executor="direct",
    conda_source="/home/lerhard/micromamba",
    conda_env="remote_handler",  # has `remote_handler` + LAMMPS on PATH
)

# Species/type order shared verbatim between the two actors: the anneal
# deck fixes LAMMPS type order to this list, so the extractor must write
# every carved box with the same ordering (via its `specorder` below).
species = ["Cu", "H", "O"]

extractor = ROIExtractor(
    remote_fs=local_fs,
    cluster_config=extract_cfg,
    # threshold/cluster_cutoff define the ROIs (regions of interest): atoms
    # with uncertainty >= threshold are seeds, and seeds within
    # cluster_cutoff of each other are merged into a single ROI. One
    # extracted/annealed box is produced per ROI.
    threshold=0.018,  # min per-atom uncertainty to seed a ROI
    cluster_cutoff=3.0,  # (Angstrom) high-uncertainty atoms within this
    # distance of each other are grouped into one ROI
    # Carved-box half-widths around each ROI, in Angstrom.
    box_x=9.0,  # half-width along x
    box_y=9.0,  # half-width along y
    box_z_below=10.0,  # half-width below the ROI along z
    box_z_above=12.0,  # half-width above the ROI along z (here bigger
    # than box_z_below to leave solvent headspace)
    buffer=2.0,  # extra padding added to every carved-cell edge
    protect_radius=4.0,  # radius around the ROI that cleanup never
    # touches; pass the same value as EmbeddingAnnealer's anneal_cutoff
    # below and as the default for normal_patch_radius
    specorder=species,  # must match EmbeddingAnnealer's `species` below
    # How the carved box is oriented before cutting. "interface" builds an
    # OVITO surface mesh (on a copy with remove_species stripped) to find
    # the local surface normal and rotates the box so z points along it.
    topology_mode="interface",
    remove_species=["O", "H"],  # stripped before meshing, so the mesh
    # tracks only the Cu surface
    phase_species=["Cu"],  # used only by the fallback normal-
    # direction heuristic (COM-offset method),
    # not by the primary OVITO mesh normal
    # Broken molecules at the cut boundary: "molecule" reforms them using
    # a solvent template built from the ase.build.molecule g2 label below.
    cleanup_mode="molecule",
    solvent="H2O",
    # Radius (Angstrom) over which nearby surface-mesh facet normals are
    # averaged when picking the interface normal, instead of just using
    # the single nearest facet. Left at None this defaults to
    # protect_radius (4.0 here). Widening it smooths out local roughness
    # (atomic steps/pits) right next to the ROI that would otherwise
    # dominate the single-nearest-facet normal.
    normal_patch_radius=5.0,
    stub=False,
)

annealer = EmbeddingAnnealer(
    remote_fs=local_fs,
    cluster_config=anneal_cfg,
    # LAMMPS/potential setup for the anneal stage.
    lammps_path="/home/lerhard/micromamba/envs/remote_handler/bin/lmp",
    pair_style="grace pad_verbose padding 0.1",
    pair_coeff_path=Path("inputs/final_model"),  # GRACE model dir
    species=species,  # LAMMPS type order -- matches extractor's specorder
    masses=[63.546, 1.008, 15.999],  # aligned with species
    anneal_cutoff=4.0,  # frozen fixSphere radius -- matches extractor's
    # protect_radius above
    topology_mode="interface",  # only used here to auto-decide
    # z_wall_epsilon below; matches the extractor's topology_mode
    anneal_stages=[
        {
            "species": ["Cu"],
            "temperatures": [300.0, 600.0, 300.0],  # ramp up then down
            "run_steps": [25000, 50000],
            "pressure_xy": 1.0,  # in-plane relaxation
        },
        {
            "species": ["H", "O"],
            "temperatures": [300.0, 300.0],  # flat equilibration
            "run_steps": [25000],
            "pressure_z": 1.0,  # density equilibration
        },
    ],
    # Repulsive wall keeping mobile atoms from drifting into (or through)
    # the frozen fixSphere core -- sits at anneal_cutoff - roi_repel_margin
    # (here 4.0 - 0.5 = 3.5 A). Without it, nothing but the pair potential's
    # own short-range repulsion resists a mobile atom approaching the rigid,
    # non-relaxing boundary. Deliberately a single, deck-wide setting shared
    # by every stage's mobile atoms (unlike z_wall_margin/cutoff below,
    # which are per-stage): this protects the frozen core itself, so it
    # should stay uniformly active regardless of which species is currently
    # mobile -- not disabled for water on the assumption that dense copper
    # already excludes it, since that's a bet on the potential's own
    # short-range repulsion being reliable enough this close to a rigid,
    # non-relaxing boundary, which isn't a risk worth taking here. Values
    # below match the group's original deck.
    roi_repel_margin=0.5,
    roi_repel_epsilon=10.0,
    roi_repel_wall_cutoff=1.0,
    # Per-stage z-confinement wall, keeping each stage's mobile atoms from
    # drifting into the vacuum/solvent headspace above the interface (there
    # is headspace here: box_z_above=12.0 > box_z_below=10.0 on the
    # extractor). Only applied to a stage when its pressure_z is unset --
    # the Cu stage above has no pressure_z, so it gets this wall; the H/O
    # stage already barostats z via its own pressure_z, so a static wall
    # would fight that box rescaling and is skipped for it automatically.
    # z_wall_epsilon is left unset: it defaults to 10.0 here because
    # topology_mode="interface" (a free surface implies possible
    # headspace); it would auto-default to 0.0/disabled for
    # topology_mode="bulk", where there's no vacuum to drift into.
    #
    # These two are the deck-wide default, used by any stage that doesn't
    # set its own z_wall_margin/z_wall_epsilon/z_wall_cutoff keys inside its
    # anneal_stages dict above. Only the Cu stage actually uses them here
    # (the H/O stage is skipped via pressure_z), so there's nothing to
    # differentiate yet -- but e.g. a stiffer wall for a rigid metal lattice
    # than for a mobile solvent stage would be a per-stage override, not a
    # reason to change these deck-wide numbers.
    z_wall_margin=1.0,
    z_wall_cutoff=1.0,
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
