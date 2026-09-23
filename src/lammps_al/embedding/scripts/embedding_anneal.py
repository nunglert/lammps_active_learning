"""Amorphous matrix embedding -- multi-stage LAMMPS anneal entry point.

Entry point for remote execution via ``SingleScriptTask`` *or*
``ArrayScriptTask`` (see ``EmbeddingAnnealer``'s ``array_job`` option -- a
campaign can produce far more ROIs than a one-job-per-box submission can
reasonably queue). ``SingleScriptTask`` passes the input JSON file itself as
``sys.argv[1]``; ``ArrayScriptTask`` passes the per-task calcdir instead (the
input JSON lives inside it as a fixed ``input_anneal.json`` name) -- both are
accepted here. Renders a multi-stage LAMMPS deck around a frozen
``fixSphere`` centered on the region of interest (ROI), runs ``lmp``, and
writes the annealed structure.

Self-contained: does not import from ``lammps_sampling.py`` (whose deck
rendering/``lmp``-invocation helpers are tied to that script's own MACE
committee sampling flow and are left untouched) -- the small deck-render/
``lmp``-invocation helper below is purpose-built for this script instead.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import ase.io
import jinja2
from ase.data import atomic_numbers

from lammps_al.embedding.inputs import EmbeddingAnnealInput

_ARG = Path(sys.argv[1])
# ArrayScriptTask's body_array.sh.j2 invokes `python <script> <calcdir>`
# (one calcdir per array element, no fixed input-file name of its own
# choosing) -- unlike SingleScriptTask, which passes the input file
# directly. Accept either: a directory means "look for the fixed
# input_anneal.json name EmbeddingAnnealer._build_array_task() writes".
INPUTFILE = _ARG / "input_anneal.json" if _ARG.is_dir() else _ARG
calcpath = INPUTFILE.parent

input_config = EmbeddingAnnealInput.from_file(INPUTFILE)
input_config.validate(calcpath)

outpath = calcpath / "out"
# exist_ok so a re-executed/resumed task on a persisted calcdir does not
# abort before doing any work.
os.makedirs(outpath, exist_ok=True)


def _template_location(calcpath: Path, template_file: str) -> tuple[Path, str]:
    """Resolve ``template_file`` to a ``(search_dir, bare_name)`` pair.

    A plain filename lives directly in ``calcpath`` (``SingleScriptTask``'s
    convention: every box's own calcdir gets its own copy of everything). A
    value with one or more leading ``"../"`` segments (``ArrayScriptTask``'s
    convention: the deck template is identical for every box, so
    ``EmbeddingAnnealer._build_array_task`` uploads it once as a shared
    resource in the array's root -- currently two directories above each
    per-box calcdir, ``rootpath/calcdirs/<i>``, though that nesting is
    ``remote_handler``'s to change) resolves against however many parent
    directories it climbs, rather than a hardcoded depth, so this stays
    correct if that nesting ever does. The ``..`` has to be consumed here
    rather than handed to ``jinja2.Environment.get_template`` --
    ``FileSystemLoader`` rejects any ``..`` path *component* in a template
    name outright (a traversal guard), so passing it straight through would
    raise ``TemplateNotFound`` even though the file genuinely exists there.
    """
    search_dir = calcpath
    while template_file.startswith("../"):
        search_dir = search_dir.parent
        template_file = template_file[len("../") :]
    return search_dir, template_file


def _render_deck(
    output_file: Path,
    template_dir: Path,
    template_file_name: str,
    **kwargs,
) -> None:
    """Render a jinja2 LAMMPS deck template to ``output_file``."""
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template(template_file_name)
    with open(output_file.resolve(), "w") as f:
        f.write(template.render(**kwargs))


def _run_lammps(
    lmp_cmd: str,
    log_file: Path,
    input_file: Path,
    logger: logging.Logger,
    extra_flags: str = "",
) -> None:
    """Invoke ``lmp`` on a rendered deck.

    ``extra_flags`` (e.g. Kokkos/GPU acceleration flags) is inserted verbatim
    between ``-in`` and ``-log``. The right flags depend on the LAMMPS build
    and the force field in use, so they are configured by the caller rather
    than hard-coded here (empty -> a plain, unaccelerated ``lmp`` run).
    """
    command = f"{lmp_cmd} -in {input_file.resolve()}"
    if extra_flags:
        command += f" {extra_flags}"
    command += f" -log {log_file.resolve()}"
    try:
        subprocess.run(
            command,
            check=True,
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as run_error:
        logger.error(
            f"LAMMPS run returned with non-zero exit code "
            f"{run_error.returncode}"
        )
        logger.error(run_error.stderr.decode("utf-8"))
        raise RuntimeError(run_error)


type_of = {sym: i + 1 for i, sym in enumerate(input_config.species)}
mass_lines = "\n".join(
    f"mass        {i + 1}       {m}" for i, m in enumerate(input_config.masses)
)

stages = [
    {
        "index": i,
        "type_list": " ".join(str(type_of[s]) for s in stage_cfg["species"]),
        "temp_start": stage_cfg["temperatures"][0],
        # Consecutive (from, to, steps) triples -- one per segment between
        # temperature waypoints, e.g. [300, 600, 300] + [50000, 100000] ->
        # [(300, 600, 50000), (600, 300, 100000)]. Arbitrary length, so a
        # stage can have any number of ramp segments, not just two.
        "segments": list(
            zip(
                stage_cfg["temperatures"][:-1],
                stage_cfg["temperatures"][1:],
                stage_cfg["run_steps"],
            )
        ),
        "pressure_xy": stage_cfg.get("pressure_xy"),
        "pressure_z": stage_cfg.get("pressure_z"),
        # Per-stage z-wall strength, falling back to the top-level default
        # -- different stages anneal different species (e.g. a rigid metal
        # lattice vs. a mobile solvent) that may reasonably want a
        # differently-tuned (or disabled, via epsilon=0) wall.
        "z_wall_margin": stage_cfg.get(
            "z_wall_margin", input_config.z_wall_margin
        ),
        "z_wall_epsilon": stage_cfg.get(
            "z_wall_epsilon", input_config.z_wall_epsilon
        ),
        "z_wall_cutoff": stage_cfg.get(
            "z_wall_cutoff", input_config.z_wall_cutoff
        ),
    }
    for i, stage_cfg in enumerate(input_config.stages)
]

config_path = outpath / input_config.lammps_config_name
dump_file = outpath / "anneal.dump"
final_data_file = outpath / "embedded.data"

_template_dir, _template_name = _template_location(
    calcpath, input_config.template_file
)
_render_deck(
    config_path,
    _template_dir,
    _template_name,
    data_file=calcpath / input_config.data_file,
    pair_style=input_config.pair_style,
    pair_coeff_path=calcpath / input_config.pair_coeff_path,
    species=input_config.species,
    mass_lines=mass_lines,
    stages=stages,
    cutoff=input_config.cutoff,
    roi_x=input_config.roi_x,
    roi_y=input_config.roi_y,
    roi_z=input_config.roi_z,
    timestep=input_config.timestep,
    seed=input_config.seed,
    dump_every=input_config.dump_every,
    dump_file=dump_file,
    final_data_file=final_data_file,
    # Radius of the harmonic repulsive wall keeping mobile atoms from
    # approaching the frozen fixSphere core -- strictly inside `cutoff` by
    # `roi_repel_margin` so the wall's own interaction range doesn't reach
    # into the frozen region. Deck-wide (not per-stage, unlike z-wall
    # strength below): this protects the frozen core itself and should stay
    # uniformly active for every mobile atom regardless of species.
    roi_repel_radius=input_config.cutoff - input_config.roi_repel_margin,
    roi_repel_epsilon=input_config.roi_repel_epsilon,
    roi_repel_wall_cutoff=input_config.roi_repel_wall_cutoff,
    # z-wall strength is resolved per-stage (each `stages` entry above
    # already carries its own z_wall_margin/epsilon/cutoff, falling back to
    # these same input_config values when a stage doesn't override them).
)

_run_lammps(
    lmp_cmd=input_config.lammps_path,
    log_file=outpath / "lmp.log",
    input_file=config_path,
    logger=logging.getLogger(),
    extra_flags=input_config.lammps_flags,
)

# lammps-data has no element symbols, only type numbers -- Z_of_type maps
# them back using the same type order the deck's mass/pair_coeff lines used.
z_of_type = {
    i + 1: atomic_numbers[sym] for i, sym in enumerate(input_config.species)
}
final_atoms = ase.io.read(
    final_data_file, format="lammps-data", Z_of_type=z_of_type
)
ase.io.write(outpath / "samples.extxyz", final_atoms, format="extxyz")

# Success marker checked by SingleScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text("embedding anneal finished")
