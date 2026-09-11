"""Read a remote_handler active-learning run's on-disk artifacts.

This is the project's *bridge*, adapted from ``jaxrens_al.dashboard.
run_layout``'s proven pattern: it understands ``StagedActiveLearningHandler``'s
run layout (the ``calchome`` tree + ``AL_progress.json``) and turns it into
plain file paths / ``Atoms`` lists that dashboard panels consume. It reads
*artifacts only* and deliberately does **not** import ``remote_handler``: the
dashboard side is coupled to the orchestrator's output *format*, not its
code. ``AL_progress.json`` is the source of truth -- the recorded output
paths are used rather than hard-coded directory names.

``AL_progress.json``'s staged schema (one object, ``outputs`` keyed by stage
then iteration)::

    { stages, entry_index, exit_index, num_iterations,
      outputs: { <stage>: { <iter>: [rel, ...] } } }

is pivoted to one internal per-iteration list of ``{<stage>_output: [rel,
...]}`` dicts, exactly like ``jaxrens_al``'s bridge -- just generalized to
this campaign's 6 stages (``jaxrens_al``'s classic 4-stage AL ring has
neither ``uncertainty_select`` nor ``annealer``).

Layout::

    calchome/
    ├── 0/  1/  ...                    # per-iteration actor working dirs
    ├── sampler_output/                # raw MD trajectories (no uncertainty yet)
    ├── uncertainty_select_output/     # whole trajectory, annotated (not reduced)
    ├── selector_output/               # ROI-carved boxes the selector chose
    ├── annealer_output/               # annealed configs sent for DFT
    ├── evaluator_output/              # DFT-labelled configs (REF_energy/REF_forces)
    ├── trainer_output/                # trained committee-member potentials
    └── AL_progress.json

One artifact is never recorded in ``AL_progress.json`` at all -- the
selector's ROI manifest -- reached into directly, by the raw per-iteration
working-dir convention (confirmed from ``remote_handler``'s ``origin/dev-NU``
source): ``calchome/{it}/extract/out/manifest.json``. The full per-atom
uncertainty trajectory, by contrast, *is* the collected
``uncertainty_select_output/`` (``CommitteeUncertaintySelector`` annotates
without reducing -- see ``lammps_al.select.selector``'s module docstring),
so it needs no raw reach-in.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Stage name -> the classic per-iteration ``<stage>_output`` key every
# accessor below understands. This campaign's ring has 6 stages (sampler ->
# uncertainty_select -> selector -> annealer -> evaluator -> trainer);
# jaxrens_al's has 4 (no uncertainty_select, no annealer).
_STAGE_TO_OUTPUT_KEY = {
    "sampler": "sampler_output",
    "uncertainty_select": "uncertainty_select_output",
    "selector": "selector_output",
    "annealer": "annealer_output",
    "evaluator": "evaluator_output",
    "trainer": "trainer_output",
}

# Raw per-iteration working-dir names, confirmed against remote_handler's
# StagedActiveLearningHandler (origin/dev-NU): calchome/{it}/<name>/...
_EXTRACT_DIRNAME = "extract"


def load_progress(calchome) -> list:
    """Load ``AL_progress.json`` as a per-iteration list of dicts.

    Accepts either the classic (a list of per-iteration dicts) or the staged
    schema (see the module docstring) and returns the same normalized
    per-iteration list either way.
    """
    import json

    calchome = Path(calchome)
    progress_file = calchome / "AL_progress.json"
    if not progress_file.exists():
        raise FileNotFoundError(
            f"No AL_progress.json found in {calchome}. Is this a "
            "remote_handler run directory?"
        )
    with open(progress_file, "r") as f:
        progress = json.load(f)
    return _as_iteration_list(progress)


def _as_iteration_list(progress) -> list:
    """Normalize any supported schema to an ordered per-iteration list."""
    if isinstance(progress, dict):
        if "outputs" in progress:  # staged schema
            return _normalize_staged(progress)
        # A bare dict keyed by iteration index (a tolerated classic variant).
        return [progress[k] for k in sorted(progress, key=lambda x: int(x))]
    return list(progress)


def _normalize_staged(progress) -> list:
    """Turn the staged ``outputs`` map into a per-iteration list of dicts.

    ``outputs`` is ``{stage: {iter: [rel, ...]}}``; this pivots it to one
    dict per iteration keyed by the classic ``<stage>_output`` names,
    covering iterations ``0..max`` that any stage recorded (partial/
    in-flight runs stop at the last completed iteration rather than padding
    to ``num_iterations``).
    """
    outputs = progress.get("outputs", {}) or {}
    max_iter = -1
    for stage_map in outputs.values():
        for it in stage_map or {}:
            max_iter = max(max_iter, int(it))

    result = []
    for i in range(max_iter + 1):
        entry = {}
        for stage, key in _STAGE_TO_OUTPUT_KEY.items():
            stage_map = outputs.get(stage, {}) or {}
            entry[key] = stage_map.get(str(i), stage_map.get(i, [])) or []
        result.append(entry)
    return result


def _collect(calchome, progress, key) -> list:
    calchome = Path(calchome)
    paths = []
    for iteration in progress:
        for rel in iteration.get(key, []) or []:
            paths.append((calchome / rel).resolve())
    return paths


def evaluated_configs(calchome, progress=None) -> list:
    """Absolute paths to all recorded DFT-evaluated config files, in order."""
    if progress is None:
        progress = load_progress(calchome)
    return _collect(calchome, progress, "evaluator_output")


def evaluated_configs_per_iteration(calchome, progress=None) -> list:
    """List (per iteration) of the recorded evaluated-config paths."""
    calchome = Path(calchome)
    if progress is None:
        progress = load_progress(calchome)
    out = []
    for iteration in progress:
        rels = iteration.get("evaluator_output", []) or []
        out.append([(calchome / rel).resolve() for rel in rels])
    return out


def trained_models_per_iteration(calchome, progress=None) -> list:
    """List (per iteration) of the recorded trained committee-model paths."""
    calchome = Path(calchome)
    if progress is None:
        progress = load_progress(calchome)
    out = []
    for iteration in progress:
        rels = iteration.get("trainer_output", []) or []
        out.append([(calchome / rel).resolve() for rel in rels])
    return out


def iteration_dirs(calchome) -> list:
    """Numeric per-iteration working dirs (``calchome/0``, ``1``, ...)."""
    calchome = Path(calchome)
    if not calchome.exists():
        return []
    dirs = [p for p in calchome.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def sampler_uncertainty_trajectory(calchome, progress, it: int):
    """The full uncertainty-tagged trajectory for iteration ``it``, or
    ``None``.

    ``CommitteeUncertaintySelector`` annotates without reducing (see
    ``lammps_al.select.selector``'s module docstring), so the whole
    per-atom-uncertainty-tagged trajectory *is* the recorded
    ``uncertainty_select_output`` for that iteration -- no raw reach-in
    needed. Returns ``None`` if that iteration never ran this stage (e.g.
    iteration 0 under ``entry_stage="trainer"``, bootstrap-train only) or
    hasn't finished yet.
    """
    if it >= len(progress):
        return None
    rels = progress[it].get("uncertainty_select_output", []) or []
    if not rels:
        return None
    return (Path(calchome) / rels[0]).resolve()


def selector_manifest(calchome, it: int):
    """This iteration's ROI manifest, or ``None``.

    ``calchome/{it}/extract/out/manifest.json`` -- the selector's
    ``output_extension="dir"`` output is copied whole, never flattened into
    ``selector_output/`` in a form that names this file directly.
    """
    p = Path(calchome) / str(it) / _EXTRACT_DIRNAME / "out" / "manifest.json"
    return p if p.exists() else None


def _frames_match(a_frames, b_frames, tol: float = 1e-8) -> bool:
    """Cheap content check between two equal-length frame lists.

    Compares atom count and ``REF_energy`` of the first and last frame only
    -- enough to distinguish "these are the same underlying frames" from
    "these are unrelated data" without reading every array in every frame.
    """
    if len(a_frames) != len(b_frames):
        return False
    for a, b in zip((a_frames[0], a_frames[-1]), (b_frames[0], b_frames[-1])):
        if len(a) != len(b):
            return False
        ea = a.info.get("REF_energy")
        eb = b.info.get("REF_energy")
        if ea is None or eb is None or abs(ea - eb) > tol:
            return False
    return True


def new_evaluator_configs_for_iteration(calchome, progress, it: int) -> list:
    """Frames genuinely new to iteration ``it``'s evaluator output.

    Whether ``evaluator_it{it}`` is a fresh delta or cumulative-to-date is
    not a settled property of this campaign's remote_handler version (see
    the dashboard plan doc) -- so this auto-detects: if iteration ``it``'s
    frames are a strict superset of iteration ``it - 1``'s (by count, with
    matching first/last-frame content), only the new tail is returned;
    otherwise the whole file is treated as already-new. Logs which case
    fired. Returns ``[]`` when iteration ``it`` has no evaluator output at
    all (e.g. iteration 0 under ``entry_stage="trainer"``).
    """
    import ase.io

    per_iter = evaluated_configs_per_iteration(calchome, progress)
    if it < 0 or it >= len(per_iter) or not per_iter[it]:
        return []

    current = []
    for f in per_iter[it]:
        current.extend(ase.io.read(f, index=":"))

    if it == 0 or it - 1 >= len(per_iter) or not per_iter[it - 1]:
        return current

    previous = []
    for f in per_iter[it - 1]:
        previous.extend(ase.io.read(f, index=":"))

    if (
        len(current) > len(previous)
        and previous
        and _frames_match(current[: len(previous)], previous)
    ):
        logger.info(
            "evaluator_it%d looks cumulative (contains evaluator_it%d's "
            "%d frames as a prefix); returning only the new tail "
            "(%d frames).",
            it,
            it - 1,
            len(previous),
            len(current) - len(previous),
        )
        return current[len(previous) :]

    logger.info(
        "evaluator_it%d looks like a delta (does not extend evaluator_it%d "
        "as a prefix); returning all %d frames as new.",
        it,
        it - 1,
        len(current),
    )
    return current
