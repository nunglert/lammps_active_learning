"""Read a remote_handler active-learning run's on-disk artifacts.

This is the project's *bridge*, adapted from ``jaxrens_al.dashboard.
run_layout``'s proven pattern: it understands ``StagedActiveLearningHandler``'s
run layout (the ``calchome`` tree + ``AL_progress.json``) and turns it into
plain file paths that dashboard panels consume (via :mod:`context`). It reads
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

# Sampler's own LAMMPS log (thermo: step/temp/energy/...), confirmed on disk:
# calchome/{it}/sample/results/out/lmp.log. Like the ROI manifest and
# trainer calcdirs, never recorded in AL_progress.json (whose
# sampler_output only keeps the uncertainty-scored trajectory extxyz).
_SAMPLE_LMP_LOG = Path("sample") / "results" / "out" / "lmp.log"

# Annealer array-job calcdirs, confirmed on disk:
# calchome/{it}/anneal/array/calcdirs/{i}/out/lmp.log -- one LAMMPS log per
# carved box, "array" (not a "array_<timestamp>" retry directory) because
# EmbeddingAnnealer's local_savepath for the array task is that fixed path
# (see annealer.py's results_dir_for) regardless of how many times a given
# iteration's anneal was resubmitted.
_ANNEAL_ARRAY_CALCDIRS = Path("anneal") / "array" / "calcdirs"

# Trainer calcdir layout, confirmed against remote_handler's
# DetachableTaskMixin/_EnsembleArrayTrainer (_iteration_subdir="train",
# default _results_subdir="results"): calchome/{it}/train/results/calcdirs/{i}
# -- one committee member per calcdir, holding pacemaker's own fit-quality
# artifacts (metrics.txt, *_pred.pckl.gzip, ...) alongside the potential
# itself. Like the selector's ROI manifest, this is never recorded in
# AL_progress.json (whose trainer_output only keeps the flat-copied
# potential .yaml, not the calcdir it came from), so it's reached into
# directly rather than derived from recorded output paths.
_TRAIN_DIRNAME = "train"


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
    result = _as_iteration_list(progress)
    logger.info(
        "Loaded %s: %d iteration(s) recorded.", progress_file, len(result)
    )
    for i, entry in enumerate(result):
        present = sorted(
            key.removesuffix("_output") for key, val in entry.items() if val
        )
        logger.debug("  iteration %d: stages with output: %s", i, present)
    return result


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


def annealed_configs(calchome, progress, it: int):
    """This iteration's recorded, annealed (post-anneal) box configs, or
    ``None``.

    ``annealer_output`` is a single combined extxyz -- ``EmbeddingAnnealer.
    get_sampled_configs()`` writes every box's own ``out/samples.extxyz``
    concatenated into one file (skipping any box whose anneal failed), so
    reading it back gives every successfully-annealed box's *final*
    structure as one flat frame list. Unlike the pre-anneal manifest, frame
    position here does not reliably correspond to a particular box index
    (a failed box is simply absent, shifting everything after it) -- fine
    for anything that pools across boxes (e.g. an aggregate RDF), not for
    anything that needs to match a specific box back to its manifest entry.
    """
    if it < 0 or it >= len(progress):
        logger.debug("iteration %d out of range for annealer_output.", it)
        return None
    rels = progress[it].get("annealer_output", []) or []
    if not rels:
        logger.debug("iteration %d has no annealer_output yet.", it)
        return None
    return (Path(calchome) / rels[0]).resolve()


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
        logger.debug(
            "iteration %d out of range (%d iteration(s) recorded).",
            it,
            len(progress),
        )
        return None
    rels = progress[it].get("uncertainty_select_output", []) or []
    if not rels:
        logger.debug("iteration %d has no uncertainty_select_output yet.", it)
        return None
    return (Path(calchome) / rels[0]).resolve()


def selector_manifest(calchome, it: int):
    """This iteration's ROI manifest, or ``None``.

    ``calchome/{it}/extract/out/manifest.json`` -- the selector's
    ``output_extension="dir"`` output is copied whole, never flattened into
    ``selector_output/`` in a form that names this file directly.
    """
    p = Path(calchome) / str(it) / _EXTRACT_DIRNAME / "out" / "manifest.json"
    if not p.exists():
        logger.debug("No selector manifest at %s.", p)
        return None
    return p


def selector_output_manifest(calchome, progress, it: int):
    """This iteration's ROI manifest, from the recorded ``selector_output``.

    Prefer this over ``selector_manifest`` when correctness matters more
    than avoiding one extra ``AL_progress.json`` lookup:
    ``calchome/{it}/extract/out/manifest.json`` is a *raw* reach-in into
    the selector's own working dir, and -- like ``sample``/``anneal/array``
    -- a re-run leaves retagged sibling dirs (``extract_<timestamp>``)
    behind without necessarily updating the un-suffixed one, so it can go
    stale relative to what the pipeline actually used downstream. The
    recorded ``selector_output`` entry is the exact copy the annealer stage
    consumed, so it can't drift from what really happened.

    Returns ``None`` if this iteration has no recorded ``selector_output``.
    """
    if it < 0 or it >= len(progress):
        logger.debug("iteration %d out of range for selector_output.", it)
        return None
    rels = progress[it].get("selector_output", []) or []
    if not rels:
        logger.debug("iteration %d has no selector_output yet.", it)
        return None
    p = (Path(calchome) / rels[0] / "manifest.json").resolve()
    if not p.exists():
        logger.debug("No manifest.json inside recorded selector_output %s.", p)
        return None
    return p


def sampler_lmp_log(calchome, it: int):
    """This iteration's sampler LAMMPS log, or ``None``.

    See ``_SAMPLE_LMP_LOG``'s comment for the raw reach-in rationale.
    ``None`` for an iteration that never ran the sampler stage (e.g.
    iteration 0 under ``entry_stage="trainer"``) or hasn't reached it yet.
    """
    p = Path(calchome) / str(it) / _SAMPLE_LMP_LOG
    if not p.exists():
        logger.debug("No sampler LAMMPS log at %s.", p)
        return None
    return p


def annealer_lmp_logs_for_iteration(calchome, it: int) -> list:
    """This iteration's per-box annealer LAMMPS logs.

    See ``_ANNEAL_ARRAY_CALCDIRS``'s comment for the raw reach-in rationale.
    Only calcdirs that actually have ``out/lmp.log`` are returned -- a box
    whose anneal crashed before LAMMPS produced any log output is silently
    skipped here (mirrors ``EmbeddingAnnealer``'s own tolerance of partial
    box failure), not an error. Sorted by numeric calcdir name (box index
    within this iteration's array, i.e. manifest position, not the box's own
    ``index`` field -- see ``EmbeddingAnnealer._build_array_task``).
    """
    base = Path(calchome) / str(it) / _ANNEAL_ARRAY_CALCDIRS
    if not base.exists():
        logger.debug("No annealer array calcdirs at %s.", base)
        return []
    calcdirs = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    logs = [c / "out" / "lmp.log" for c in calcdirs]
    logs = [p for p in logs if p.exists()]
    logger.debug(
        "iteration %d: %d/%d box(es) have an annealer LAMMPS log at %s.",
        it,
        len(logs),
        len(calcdirs),
        base,
    )
    return logs


def trainer_calcdirs_for_iteration(calchome, it: int) -> list:
    """This iteration's per-committee-member trainer calcdirs, or ``[]``.

    ``calchome/{it}/train/results/calcdirs/{i}`` -- see ``_TRAIN_DIRNAME``'s
    comment for why this is a raw reach-in rather than derived from
    ``AL_progress.json``'s recorded ``trainer_output`` (which only keeps the
    flat-copied potential ``.yaml``, not the calcdir holding pacemaker's own
    fit diagnostics). Sorted by numeric calcdir name (committee-member
    index), matching ``ArrayScriptTask``'s own ``calcdirs/<i>`` convention.
    """
    base = Path(calchome) / str(it) / _TRAIN_DIRNAME / "results" / "calcdirs"
    if not base.exists():
        logger.debug("No trainer calcdirs at %s.", base)
        return []
    dirs = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    logger.debug(
        "iteration %d: %d trainer calcdir(s) at %s.", it, len(dirs), base
    )
    return dirs
