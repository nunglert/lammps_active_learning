"""The resolved artifacts one dashboard page is built from.

The single seam where "where the files live" (:mod:`run_layout`) meets
"what the panels need": every path a page's panels read is resolved here
once, up front, and panel builders (registered alongside each panel, see
:mod:`panel_specs`) only gate on and pass along these fields -- neither the
builders nor the panels themselves call :mod:`run_layout`. Mirrors
``jaxrens_al.dashboard.RunContext``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lammps_al.dashboard import run_layout

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Artifacts + settings one page is assembled from.

    ``iteration is None`` marks the campaign-summary page; otherwise it is
    the AL iteration this page covers. Fields a view doesn't use stay empty
    (per-iteration fields on the summary page and vice versa), so builders
    gate on them uniformly.
    """

    calchome: Path
    iteration: Optional[int] = None
    # LAMMPS type order; inherited by any panel spec declaring ``species``.
    species: list = field(default_factory=list)

    # Per-iteration view.
    model_files: list = field(default_factory=list)
    config_files: list = field(default_factory=list)
    # The previous iteration's committee (the one that drove this
    # iteration's sampling) and evaluated configs -- what the held-out
    # parity panel scores against / diffs against.
    prev_model_files: list = field(default_factory=list)
    prev_config_files: list = field(default_factory=list)
    trainer_calcdirs: list = field(default_factory=list)
    selector_manifest: Optional[Path] = None
    annealed_configs: Optional[Path] = None
    sampler_log: Optional[Path] = None
    annealer_logs: list = field(default_factory=list)

    # Campaign-summary view: ``{iteration: path}``, unfinished iterations
    # absent.
    uncertainty_trajectories: dict = field(default_factory=dict)

    @property
    def is_summary(self) -> bool:
        return self.iteration is None

    @property
    def page_label(self) -> str:
        if self.is_summary:
            return "campaign summary"
        return f"iteration {self.iteration}"

    def skip(self, panel: str, reason: str) -> None:
        """Log why a builder dropped ``panel`` from this page; returns
        ``None`` so a builder can ``return ctx.skip(...)``."""
        logger.info("%s: skipping %s (%s).", self.page_label, panel, reason)


def named(spec) -> dict:
    """``Entry`` kwargs for an explicit spec ``name`` (else the panel class's
    own default name stands)."""
    return {"name": spec.name} if spec.name else {}


def _at(per_iter: list, it: int) -> list:
    return per_iter[it] if 0 <= it < len(per_iter) else []


def iteration_context(calchome, progress, it: int, species) -> RunContext:
    """Resolve everything iteration ``it``'s page can show."""
    calchome = Path(calchome)
    trained = run_layout.trained_models_per_iteration(calchome, progress)
    evaluated = run_layout.evaluated_configs_per_iteration(calchome, progress)
    return RunContext(
        calchome=calchome,
        iteration=it,
        species=list(species),
        model_files=_at(trained, it),
        config_files=_at(evaluated, it),
        prev_model_files=_at(trained, it - 1),
        prev_config_files=_at(evaluated, it - 1),
        trainer_calcdirs=run_layout.trainer_calcdirs_for_iteration(
            calchome, it
        ),
        selector_manifest=run_layout.selector_output_manifest(
            calchome, progress, it
        ),
        annealed_configs=run_layout.annealed_configs(calchome, progress, it),
        sampler_log=run_layout.sampler_lmp_log(calchome, it),
        annealer_logs=run_layout.annealer_lmp_logs_for_iteration(calchome, it),
    )


def campaign_context(calchome, progress, iterations, species) -> RunContext:
    """Resolve everything the campaign-summary page can show."""
    calchome = Path(calchome)
    trajectories = {}
    for it in iterations:
        path = run_layout.sampler_uncertainty_trajectory(
            calchome, progress, it
        )
        if path is not None:
            trajectories[it] = path
    return RunContext(
        calchome=calchome,
        species=list(species),
        uncertainty_trajectories=trajectories,
    )
