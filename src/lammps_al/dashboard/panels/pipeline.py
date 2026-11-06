"""Pipeline attrition: how many candidates survive each stage, per
iteration -- ROIs extracted -> boxes annealed -> DFT finished -> configs
actually added.

Every drop is a box the selector paid for that never reached training; the
panel names each lost anneal box with the last ``ERROR`` line of its LAMMPS
log (a box that blew up mid-anneal shows up here instead of being silently
averaged into ``annealer_thermo``'s percentile band). Pure file counting --
no structures beyond the (small) evaluator output are read.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.context import named
from lammps_al.dashboard.panels.committee_parity import _new_frames, _read_all

logger = logging.getLogger(__name__)

STAGES = [
    "ROIs extracted",
    "anneal launched",
    "anneal finished",
    "DFT launched",
    "DFT finished",
    "configs added",
]

# Longest failure note kept per lost box (one line of a PDF panel).
_MAX_NOTE_CHARS = 90


def _last_error_line(log_path) -> str:
    """The last LAMMPS ``ERROR`` line of ``log_path``, else its last
    non-empty line (a box killed without LAMMPS reporting why)."""
    if log_path is None:
        return "no LAMMPS log"
    lines = [
        ln.strip()
        for ln in Path(log_path).read_text(errors="replace").splitlines()
        if ln.strip()
    ]
    errors = [ln for ln in lines if ln.startswith("ERROR")]
    if errors:
        return errors[-1]
    return lines[-1] if lines else "empty LAMMPS log"


class PipelineAttritionData(PanelData):
    """Per-iteration stage counts (``None`` = stage never ran) + notes."""

    iterations: list[int]
    stages: list[str]
    # Keyed by str(iteration); one count per STAGES entry.
    counts: dict[str, list[int | None]]
    notes: list[str]


class PipelineAttritionPanel(Entry):
    """Grouped bars: one group per stage, one bar per iteration.

    Args:
        iterations: ``{iteration: artifacts}``, each ``artifacts`` a dict
            with ``selector_manifest`` (path or ``None``),
            ``annealer_boxes`` (``[(box, finished, log_or_None), ...]``),
            ``dft_calcs`` (``[(calcdir, finished), ...]``),
            ``config_files`` and ``prev_config_files`` (evaluated-config
            path lists; the latter de-duplicates a cumulative evaluator
            output, see ``committee_parity._new_frames``).
    """

    name = "pipeline_attrition"
    data_class = PipelineAttritionData
    colspan = 2

    def __init__(self, iterations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterations = dict(sorted(iterations.items()))

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(2, 1, height_ratios=[3, 1])
        self.ax = fig.add_subplot(self.gs[0])
        self.ax_notes = fig.add_subplot(self.gs[1])

    def compute_data(self, shared_ressources: dict) -> PipelineAttritionData:
        counts, notes = {}, []
        for it, a in self._iterations.items():
            manifest = a["selector_manifest"]
            rois = (
                len(json.loads(Path(manifest).read_text()))
                if manifest is not None
                else None
            )
            boxes = a["annealer_boxes"]
            dft = a["dft_calcs"]
            added = (
                len(
                    _new_frames(
                        _read_all(a["config_files"]),
                        _read_all(a["prev_config_files"]),
                    )
                )
                if a["config_files"]
                else None
            )
            counts[str(it)] = [
                rois,
                len(boxes) if boxes else None,
                sum(ok for _, ok, _ in boxes) if boxes else None,
                len(dft) if dft else None,
                sum(ok for _, ok in dft) if dft else None,
                added,
            ]
            for box, ok, log in boxes:
                if not ok:
                    note = f"it{it} anneal box {box}: {_last_error_line(log)}"
                    notes.append(note[:_MAX_NOTE_CHARS])
            for calc, ok in dft:
                if not ok:
                    notes.append(f"it{it} DFT calcdir {calc}: not finished")
        if not any(any(c is not None for c in v) for v in counts.values()):
            raise ValueError("No pipeline stage has run in any iteration.")
        logger.info(
            "Pipeline attrition over %d iteration(s); %d lost item(s).",
            len(counts),
            len(notes),
        )
        return PipelineAttritionData(
            iterations=list(self._iterations),
            stages=STAGES,
            counts=counts,
            notes=notes,
        )

    def plot_data(self):
        d = self.data
        n_it = len(d.iterations)
        x = np.arange(len(d.stages))
        width = 0.8 / max(n_it, 1)
        for i, it in enumerate(d.iterations):
            values = [np.nan if c is None else c for c in d.counts[str(it)]]
            bars = self.ax.bar(
                x + (i - (n_it - 1) / 2) * width,
                values,
                width=width * 0.9,
                label=f"iter {it}",
            )
            # Counts are small integers: label every bar while there are few
            # enough iterations for the labels not to collide.
            if n_it <= 3:
                self.ax.bar_label(
                    bars,
                    labels=[
                        "" if c is None else str(c) for c in d.counts[str(it)]
                    ],
                    fontsize="x-small",
                )
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(d.stages, rotation=20, ha="right")
        self.ax.set_ylabel("count")
        if n_it > 1:
            self.ax.legend(fontsize="small")
        self.ax.set_title(self.name)

        self.ax_notes.axis("off")
        text = "\n".join(d.notes) if d.notes else "no losses"
        self.ax_notes.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize="x-small",
            family="monospace",
            transform=self.ax_notes.transAxes,
        )


class PipelineAttritionSpec(PanelSpec):
    type: str = "pipeline_attrition"


def _build_pipeline_attrition(spec, ctx, common):
    if not ctx.is_summary:
        return None
    iterations = {
        c.iteration: {
            "selector_manifest": c.selector_manifest,
            "annealer_boxes": c.annealer_boxes,
            "dft_calcs": c.dft_calcs,
            "config_files": c.config_files,
            "prev_config_files": c.prev_config_files,
        }
        for c in ctx.iteration_contexts
        if c.selector_manifest
        or c.annealer_boxes
        or c.dft_calcs
        or c.config_files
    }
    if not iterations:
        return ctx.skip(spec.type, "no selector/annealer/DFT output yet")
    return PipelineAttritionPanel(iterations, **named(spec), **common)


register_panel(
    "pipeline_attrition", PipelineAttritionSpec, _build_pipeline_attrition
)
