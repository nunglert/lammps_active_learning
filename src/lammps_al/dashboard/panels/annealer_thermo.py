"""Annealer MD diagnostics: classical thermo trajectory, aggregated across
one iteration's carved boxes.

Same idea as ``sampler_thermo`` (read LAMMPS's own log, don't recompute
anything), but an iteration anneals many boxes (one array-job task each,
see ``run_layout.annealer_lmp_logs_for_iteration``) rather than one MD run,
so a single-line plot isn't the right shape here -- this instead plots the
mean +/- a percentile band across boxes at each step, the same "typical
trajectory + spread" idiom ``uncertainty.UncertaintyTrendPanel`` already
uses. Boxes vary in atom count, so raw total energy isn't directly
comparable across them (a bigger box's TotEng is more negative purely from
size) -- each box's own energy trace is expressed as a delta from *that
box's* step-0 value instead, making "did this box's energy stay flat or
drift/blow up during annealing" comparable across every box regardless of
size. Temperature needs no such normalization (already intensive).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard import run_layout
from lammps_al.dashboard.lammps_log import read_lammps_thermo

logger = logging.getLogger(__name__)


class AnnealerThermoData(PanelData):
    """Mean +/- percentile-band thermo trace across one iteration's boxes."""

    step: list[float]
    n_boxes_at_step: list[int]
    temp_mean: list[float]
    temp_low: list[float]
    temp_high: list[float]
    energy_delta_mean: list[float]
    energy_delta_low: list[float]
    energy_delta_high: list[float]
    stage_boundary_steps: list[float]
    n_boxes_total: int


class AnnealerThermoPanel(Entry):
    """Temperature and per-box energy-drift vs. step, across one
    iteration's annealed boxes.

    Args:
        calchome: The campaign's ``calchome`` directory.
        iteration: Which iteration's annealer boxes to read.
        low_pct / high_pct: Percentile band to shade.
    """

    name = "annealer_thermo"
    data_class = AnnealerThermoData
    colspan = 2

    def __init__(
        self,
        calchome,
        iteration: int,
        *args,
        low_pct=10,
        high_pct=90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calchome = Path(calchome)
        self._iteration = iteration
        self._low_pct = low_pct
        self._high_pct = high_pct

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 2)
        self.ax_temp = fig.add_subplot(self.gs[0])
        self.ax_energy = fig.add_subplot(self.gs[1])

    def compute_data(self, shared_ressources: dict) -> AnnealerThermoData:
        import pandas as pd

        logs = run_layout.annealer_lmp_logs_for_iteration(
            self._calchome, self._iteration
        )
        if not logs:
            raise ValueError(
                f"No annealer LAMMPS logs for iteration {self._iteration}."
            )

        per_box = []
        stage_boundary_steps: list = []
        for box_idx, log_path in enumerate(logs):
            df = read_lammps_thermo(log_path)
            if (
                df.empty
                or "Temp" not in df.columns
                or "TotEng" not in df.columns
            ):
                logger.info(
                    "iteration %d, box %d: no usable thermo data in %s -- "
                    "skipping this box.",
                    self._iteration,
                    box_idx,
                    log_path,
                )
                continue
            df = df.sort_values("Step")
            df["energy_delta"] = df["TotEng"] - df["TotEng"].iloc[0]
            df["box"] = box_idx
            per_box.append(df)
            if not stage_boundary_steps:
                prev = None
                for step, rb in zip(df["Step"], df["run_block"]):
                    if rb != prev and prev is not None:
                        stage_boundary_steps.append(float(step))
                    prev = rb

        if not per_box:
            raise ValueError(
                f"No box produced usable thermo data for iteration "
                f"{self._iteration} ({len(logs)} log(s) checked)."
            )
        logger.info(
            "iteration %d: aggregating thermo across %d/%d box(es).",
            self._iteration,
            len(per_box),
            len(logs),
        )

        all_rows = pd.concat(per_box, ignore_index=True)
        grouped = all_rows.groupby("Step")

        step = sorted(grouped.groups.keys())
        n_boxes_at_step = [len(grouped.get_group(s)) for s in step]
        temp_mean, temp_low, temp_high = [], [], []
        e_mean, e_low, e_high = [], [], []
        for s in step:
            g = grouped.get_group(s)
            temp_mean.append(float(g["Temp"].mean()))
            temp_low.append(float(np.percentile(g["Temp"], self._low_pct)))
            temp_high.append(float(np.percentile(g["Temp"], self._high_pct)))
            e_mean.append(float(g["energy_delta"].mean()))
            e_low.append(
                float(np.percentile(g["energy_delta"], self._low_pct))
            )
            e_high.append(
                float(np.percentile(g["energy_delta"], self._high_pct))
            )

        return AnnealerThermoData(
            step=step,
            n_boxes_at_step=n_boxes_at_step,
            temp_mean=temp_mean,
            temp_low=temp_low,
            temp_high=temp_high,
            energy_delta_mean=e_mean,
            energy_delta_low=e_low,
            energy_delta_high=e_high,
            stage_boundary_steps=stage_boundary_steps,
            n_boxes_total=len(per_box),
        )

    def plot_data(self):
        d = self.data
        self.ax_temp.plot(d.step, d.temp_mean, lw=1, label="mean")
        self.ax_temp.fill_between(
            d.step,
            d.temp_low,
            d.temp_high,
            alpha=0.2,
            label=f"{self._low_pct}-{self._high_pct} pct",
        )
        self.ax_energy.plot(d.step, d.energy_delta_mean, lw=1)
        self.ax_energy.fill_between(
            d.step, d.energy_delta_low, d.energy_delta_high, alpha=0.2
        )
        for ax in (self.ax_temp, self.ax_energy):
            for s in d.stage_boundary_steps:
                ax.axvline(s, color="k", lw=0.5, ls=":")
            ax.set_xlabel("step")
        self.ax_temp.set_ylabel("temperature (K)")
        self.ax_temp.legend(fontsize="small")
        self.ax_temp.set_title(f"{self.name} ({d.n_boxes_total} boxes)")
        self.ax_energy.set_ylabel("total energy - step-0 value (eV)")
        self.ax_energy.set_title("per-box energy drift")
