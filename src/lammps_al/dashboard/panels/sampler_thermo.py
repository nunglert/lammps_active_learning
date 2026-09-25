"""Sampler MD diagnostics: classical thermo trajectory, read straight from
LAMMPS's own log (``run_layout.sampler_lmp_log``) via ``lammps_log``.

We're driving classical MD here (a committee potential, not DFT), so the
usual sanity checks apply and are cheap to get for free: did temperature
equilibrate to the target, did total energy stay flat (a runaway/blown-up
trajectory shows up immediately as a diverging TotEng), etc. This never
recomputes anything -- it's exactly what LAMMPS already printed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard import run_layout
from lammps_al.dashboard.lammps_log import read_lammps_thermo

logger = logging.getLogger(__name__)

# thermo column -> (axis label, log-scale). Fixed set/order (not derived
# from a given log's actual columns) since allocate_panels -- which must
# decide the subplot count -- always runs before compute_data (an alviz
# Entry contract, not something this panel controls); a column absent from
# a particular log just renders as an empty "(no data)" subplot instead of
# resizing the grid.
_FIELDS = [
    ("Temp", "temperature (K)"),
    ("TotEng", "total energy (eV)"),
    ("Press", "pressure (bar)"),
]


class SamplerThermoData(PanelData):
    """One iteration's sampler thermo trace."""

    step: list[float]
    run_block: list[int]
    fields: dict[str, list[float]]  # only columns actually present


class SamplerThermoPanel(Entry):
    """Sampler LAMMPS thermo (temperature/energy/pressure) vs. step.

    Args:
        calchome: The campaign's ``calchome`` directory.
        iteration: Which iteration's sampler run to read.
    """

    name = "sampler_thermo"
    data_class = SamplerThermoData
    colspan = 2

    def __init__(self, calchome, iteration: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calchome = Path(calchome)
        self._iteration = iteration

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, len(_FIELDS))
        self.axes = [fig.add_subplot(self.gs[i]) for i in range(len(_FIELDS))]

    def compute_data(self, shared_ressources: dict) -> SamplerThermoData:
        log_path = run_layout.sampler_lmp_log(self._calchome, self._iteration)
        if log_path is None:
            raise ValueError(
                f"No sampler LAMMPS log for iteration {self._iteration}."
            )
        df = read_lammps_thermo(log_path)
        if df.empty:
            raise ValueError(f"No thermo data parsed from {log_path}.")

        present = [f for f, _ in _FIELDS if f in df.columns]
        logger.info(
            "iteration %d: %d thermo row(s), fields=%s, from %s.",
            self._iteration,
            len(df),
            present,
            log_path,
        )
        fields = {f: df[f].tolist() for f in present}
        return SamplerThermoData(
            step=df["Step"].tolist(),
            run_block=df["run_block"].tolist(),
            fields=fields,
        )

    def plot_data(self):
        d = self.data
        boundaries = sorted(set(d.run_block))
        for ax, (field, label) in zip(self.axes, _FIELDS):
            if field not in d.fields:
                ax.set_title(f"{field} (no data)")
                ax.axis("off")
                continue
            ax.plot(d.step, d.fields[field], lw=1)
            if len(boundaries) > 1:
                # Mark stage boundaries (multi-run logs only -- a plain
                # single-run sampler log has one run_block and no lines).
                prev = None
                for i, rb in enumerate(d.run_block):
                    if rb != prev and prev is not None:
                        ax.axvline(d.step[i], color="k", lw=0.5, ls=":")
                    prev = rb
            ax.set_xlabel("step")
            ax.set_ylabel(label)
        self.axes[0].set_title(self.name)
