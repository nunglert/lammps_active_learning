"""Per-committee-member training convergence, read directly from
pacemaker's own fit log -- no re-scoring.

Distinct from ``committee_parity``'s panels, which re-score configs against
the committee to check *predictive* fit quality (in-sample or held-out).
This instead answers "did the fit itself converge, and how much do members
disagree in their own fit quality" -- read straight from ``metrics.txt``/
``test_metrics.txt`` (see ``committee_eval.read_pacemaker_metrics``), so it
costs nothing beyond reading a handful of small text files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.committee_eval import read_pacemaker_metrics
from lammps_al.dashboard.context import named

logger = logging.getLogger(__name__)


class TrainingConvergenceData(PanelData):
    """Per-member fitting-iteration RMSE curves for one AL iteration."""

    # Keyed by str(member index) (JSON object keys are always strings).
    iter_num: dict[str, list[int]]
    rmse_epa: dict[str, list[float]]
    rmse_f_comp: dict[str, list[float]]


class TrainingConvergencePanel(Entry):
    """Energy/force RMSE vs. fitting iteration, one line per committee
    member.

    Answers two things a parity plot alone can't: whether the fit actually
    converged (a curve still dropping at the last logged iteration means
    ``pacemaker``'s own iteration budget cut it short) and how much the
    members' own fit quality already disagrees *before* any held-out
    scoring -- a big spread here is a first, cheap hint that the committee's
    predictive disagreement (see ``uncertainty`` panels) may partly reflect
    fit variance rather than genuine data-driven epistemic uncertainty.

    Args:
        calcdirs: One pacemaker calcdir per committee member, named by
            member index (see ``run_layout.trainer_calcdirs_for_iteration``).
        split: ``"train"`` (every fitting iteration) or ``"test"`` (sparser
            checkpoints, if pacemaker was configured with a held-out split).
    """

    name = "training_convergence"
    data_class = TrainingConvergenceData
    colspan = 2

    def __init__(
        self,
        calcdirs,
        *args,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calcdirs = [Path(p) for p in calcdirs]
        self._split = split

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 2)
        self.ax_e = fig.add_subplot(self.gs[0])
        self.ax_f = fig.add_subplot(self.gs[1])

    def compute_data(self, shared_ressources: dict) -> TrainingConvergenceData:
        calcdirs = self._calcdirs
        if not calcdirs:
            raise ValueError("No trainer calcdirs given.")

        iter_num: dict = {}
        rmse_epa: dict = {}
        rmse_f_comp: dict = {}
        for calcdir in calcdirs:
            member = calcdir.name
            try:
                df = read_pacemaker_metrics(calcdir, split=self._split)
            except FileNotFoundError:
                logger.info(
                    "member %s (%s): no %s metrics file -- skipping this "
                    "member.",
                    member,
                    calcdir,
                    self._split,
                )
                continue
            iter_num[member] = df["iter_num"].tolist()
            rmse_epa[member] = df["rmse_epa"].tolist()
            rmse_f_comp[member] = df["rmse_f_comp"].tolist()

        if not iter_num:
            raise ValueError(
                f"No readable {self._split} metrics for any committee "
                f"member ({len(calcdirs)} calcdir(s) checked)."
            )
        return TrainingConvergenceData(
            iter_num=iter_num, rmse_epa=rmse_epa, rmse_f_comp=rmse_f_comp
        )

    def plot_data(self):
        d = self.data
        for member in sorted(d.iter_num, key=int):
            self.ax_e.plot(
                d.iter_num[member],
                d.rmse_epa[member],
                label=f"member {member}",
            )
            self.ax_f.plot(d.iter_num[member], d.rmse_f_comp[member])
        self.ax_e.set_xlabel("fitting iteration")
        self.ax_e.set_ylabel("RMSE energy/atom (eV)")
        self.ax_e.set_yscale("log")
        self.ax_e.legend(fontsize="small")
        self.ax_e.set_title(f"{self._split}: energy")
        self.ax_f.set_xlabel("fitting iteration")
        self.ax_f.set_ylabel("RMSE force component (eV/Å)")
        self.ax_f.set_yscale("log")
        self.ax_f.set_title(f"{self._split}: forces")


class TrainingConvergenceSpec(PanelSpec):
    type: str = "training_convergence"
    split: Literal["train", "test"] = "train"


def _build_training_convergence(spec, ctx, common):
    if ctx.is_summary:
        return None
    if not ctx.trainer_calcdirs:
        return ctx.skip(spec.type, "no trainer calcdirs found")
    return TrainingConvergencePanel(
        ctx.trainer_calcdirs, split=spec.split, **named(spec), **common
    )


register_panel(
    "training_convergence",
    TrainingConvergenceSpec,
    _build_training_convergence,
)
