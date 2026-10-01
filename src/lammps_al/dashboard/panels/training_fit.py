"""Per-committee-member fit quality of the freshly trained committee, read
directly from pacemaker's own outputs -- no re-scoring.

Distinct from ``committee_parity``'s panels, which re-score configs against
the committee to check *predictive* fit quality (in-sample or held-out).
These instead answer "did the fit itself converge, how good is each member's
final fit, and how much do members disagree in their own fit quality":
convergence curves from ``metrics.txt``/``test_metrics.txt`` (see
``committee_eval.read_pacemaker_metrics``), final-fit parity from
``train_pred``/``test_pred.pckl.gzip`` (see
``committee_eval.read_pacemaker_predictions``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData
from alviz.panels.training.parity import parity_scatter

from lammps_al.dashboard.committee_eval import (
    read_pacemaker_metrics,
    read_pacemaker_predictions,
)
from lammps_al.dashboard.context import named
from lammps_al.dashboard.panels.committee_parity import _mae, _rmse

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


class TrainingParityData(PanelData):
    """Per-member final-fit parity on each member's own split."""

    # Keyed by str(member index), like TrainingConvergenceData.
    e_obs: dict[str, list[float]]
    e_pred: dict[str, list[float]]
    # Possibly a random subsample (see TrainingParityPanel's
    # max_force_points); the error metrics below are always over all
    # force components.
    f_obs: dict[str, list[float]]
    f_pred: dict[str, list[float]]
    e_rmse: dict[str, float]
    e_mae: dict[str, float]
    f_rmse: dict[str, float]
    f_mae: dict[str, float]


def _member_stat(values: dict) -> str:
    """One metric across members: the value itself for a single member,
    else mean ± std over members."""
    vals = np.asarray(list(values.values()), dtype=float)
    if vals.size == 1:
        return f"{vals[0]:.4g}"
    return f"{vals.mean():.4g} ± {vals.std():.2g}"


class TrainingParityPanel(Entry):
    """Energy/force parity of the freshly trained committee, one color per
    member, on each member's own train or test split.

    Read from pacemaker's ``{split}_pred.pckl.gzip`` (the final potential's
    predictions, written at the end of each member's fit), so it costs no
    re-scoring. Complements ``training_convergence`` (RMSE vs. fitting
    iteration) with the final fit's full error distribution -- outliers and
    systematic bias a single RMSE hides. Energies are per atom, on
    pacemaker's ``energy_corrected`` scale (see
    ``committee_eval.read_pacemaker_predictions``).

    Members' splits generally differ (pacemaker resamples per member), so
    points are per member, not a committee mean.

    Args:
        calcdirs: One pacemaker calcdir per committee member, named by
            member index (see ``run_layout.trainer_calcdirs_for_iteration``).
        split: ``"train"`` or ``"test"`` (only present if pacemaker was
            configured with a held-out split).
        max_force_points: Cap on force components *plotted* (and cached)
            per member, drawn uniformly at random with a fixed seed; a
            training set easily has ~1e6 components per member. ``None``
            keeps all. RMSE/MAE are always over all components.
    """

    name = "training_parity"
    data_class = TrainingParityData
    colspan = 2

    def __init__(
        self,
        calcdirs,
        *args,
        split: str = "train",
        max_force_points: int = 20_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calcdirs = [Path(p) for p in calcdirs]
        self._split = split
        self._max_force_points = max_force_points

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 2)
        self.ax_e = fig.add_subplot(self.gs[0])
        self.ax_f = fig.add_subplot(self.gs[1])

    def compute_data(self, shared_ressources: dict) -> TrainingParityData:
        if not self._calcdirs:
            raise ValueError("No trainer calcdirs given.")

        rng = np.random.default_rng(0)
        fields = {k: {} for k in TrainingParityData.model_fields}
        for calcdir in self._calcdirs:
            member = calcdir.name
            try:
                df = read_pacemaker_predictions(calcdir, split=self._split)
            except FileNotFoundError:
                logger.info(
                    "member %s (%s): no %s predictions -- skipping this "
                    "member.",
                    member,
                    calcdir,
                    self._split,
                )
                continue
            n_atoms = df["NUMBER_OF_ATOMS"].to_numpy(dtype=float)
            e_obs = df["energy_corrected"].to_numpy(dtype=float) / n_atoms
            e_pred = df["energy_pred"].to_numpy(dtype=float) / n_atoms
            f_obs = np.concatenate(
                [np.asarray(f, dtype=float).ravel() for f in df["forces"]]
            )
            f_pred = np.concatenate(
                [np.asarray(f, dtype=float).ravel() for f in df["forces_pred"]]
            )

            fields["e_rmse"][member] = _rmse(e_obs, e_pred)
            fields["e_mae"][member] = _mae(e_obs, e_pred)
            fields["f_rmse"][member] = _rmse(f_obs, f_pred)
            fields["f_mae"][member] = _mae(f_obs, f_pred)

            if (
                self._max_force_points is not None
                and f_obs.size > self._max_force_points
            ):
                keep = rng.choice(
                    f_obs.size, self._max_force_points, replace=False
                )
                f_obs, f_pred = f_obs[keep], f_pred[keep]
            fields["e_obs"][member] = e_obs.tolist()
            fields["e_pred"][member] = e_pred.tolist()
            fields["f_obs"][member] = f_obs.tolist()
            fields["f_pred"][member] = f_pred.tolist()

        if not fields["e_obs"]:
            raise ValueError(
                f"No readable {self._split} predictions for any committee "
                f"member ({len(self._calcdirs)} calcdir(s) checked)."
            )
        return TrainingParityData(**fields)

    def plot_data(self):
        d = self.data
        members = sorted(d.e_obs, key=int)
        for i, member in enumerate(members):
            style = dict(s=4, alpha=0.3, color=f"C{i}", rasterized=True, lw=0)
            parity_scatter(
                self.ax_e,
                d.e_obs[member],
                d.e_pred[member],
                label=f"member {member}",
                **style,
            )
            parity_scatter(
                self.ax_f, d.f_obs[member], d.f_pred[member], **style
            )
        for ax, rmse, mae in (
            (self.ax_e, d.e_rmse, d.e_mae),
            (self.ax_f, d.f_rmse, d.f_mae),
        ):
            ax.text(
                0.05,
                0.95,
                f"RMSE = {_member_stat(rmse)}\nMAE = {_member_stat(mae)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
        self.ax_e.set_xlabel("reference E (eV/atom)")
        self.ax_e.set_ylabel("predicted E (eV/atom)")
        self.ax_e.set_title(f"{self._split}: energy")
        self.ax_e.legend(fontsize="small", loc="lower right", markerscale=3)
        self.ax_f.set_xlabel("reference F (eV/Å)")
        self.ax_f.set_ylabel("predicted F (eV/Å)")
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


class TrainingParitySpec(PanelSpec):
    type: str = "training_parity"
    split: Literal["train", "test"] = "train"
    max_force_points: Optional[int] = 20_000


def _build_training_parity(spec, ctx, common):
    if ctx.is_summary:
        return None
    if not ctx.trainer_calcdirs:
        return ctx.skip(spec.type, "no trainer calcdirs found")
    return TrainingParityPanel(
        ctx.trainer_calcdirs,
        split=spec.split,
        max_force_points=spec.max_force_points,
        **named(spec),
        **common,
    )


register_panel("training_parity", TrainingParitySpec, _build_training_parity)
