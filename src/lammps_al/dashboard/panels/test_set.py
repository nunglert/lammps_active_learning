"""Fixed test-set errors: every iteration's committee scored on one
never-trained-on set of configurations.

``cross_iteration_errors`` scores each committee on the data blocks the
loop itself produced, so every column is different data and the matrix
mixes "the model improved" with "this iteration's data was easier". A
fixed test set (e.g. configurations deliberately held out of the seed
database) answers only the first question: does each new committee
predict the same configurations better?

Optionally a reference committee (e.g. one trained on the test set's
configurations, or a previous campaign's final committee) is scored on the
same set and drawn as a horizontal line -- the accuracy the loop is meant
to reach.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.ticker as mticker
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.committee_eval import score_with_committee
from lammps_al.dashboard.context import named
from lammps_al.dashboard.panels.committee_parity import _read_all
from lammps_al.dashboard.panels.cross_iteration import _block_errors

logger = logging.getLogger(__name__)


class FixedSetErrorData(PanelData):
    """Errors on the test set per committee iteration (+ reference)."""

    n_frames: int
    model_iterations: list[int]
    e_rmse: list[float]  # meV/atom
    f_rmse: list[float]  # eV/Å, per component
    e_mae: list[float]  # meV/atom
    f_mae: list[float]  # eV/Å, per component
    # ``(e_rmse, f_rmse, e_mae, f_mae)`` of the reference committee.
    reference: Optional[list[float]] = None


class FixedSetErrorPanel(Entry):
    """Energy/force RMSE and MAE on a fixed test set vs. committee
    iteration.

    Args:
        model_files: ``{iteration: [committee .yaml paths]}``.
        test_set: Configurations with ``REF_energy``/``REF_forces``.
        reference_models: Optional committee scored on the same set and
            drawn as a horizontal line.
        reference_label: Legend label of that line.
    """

    name = "test_set_errors"
    data_class = FixedSetErrorData
    colspan = 2

    def __init__(
        self,
        model_files,
        test_set,
        *args,
        reference_models=None,
        reference_label: str = "reference",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_files = {
            it: [Path(p) for p in files]
            for it, files in sorted(model_files.items())
        }
        self._test_set = Path(test_set)
        self._reference_models = (
            [Path(p) for p in reference_models] if reference_models else None
        )
        self._reference_label = reference_label

    def allocate_panels(self, fig):
        gs = self.subspec.subgridspec(1, 2, wspace=0.3)
        self.ax_e = fig.add_subplot(gs[0, 0])
        self.ax_f = fig.add_subplot(gs[0, 1])

    def compute_data(self, shared_ressources: dict) -> FixedSetErrorData:
        frames = _read_all([self._test_set])
        if not frames:
            raise ValueError(f"No frames in test set {self._test_set}.")
        if not self._model_files:
            raise ValueError("No trained committees to score with.")

        def score(model_files):
            energies, forces = score_with_committee(model_files, frames)
            return _block_errors(frames, energies, forces)

        rows = []
        for it, model_files in self._model_files.items():
            logger.info(
                "Scoring iteration %d's committee on %d test frame(s).",
                it,
                len(frames),
            )
            rows.append(score(model_files))
        reference = None
        if self._reference_models:
            logger.info("Scoring the reference committee on the test set.")
            reference = list(score(self._reference_models))

        e_rmse, f_rmse, e_mae, f_mae = (list(col) for col in zip(*rows))
        return FixedSetErrorData(
            n_frames=len(frames),
            model_iterations=list(self._model_files),
            e_rmse=e_rmse,
            f_rmse=f_rmse,
            e_mae=e_mae,
            f_mae=f_mae,
            reference=reference,
        )

    def plot_data(self):
        d = self.data
        for ax, rmse, mae, ref_idx, unit, quantity in (
            (self.ax_e, d.e_rmse, d.e_mae, 0, "meV/atom", "energy"),
            (self.ax_f, d.f_rmse, d.f_mae, 1, "eV/Å", "force"),
        ):
            ax.plot(d.model_iterations, rmse, "o-", label="RMSE")
            ax.plot(d.model_iterations, mae, "s--", label="MAE")
            if d.reference is not None:
                for value, style in (
                    (d.reference[ref_idx], "-"),
                    (d.reference[ref_idx + 2], "--"),
                ):
                    ax.axhline(value, color="k", ls=style, lw=0.8)
                # One legend entry for both reference lines.
                ax.plot([], [], "k-", lw=0.8, label=self._reference_label)
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_xlabel("committee iteration")
            ax.set_ylabel(f"{quantity} error ({unit})")
            ax.set_title(f"test set: {quantity} (n={d.n_frames})")
            ax.legend(fontsize=8)


class FixedSetErrorSpec(PanelSpec):
    type: str = "test_set_errors"
    # Paths relative to calchome, or absolute. test_set is required:
    # without it the panel is skipped.
    test_set: Optional[str] = None
    reference_models: Optional[list[str]] = None
    reference_label: str = "reference"


def _build_test_set_errors(spec, ctx, common):
    if not ctx.is_summary:
        return None
    model_files = {
        c.iteration: c.model_files
        for c in ctx.iteration_contexts
        if c.model_files
    }
    test_set = (
        ctx.calchome / spec.test_set if spec.test_set is not None else None
    )
    if test_set is None or not test_set.exists() or not model_files:
        return ctx.skip(
            spec.type,
            f"test set {'not set' if test_set is None else test_set}"
            f"{'' if test_set is None or test_set.exists() else ' missing'}"
            f", models for {len(model_files)} iteration(s)",
        )
    return FixedSetErrorPanel(
        model_files,
        test_set,
        reference_models=[
            ctx.calchome / p for p in spec.reference_models or []
        ],
        reference_label=spec.reference_label,
        **named(spec),
        **common,
    )


register_panel("test_set_errors", FixedSetErrorSpec, _build_test_set_errors)
