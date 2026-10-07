"""Cross-iteration errors: every iteration's committee scored on every
iteration's evaluated data.

The per-iteration parity panels each answer one cell of a bigger question
-- ``committee_parity`` is model ``N`` on data ``N`` (in-sample),
``held_out_parity`` is model ``N-1`` on data ``N`` (out-of-sample). This
panel fills in the whole model x data matrix, which is what "does the added
data actually extend what the committee can do" needs:

- reading *down a column* (one data block, successive models): does the
  error on configs the committee once failed on drop once they are trained
  on, and keep dropping/stay low afterwards?
- the cells *right of the staircase* (data not yet trained on): do later
  models already do better on data they never saw -- i.e. does the added
  data generalize, rather than just being memorized?
- the ``init`` column: does adding AL data degrade the fit to the original
  database (forgetting)?

A data block is either the initial training database or the frames one
iteration's evaluator *added* (see ``committee_parity._new_frames``), so
each config is counted once. Model ``m`` has been trained on block ``d``
iff ``d <= m`` (``init`` counting as iteration 0) -- pacemaker's own
held-out ``test_size`` split still hides a few of those frames, so
"in-sample" here is approximate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.committee_eval import score_with_committee
from lammps_al.dashboard.context import named
from lammps_al.dashboard.panels.committee_parity import _new_frames, _read_all

logger = logging.getLogger(__name__)


class CrossIterationErrorData(PanelData):
    """Model x data-block RMSE matrices (rows: models, cols: blocks)."""

    model_iterations: list[int]
    block_labels: list[str]
    # Iteration whose committee first trained on each block (``init`` -> 0).
    block_iterations: list[int]
    block_sizes: list[int]
    e_rmse: list[list[Optional[float]]]  # meV/atom
    f_rmse: list[list[Optional[float]]]  # eV/Å, per component


def _data_blocks(initial_frames, config_files_per_iteration) -> list:
    """``[(label, first_trained_iteration, frames), ...]``, one per
    non-empty block, each holding only the frames new to that block."""
    blocks = []
    if initial_frames:
        blocks.append(("init", 0, list(initial_frames)))
    previous = []
    for it, config_files in sorted(config_files_per_iteration.items()):
        current = _read_all(config_files)
        if not current:
            continue
        new = _new_frames(current, previous)
        if new:
            blocks.append((f"it{it}", it, new))
        previous = current
    return blocks


def _block_rmse(atoms_list, energies, forces):
    e_err = [
        (e - float(a.info["REF_energy"])) / len(a)
        for a, e in zip(atoms_list, energies)
    ]
    f_err = np.concatenate(
        [
            (np.asarray(f) - np.asarray(a.arrays["REF_forces"])).ravel()
            for a, f in zip(atoms_list, forces)
        ]
    )
    return (
        1e3 * float(np.sqrt(np.mean(np.square(e_err)))),
        float(np.sqrt(np.mean(np.square(f_err)))),
    )


class CrossIterationErrorPanel(Entry):
    """Energy/force RMSE of each iteration's committee on each data block.

    Every committee scores every block in one pass (one calculator build
    per committee), so the cost is ``n_models x total frames`` -- cap the
    usually much larger initial database with ``max_initial_frames``.

    Args:
        model_files: ``{iteration: [committee .yaml paths]}``.
        config_files: ``{iteration: [evaluated extxyz paths]}``.
        initial_database: Initial training database (``REF_energy``/
            ``REF_forces``), or ``None`` to leave out the ``init`` block.
        max_initial_frames: Random (seeded) subsample size of the initial
            database; ``None`` scores all of it.
        seed: Subsample seed.
    """

    name = "cross_iteration_errors"
    data_class = CrossIterationErrorData
    colspan = 2

    def __init__(
        self,
        model_files,
        config_files,
        *args,
        initial_database=None,
        max_initial_frames: Optional[int] = 300,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_files = {
            it: [Path(p) for p in files]
            for it, files in sorted(model_files.items())
        }
        self._config_files = {
            it: [Path(p) for p in files]
            for it, files in sorted(config_files.items())
        }
        self._initial_database = (
            Path(initial_database) if initial_database is not None else None
        )
        self._max_initial_frames = max_initial_frames
        self._seed = seed

    def allocate_panels(self, fig):
        self._fig = fig
        gs = self.subspec.subgridspec(1, 2, wspace=0.35)
        self.ax_e = fig.add_subplot(gs[0])
        self.ax_f = fig.add_subplot(gs[1])

    def _initial_frames(self) -> list:
        if self._initial_database is None:
            return []
        frames = _read_all([self._initial_database])
        n_max = self._max_initial_frames
        if n_max is not None and len(frames) > n_max:
            rng = np.random.default_rng(self._seed)
            keep = np.sort(rng.choice(len(frames), n_max, replace=False))
            logger.info(
                "Subsampling the initial database to %d of %d frames.",
                n_max,
                len(frames),
            )
            frames = [frames[i] for i in keep]
        return frames

    def compute_data(self, shared_ressources: dict) -> CrossIterationErrorData:
        blocks = _data_blocks(self._initial_frames(), self._config_files)
        if not blocks:
            raise ValueError("No evaluated data to score.")
        if not self._model_files:
            raise ValueError("No trained committees to score with.")

        all_frames = [a for _, _, frames in blocks for a in frames]
        bounds = np.cumsum([0] + [len(frames) for _, _, frames in blocks])
        e_rmse, f_rmse = [], []
        for it, model_files in self._model_files.items():
            logger.info(
                "Scoring iteration %d's committee on %d frame(s) in %d "
                "block(s).",
                it,
                len(all_frames),
                len(blocks),
            )
            energies, forces = score_with_committee(model_files, all_frames)
            row_e, row_f = [], []
            for (_, _, frames), lo, hi in zip(blocks, bounds, bounds[1:]):
                e, f = _block_rmse(frames, energies[lo:hi], forces[lo:hi])
                row_e.append(e)
                row_f.append(f)
            e_rmse.append(row_e)
            f_rmse.append(row_f)

        return CrossIterationErrorData(
            model_iterations=list(self._model_files),
            block_labels=[label for label, _, _ in blocks],
            block_iterations=[it for _, it, _ in blocks],
            block_sizes=[len(frames) for _, _, frames in blocks],
            e_rmse=e_rmse,
            f_rmse=f_rmse,
        )

    def _heatmap(self, ax, matrix, title):
        d = self.data
        values = np.array(
            [[np.nan if v is None else v for v in row] for row in matrix],
            dtype=float,
        )
        im = ax.imshow(values, cmap="viridis", aspect="auto")
        self._fig.colorbar(im, ax=ax, shrink=0.8)
        vmid = np.nanmean([np.nanmin(values), np.nanmax(values)])
        for r, m in enumerate(d.model_iterations):
            for c, d_it in enumerate(d.block_iterations):
                if np.isnan(values[r, c]):
                    continue
                ax.text(
                    c,
                    r,
                    f"{values[r, c]:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    # Unseen (out-of-sample) cells in italics.
                    style="normal" if d_it <= m else "italic",
                    color="k" if values[r, c] > vmid else "w",
                )
        # Staircase between trained-on (left) and unseen (right) blocks.
        for r, m in enumerate(d.model_iterations):
            n_seen = sum(d_it <= m for d_it in d.block_iterations)
            ax.plot(
                [n_seen - 0.5, n_seen - 0.5], [r - 0.5, r + 0.5], "r-", lw=2
            )
        ax.set_xticks(range(len(d.block_labels)))
        ax.set_xticklabels(
            [
                f"{lbl}\n(n={n})"
                for lbl, n in zip(d.block_labels, d.block_sizes)
            ]
        )
        ax.set_yticks(range(len(d.model_iterations)))
        ax.set_yticklabels([f"model it{m}" for m in d.model_iterations])
        ax.set_xlabel("data block (right of red line: not trained on)")
        ax.set_title(title)

    def plot_data(self):
        self._heatmap(self.ax_e, self.data.e_rmse, "energy RMSE (meV/atom)")
        self._heatmap(self.ax_f, self.data.f_rmse, "force RMSE (eV/Å)")


class CrossIterationErrorSpec(PanelSpec):
    type: str = "cross_iteration_errors"
    # Relative to calchome; ``None`` (or a missing file) drops the block.
    initial_database: Optional[str] = "initial_database.xyz"
    max_initial_frames: Optional[int] = 300
    seed: int = 0


def _build_cross_iteration_errors(spec, ctx, common):
    if not ctx.is_summary:
        return None
    model_files = {
        c.iteration: c.model_files
        for c in ctx.iteration_contexts
        if c.model_files
    }
    config_files = {
        c.iteration: c.config_files
        for c in ctx.iteration_contexts
        if c.config_files
    }
    initial = (
        ctx.calchome / spec.initial_database
        if spec.initial_database is not None
        else None
    )
    if initial is not None and not initial.exists():
        initial = None
    if not model_files or not (config_files or initial):
        return ctx.skip(
            spec.type,
            f"models for {len(model_files)} iteration(s), evaluated data "
            f"for {len(config_files)}, initial database "
            f"{'found' if initial else 'absent'}",
        )
    return CrossIterationErrorPanel(
        model_files,
        config_files,
        initial_database=initial,
        max_initial_frames=spec.max_initial_frames,
        seed=spec.seed,
        **named(spec),
        **common,
    )


register_panel(
    "cross_iteration_errors",
    CrossIterationErrorSpec,
    _build_cross_iteration_errors,
)
