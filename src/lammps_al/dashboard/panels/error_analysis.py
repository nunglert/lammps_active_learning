"""Per-atom force-error diagnostics on an iteration's *new* DFT configs,
scored with the committee that selected them (the previous iteration's).

Two questions the held-out parity plot can't answer:

* ``uncertainty_calibration`` -- does committee disagreement actually
  predict force error? Uncertainty-driven sampling is only better than
  random if it does.
* ``error_vs_roi_distance`` -- is the error concentrated at the ROI (the
  region the box was carved and annealed around) or out in the embedding
  matrix / near the carved boundary, where it would mean the DFT budget is
  being spent on embedding artifacts?

Both panels share one committee scoring pass per page via alviz's
``shared_ressources`` (scoring is the expensive part; the rest is binning).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.committee_eval import committee_force_spread
from lammps_al.dashboard.context import named
from lammps_al.dashboard.panels.committee_parity import _new_frames, _read_all

logger = logging.getLogger(__name__)


def _scored_new_configs(shared, config_files, prev_config_files, model_files):
    """``(atoms_list, forces, forces_std)`` for the new configs, computed
    once per page and shared between this module's panels."""
    key = (
        "committee_force_spread",
        tuple(map(str, config_files)),
        tuple(map(str, prev_config_files)),
        tuple(map(str, model_files)),
    )
    if key not in shared:
        atoms_list = _new_frames(
            _read_all(config_files), _read_all(prev_config_files)
        )
        if not atoms_list:
            raise ValueError("No new evaluator configs; nothing to score.")
        forces, forces_std = committee_force_spread(model_files, atoms_list)
        shared[key] = (atoms_list, forces, forces_std)
    return shared[key]


def _force_error(atoms, forces) -> np.ndarray:
    """Per-atom ``|F_committee - F_ref|`` (eV/Å)."""
    return np.linalg.norm(forces - atoms.arrays["REF_forces"], axis=1)


def _spearman(a, b) -> float:
    """Spearman rank correlation (no scipy dependency for one number)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


class _NewConfigErrorPanel(Entry):
    """Shared constructor for panels scoring an iteration's new configs."""

    def __init__(
        self, config_files, prev_config_files, model_files, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._config_files = [Path(p) for p in config_files]
        self._prev_config_files = [Path(p) for p in prev_config_files]
        self._model_files = [Path(p) for p in model_files]

    def _scored(self, shared):
        return _scored_new_configs(
            shared,
            self._config_files,
            self._prev_config_files,
            self._model_files,
        )


class UncertaintyCalibrationData(PanelData):
    uncertainty: list[float]
    error: list[float]
    symbols: list[str]
    bin_uncertainty: list[float]
    bin_error_median: list[float]
    spearman: float


class UncertaintyCalibrationPanel(_NewConfigErrorPanel):
    """Per-atom committee uncertainty vs. actual force error, log-log.

    Uncertainty is aggregated exactly as the sampler-side annotation does
    (``committee_selection.aggregate_uncertainties`` over
    ``cutoff_radius``), so the x-axis is the same quantity ROIs were
    selected on. A rising binned-median line / clearly positive Spearman
    rho means the selection signal is informative; a flat cloud means
    uncertainty-driven selection is no better than random.

    Args:
        config_files / prev_config_files: The iteration's evaluated configs
            and the previous iteration's (to drop already-seen frames).
        model_files: The committee that selected these configs.
        cutoff_radius: Aggregation radius (Å); match the selector's
            ``cutoff_radius``.
        nbins: Number of equal-population uncertainty bins for the median
            line.
    """

    name = "uncertainty_calibration"
    data_class = UncertaintyCalibrationData

    def __init__(
        self,
        config_files,
        prev_config_files,
        model_files,
        *args,
        cutoff_radius: float = 4.0,
        nbins: int = 10,
        **kwargs,
    ):
        super().__init__(
            config_files, prev_config_files, model_files, *args, **kwargs
        )
        self._cutoff_radius = cutoff_radius
        self._nbins = nbins

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared_ressources):
        from lammps_al.select.scripts.committee_selection import (
            aggregate_uncertainties,
        )

        atoms_list, forces, forces_std = self._scored(shared_ressources)
        unc, err, symbols = [], [], []
        for atoms, f, f_std in zip(atoms_list, forces, forces_std):
            unc.append(
                aggregate_uncertainties(
                    atom_pos=atoms.get_positions(),
                    unit_cell=atoms.cell,
                    boundary_conditions=atoms.pbc,
                    atomic_uncertainties=f_std,
                    r_cut=self._cutoff_radius,
                )
            )
            err.append(_force_error(atoms, f))
            symbols.extend(atoms.get_chemical_symbols())
        unc = np.concatenate(unc)
        err = np.concatenate(err)

        # Equal-population bins: robust to the heavy uncertainty tail.
        order = np.argsort(unc)
        chunks = np.array_split(order, min(self._nbins, len(order)))
        bin_unc = [float(np.median(unc[c])) for c in chunks if len(c)]
        bin_err = [float(np.median(err[c])) for c in chunks if len(c)]
        rho = _spearman(unc, err)
        logger.info(
            "Calibration over %d atom(s) in %d config(s): Spearman rho=%.2f.",
            len(unc),
            len(atoms_list),
            rho,
        )
        return UncertaintyCalibrationData(
            uncertainty=unc.tolist(),
            error=err.tolist(),
            symbols=symbols,
            bin_uncertainty=bin_unc,
            bin_error_median=bin_err,
            spearman=rho,
        )

    def plot_data(self):
        d = self.data
        unc = np.asarray(d.uncertainty)
        err = np.asarray(d.error)
        symbols = np.asarray(d.symbols)
        # Species in order of first appearance -- fixed per dataset, so the
        # same species keeps its color across this panel's pages.
        for sym in dict.fromkeys(d.symbols):
            mask = symbols == sym
            self.ax.scatter(
                unc[mask], err[mask], s=4, alpha=0.3, lw=0, label=sym
            )
        self.ax.plot(
            d.bin_uncertainty,
            d.bin_error_median,
            color="k",
            lw=2,
            marker="o",
            ms=8,
            label="binned median",
        )
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("aggregated committee uncertainty (eV/Å)")
        self.ax.set_ylabel("|F_committee - F_DFT| (eV/Å)")
        self.ax.legend(fontsize="small", markerscale=2)
        self.ax.set_title(f"{self.name} (Spearman ρ = {d.spearman:.2f})")


class ErrorVsROIDistanceData(PanelData):
    bin_centers: list[float]
    error_median: list[float | None]
    error_high: list[float | None]
    spread_median: list[float | None]
    counts: list[int]
    core_radius: float


class ErrorVsROIDistancePanel(_NewConfigErrorPanel):
    """Per-atom force error and committee spread vs. distance from the ROI
    center, binned, with the frozen-core radius marked.

    The ROI center is the selector manifest's ``roi_position`` -- the box
    coordinate the ROI was carved around, which the frozen core keeps fixed
    through annealing and DFT. Distances use the minimum-image convention
    (annealed positions are not wrapped back into the cell).

    Args:
        config_files / prev_config_files / model_files: As
            :class:`UncertaintyCalibrationPanel`.
        manifest_path: The iteration's ``selector_output`` manifest.
        core_radius: Frozen-core radius (Å) to mark; match
            ``ROIExtractor.protect_radius`` / ``EmbeddingAnnealer.
            anneal_cutoff``.
        bin_width: Distance bin width (Å).
        high_pct: Upper percentile of the error band.
    """

    name = "error_vs_roi_distance"
    data_class = ErrorVsROIDistanceData
    colspan = 2

    def __init__(
        self,
        config_files,
        prev_config_files,
        model_files,
        manifest_path,
        *args,
        core_radius: float = 4.0,
        bin_width: float = 1.0,
        high_pct: float = 90,
        **kwargs,
    ):
        super().__init__(
            config_files, prev_config_files, model_files, *args, **kwargs
        )
        self._manifest_path = Path(manifest_path)
        self._core_radius = core_radius
        self._bin_width = bin_width
        self._high_pct = high_pct

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def _roi_center(self) -> np.ndarray:
        positions = np.array(
            [
                e["roi_position"]
                for e in json.loads(self._manifest_path.read_text())
            ]
        )
        center = np.median(positions, axis=0)
        spread = float(np.abs(positions - center).max())
        if spread > 0.5:
            logger.warning(
                "ROI box centers in %s differ by up to %.2f Å; using their "
                "median %s for every config.",
                self._manifest_path,
                spread,
                center,
            )
        return center

    def compute_data(self, shared_ressources):
        from ase.geometry import find_mic

        center = self._roi_center()
        atoms_list, forces, forces_std = self._scored(shared_ressources)
        dist, err, spread = [], [], []
        for atoms, f, f_std in zip(atoms_list, forces, forces_std):
            _, d = find_mic(atoms.positions - center, atoms.cell, atoms.pbc)
            dist.append(d)
            err.append(_force_error(atoms, f))
            spread.append(np.linalg.norm(f_std, axis=1))
        dist = np.concatenate(dist)
        err = np.concatenate(err)
        spread = np.concatenate(spread)

        # Enough bins that the farthest atom falls inside the last one.
        n_bins = int(dist.max() // self._bin_width) + 1
        edges = self._bin_width * np.arange(n_bins + 1)
        idx = np.digitize(dist, edges) - 1
        centers, e_med, e_high, s_med, counts = [], [], [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            centers.append(float((edges[b] + edges[b + 1]) / 2))
            counts.append(int(m.sum()))
            if m.any():
                e_med.append(float(np.median(err[m])))
                e_high.append(float(np.percentile(err[m], self._high_pct)))
                s_med.append(float(np.median(spread[m])))
            else:
                e_med.append(None)
                e_high.append(None)
                s_med.append(None)
        return ErrorVsROIDistanceData(
            bin_centers=centers,
            error_median=e_med,
            error_high=e_high,
            spread_median=s_med,
            counts=counts,
            core_radius=self._core_radius,
        )

    def plot_data(self):
        d = self.data
        keep = [i for i, v in enumerate(d.error_median) if v is not None]
        x = [d.bin_centers[i] for i in keep]
        self.ax.plot(
            x,
            [d.error_median[i] for i in keep],
            lw=2,
            marker="o",
            ms=8,
            label="force error, median",
        )
        self.ax.fill_between(
            x,
            [d.error_median[i] for i in keep],
            [d.error_high[i] for i in keep],
            alpha=0.2,
            label=f"force error, median-p{self._high_pct:g}",
        )
        self.ax.plot(
            x,
            [d.spread_median[i] for i in keep],
            lw=2,
            ls="--",
            marker="s",
            ms=8,
            label="committee spread, median",
        )
        self.ax.axvline(d.core_radius, color="k", lw=1, ls=":")
        self.ax.annotate(
            "frozen core",
            xy=(d.core_radius, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(-4, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize="small",
        )
        self.ax.set_xlabel("distance from ROI center (Å)")
        self.ax.set_ylabel("force (eV/Å)")
        self.ax.set_ylim(bottom=0)
        self.ax.legend(fontsize="small")
        self.ax.set_title(self.name)


def _gate(spec, ctx):
    """The shared input check; returns a skip reason or ``None``."""
    if not (ctx.prev_model_files and ctx.config_files):
        return (
            f"previous iteration's model_files={len(ctx.prev_model_files)},"
            f" config_files={len(ctx.config_files)}"
        )
    return None


class UncertaintyCalibrationSpec(PanelSpec):
    type: str = "uncertainty_calibration"
    cutoff_radius: float = 4.0
    nbins: int = 10


def _build_uncertainty_calibration(spec, ctx, common):
    if ctx.is_summary:
        return None
    reason = _gate(spec, ctx)
    if reason:
        return ctx.skip(spec.type, reason)
    return UncertaintyCalibrationPanel(
        ctx.config_files,
        ctx.prev_config_files,
        ctx.prev_model_files,
        cutoff_radius=spec.cutoff_radius,
        nbins=spec.nbins,
        **named(spec),
        **common,
    )


class ErrorVsROIDistanceSpec(PanelSpec):
    type: str = "error_vs_roi_distance"
    core_radius: float = 4.0
    bin_width: float = 1.0
    high_pct: float = 90


def _build_error_vs_roi_distance(spec, ctx, common):
    if ctx.is_summary:
        return None
    reason = _gate(spec, ctx)
    if reason:
        return ctx.skip(spec.type, reason)
    if ctx.selector_manifest is None:
        return ctx.skip(spec.type, "no selector_output recorded")
    return ErrorVsROIDistancePanel(
        ctx.config_files,
        ctx.prev_config_files,
        ctx.prev_model_files,
        ctx.selector_manifest,
        core_radius=spec.core_radius,
        bin_width=spec.bin_width,
        high_pct=spec.high_pct,
        **named(spec),
        **common,
    )


register_panel(
    "uncertainty_calibration",
    UncertaintyCalibrationSpec,
    _build_uncertainty_calibration,
)
register_panel(
    "error_vs_roi_distance",
    ErrorVsROIDistanceSpec,
    _build_error_vs_roi_distance,
)
