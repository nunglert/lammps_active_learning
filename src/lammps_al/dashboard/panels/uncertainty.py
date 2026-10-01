"""Committee-disagreement (uncertainty) diagnostics across AL iterations.

All panels read ``uncertainty_select_output`` (see
``run_layout.sampler_uncertainty_trajectory``) --
``CommitteeUncertaintySelector`` annotates without reducing, so this is the
full per-atom committee-disagreement signal for every frame the sampler
produced;
``sampler_output/`` itself holds the raw, un-annotated MD trajectory. These
are full-system dumps (GBs per iteration), so every panel streams only the
columns it needs via :mod:`~lammps_al.dashboard.extxyz_columns` rather
than parsing whole ``Atoms`` with ``ase.io``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.context import named
from lammps_al.dashboard.extxyz_columns import iter_columns

logger = logging.getLogger(__name__)


class UncertaintyTrendData(PanelData):
    """Per-atom uncertainty summary statistics, one point per iteration."""

    iterations: list[int]
    mean: list[float]
    low: list[float]
    high: list[float]


class UncertaintyTrendPanel(Entry):
    """Per-atom committee disagreement vs. iteration.

    Flattens every sampled frame's ``atoms.arrays["uncertainty"]`` (across
    all atoms and frames) for each iteration and plots the mean plus a
    percentile band against iteration index -- the core "is the committee
    converging" trend line this campaign's premise rests on. If this isn't
    trending down, the AL loop isn't working regardless of what the parity
    panels say.

    Args:
        trajectories: ``{iteration: path}`` of each iteration's recorded
            uncertainty-annotated sampler trajectory (unfinished iterations
            simply absent).
        low_pct / high_pct: Percentile band to shade.
    """

    name = "uncertainty_trend"
    data_class = UncertaintyTrendData

    def __init__(
        self,
        trajectories,
        *args,
        low_pct=10,
        high_pct=90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._trajectories = dict(sorted(trajectories.items()))
        self._low_pct = low_pct
        self._high_pct = high_pct

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared_ressources: dict) -> UncertaintyTrendData:
        iterations, means, lows, highs = [], [], [], []
        for it, traj_path in self._trajectories.items():
            frames = [
                a["uncertainty"]
                for _, a in iter_columns(traj_path, ["uncertainty"])
            ]
            if not frames:
                continue
            values = np.concatenate(frames).astype(float)
            iterations.append(it)
            means.append(float(values.mean()))
            lows.append(float(np.percentile(values, self._low_pct)))
            highs.append(float(np.percentile(values, self._high_pct)))

        if not iterations:
            raise ValueError("No sampler uncertainty trajectories found.")
        return UncertaintyTrendData(
            iterations=iterations, mean=means, low=lows, high=highs
        )

    def plot_data(self):
        d = self.data
        self.ax.plot(d.iterations, d.mean, marker="o", label="mean")
        self.ax.fill_between(
            d.iterations,
            d.low,
            d.high,
            alpha=0.2,
            label=f"{self._low_pct}-{self._high_pct} pct",
        )
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("per-atom uncertainty")
        self.ax.legend()
        self.ax.set_title(self.name)


class UncertaintyDepthProfileData(PanelData):
    """Per-iteration mean uncertainty binned by depth from the interface."""

    depth_bin_centers: list[float]
    iterations: list[int]
    # Keyed by str(iteration) (JSON object keys are always strings; pydantic
    # would otherwise fail to round-trip an int-keyed dict).
    profiles: dict[str, list[Optional[float]]]


class UncertaintyDepthProfilePanel(Entry):
    """Per-atom uncertainty vs. depth from the Cu/water interface, per
    iteration.

    Per frame, the interface height is estimated as the mean z of that
    frame's ``interface_species`` atoms (the rough surface moves frame to
    frame, so a single fixed z0 would be wrong); every atom's depth is then
    ``z - z0`` (negative = inside the slab, positive = into the water).
    Directly answers this campaign's actual scientific framing -- whether
    committee disagreement concentrates at the interface vs. bulk (the
    erhard2026-style spatially-resolved framing named in ``run_al.py``'s own
    module docstring).

    Args:
        trajectories: ``{iteration: path}`` of each iteration's recorded
            uncertainty-annotated sampler trajectory.
        interface_species: Species whose mean z estimates the interface
            height (default ``("Cu",)``).
        depth_range: ``(low, high)`` depth bounds in Å, relative to the
            interface (should roughly match the campaign's
            ``ROIExtractor.box_z_below``/``box_z_above``).
        bin_width: Bin width in Å.
    """

    name = "uncertainty_depth_profile"
    data_class = UncertaintyDepthProfileData
    colspan = 2

    def __init__(
        self,
        trajectories,
        *args,
        interface_species=("Cu",),
        depth_range=(-10.0, 12.0),
        bin_width: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._trajectories = dict(sorted(trajectories.items()))
        self._interface_species = set(interface_species)
        self._depth_range = depth_range
        self._bin_width = bin_width

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def _bin_edges(self) -> np.ndarray:
        lo, hi = self._depth_range
        n_bins = max(1, int(round((hi - lo) / self._bin_width)))
        return np.linspace(lo, hi, n_bins + 1)

    def compute_data(
        self, shared_ressources: dict
    ) -> UncertaintyDepthProfileData:
        edges = self._bin_edges()
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        n_bins = len(centers)
        interface = list(self._interface_species)

        profiles: dict = {}
        used_iterations = []
        for it, traj_path in self._trajectories.items():
            # Running per-bin sums/counts: nothing per-atom is kept across
            # frames.
            sums = np.zeros(n_bins)
            counts = np.zeros(n_bins)
            for _, a in iter_columns(
                traj_path, ["species", ("pos", 2), "uncertainty"]
            ):
                z = a[("pos", 2)].astype(float)
                interface_mask = np.isin(a["species"], interface)
                if not interface_mask.any():
                    continue
                bin_idx = np.digitize(z - z[interface_mask].mean(), edges) - 1
                inside = (bin_idx >= 0) & (bin_idx < n_bins)
                sums += np.bincount(
                    bin_idx[inside],
                    weights=a["uncertainty"][inside].astype(float),
                    minlength=n_bins,
                )
                counts += np.bincount(bin_idx[inside], minlength=n_bins)

            if not counts.any():
                continue
            profiles[str(it)] = [
                float(s / c) if c else None for s, c in zip(sums, counts)
            ]
            used_iterations.append(it)

        if not used_iterations:
            raise ValueError("No sampler uncertainty trajectories found.")
        return UncertaintyDepthProfileData(
            depth_bin_centers=centers,
            iterations=used_iterations,
            profiles=profiles,
        )

    def plot_data(self):
        d = self.data
        for it in d.iterations:
            values = d.profiles[str(it)]
            xs = [
                c for c, v in zip(d.depth_bin_centers, values) if v is not None
            ]
            ys = [v for v in values if v is not None]
            self.ax.plot(xs, ys, marker="o", label=f"iter {it}")
        self.ax.axvline(0.0, color="k", lw=1, ls="--")
        self.ax.set_xlabel("depth from interface (Å, + = into water)")
        self.ax.set_ylabel("mean per-atom uncertainty")
        self.ax.legend()
        self.ax.set_title(self.name)


class UncertaintyTrendSpec(PanelSpec):
    type: str = "uncertainty_trend"
    low_pct: float = 10
    high_pct: float = 90


def _build_uncertainty_trend(spec, ctx, common):
    if not ctx.is_summary:
        return None
    if not ctx.uncertainty_trajectories:
        return ctx.skip(spec.type, "no uncertainty trajectories recorded")
    return UncertaintyTrendPanel(
        ctx.uncertainty_trajectories,
        low_pct=spec.low_pct,
        high_pct=spec.high_pct,
        **named(spec),
        **common,
    )


class UncertaintyDepthProfileSpec(PanelSpec):
    type: str = "uncertainty_depth_profile"
    interface_species: list[str] = ["Cu"]
    depth_range: tuple[float, float] = (-10.0, 12.0)
    bin_width: float = 1.0


def _build_uncertainty_depth_profile(spec, ctx, common):
    if not ctx.is_summary:
        return None
    if not ctx.uncertainty_trajectories:
        return ctx.skip(spec.type, "no uncertainty trajectories recorded")
    return UncertaintyDepthProfilePanel(
        ctx.uncertainty_trajectories,
        interface_species=spec.interface_species,
        depth_range=spec.depth_range,
        bin_width=spec.bin_width,
        **named(spec),
        **common,
    )


register_panel(
    "uncertainty_trend", UncertaintyTrendSpec, _build_uncertainty_trend
)
register_panel(
    "uncertainty_depth_profile",
    UncertaintyDepthProfileSpec,
    _build_uncertainty_depth_profile,
)


class UncertaintyTimelineData(PanelData):
    """Per-frame uncertainty summary along one sampler trajectory, plus the
    uncertainty of every extracted ROI's center atom."""

    step: list[float]
    mean: list[float]
    high: list[float]
    max: list[float]
    roi_step: list[float]
    roi_uncertainty: list[float]


class UncertaintyTimelinePanel(Entry):
    """Per-frame mean / high-percentile / max uncertainty vs. MD step, with
    the extracted ROIs marked at their source frame.

    Answers two things the campaign-level trend can't: whether the sampler
    drifts into unknown territory as the run goes on (a steadily rising max
    is a precursor to a blow-up, and argues for shorter MD), and how extreme
    the chosen ROIs are -- each ROI's center-atom uncertainty sits on the
    same axis as the per-frame max/percentile it was picked against.

    Streams only the ``uncertainty`` column, one frame at a time.

    Args:
        trajectory_path: The iteration's recorded uncertainty-annotated
            sampler trajectory.
        manifest_path: The iteration's ``selector_output`` manifest (its
            ``frame``/``roi_index`` fields locate each ROI's center atom), or
            ``None`` to plot without ROI markers.
        high_pct: The upper percentile line.
        log_y: Log-scale the uncertainty axis (max and mean typically differ
            by orders of magnitude).
    """

    name = "uncertainty_timeline"
    data_class = UncertaintyTimelineData
    colspan = 2

    def __init__(
        self,
        trajectory_path,
        *args,
        manifest_path=None,
        high_pct: float = 99,
        log_y: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._trajectory_path = Path(trajectory_path)
        self._manifest_path = (
            Path(manifest_path) if manifest_path is not None else None
        )
        self._high_pct = high_pct
        self._log_y = log_y

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared_ressources: dict) -> UncertaintyTimelineData:
        rois_by_frame: dict = {}
        if self._manifest_path is not None:
            for entry in json.loads(self._manifest_path.read_text()):
                if "frame" in entry and "roi_index" in entry:
                    rois_by_frame.setdefault(entry["frame"], []).append(
                        entry["roi_index"]
                    )

        step, mean, high, mx = [], [], [], []
        roi_step, roi_unc = [], []
        for i, (info, a) in enumerate(
            iter_columns(
                self._trajectory_path,
                ["uncertainty"],
                info_keys=("timestep",),
            )
        ):
            u = a["uncertainty"].astype(float)
            s = float(info.get("timestep", i))
            step.append(s)
            mean.append(float(u.mean()))
            high.append(float(np.percentile(u, self._high_pct)))
            mx.append(float(u.max()))
            for roi_index in rois_by_frame.get(i, []):
                roi_step.append(s)
                roi_unc.append(float(u[roi_index]))

        if not step:
            raise ValueError(f"No frames in {self._trajectory_path}.")
        logger.info(
            "%d frame(s), %d ROI(s) located, from %s.",
            len(step),
            len(roi_step),
            self._trajectory_path,
        )
        return UncertaintyTimelineData(
            step=step,
            mean=mean,
            high=high,
            max=mx,
            roi_step=roi_step,
            roi_uncertainty=roi_unc,
        )

    def plot_data(self):
        d = self.data
        self.ax.plot(d.step, d.max, lw=1, label="max")
        self.ax.plot(d.step, d.high, lw=1, label=f"p{self._high_pct:g}")
        self.ax.plot(d.step, d.mean, lw=1, label="mean")
        if d.roi_step:
            self.ax.scatter(
                d.roi_step,
                d.roi_uncertainty,
                s=40,
                marker="x",
                color="k",
                zorder=3,
                label=f"ROI centers ({len(d.roi_step)})",
            )
        if self._log_y:
            self.ax.set_yscale("log")
        self.ax.set_xlabel("MD step")
        self.ax.set_ylabel("per-atom uncertainty")
        self.ax.legend(fontsize="small")
        self.ax.set_title(self.name)


class UncertaintyTimelineSpec(PanelSpec):
    type: str = "uncertainty_timeline"
    high_pct: float = 99
    log_y: bool = True


def _build_uncertainty_timeline(spec, ctx, common):
    if ctx.is_summary:
        return None
    if ctx.uncertainty_trajectory is None:
        return ctx.skip(spec.type, "no uncertainty trajectory recorded")
    return UncertaintyTimelinePanel(
        ctx.uncertainty_trajectory,
        manifest_path=ctx.selector_manifest,
        high_pct=spec.high_pct,
        log_y=spec.log_y,
        **named(spec),
        **common,
    )


register_panel(
    "uncertainty_timeline",
    UncertaintyTimelineSpec,
    _build_uncertainty_timeline,
)
