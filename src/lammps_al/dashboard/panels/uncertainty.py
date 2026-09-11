"""Committee-disagreement (uncertainty) diagnostics across AL iterations.

Both panels read ``uncertainty_select_output`` (see
``run_layout.sampler_uncertainty_trajectory``) -- ``CommitteeUncertaintySelector``
annotates without reducing, so this is the full per-atom committee-
disagreement signal for every frame the sampler produced;
``sampler_output/`` itself holds the raw, un-annotated MD trajectory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard import run_layout


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
        calchome: The campaign's ``calchome`` directory.
        progress: Normalized per-iteration progress list (from
            ``run_layout.load_progress``).
        iterations: Iteration indices to include (missing/unfinished ones
            are silently skipped, not errored).
        low_pct / high_pct: Percentile band to shade.
    """

    name = "uncertainty_trend"
    data_class = UncertaintyTrendData

    def __init__(
        self,
        calchome,
        progress,
        iterations,
        *args,
        low_pct=10,
        high_pct=90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calchome = calchome
        self._progress = progress
        self._iterations = list(iterations)
        self._low_pct = low_pct
        self._high_pct = high_pct

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared_ressources: dict) -> UncertaintyTrendData:
        import ase.io

        iterations, means, lows, highs = [], [], [], []
        for it in self._iterations:
            traj_path = run_layout.sampler_uncertainty_trajectory(
                self._calchome, self._progress, it
            )
            if traj_path is None:
                continue
            values = []
            for atoms in ase.io.read(traj_path, index=":"):
                values.extend(np.asarray(atoms.arrays["uncertainty"]).ravel())
            if not values:
                continue
            values = np.asarray(values, dtype=float)
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
        calchome: The campaign's ``calchome`` directory.
        progress: Normalized per-iteration progress list (from
            ``run_layout.load_progress``).
        iterations: Iteration indices to include.
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
        calchome,
        progress,
        iterations,
        *args,
        interface_species=("Cu",),
        depth_range=(-10.0, 12.0),
        bin_width: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calchome = calchome
        self._progress = progress
        self._iterations = list(iterations)
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
        import ase.io

        edges = self._bin_edges()
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()

        profiles: dict = {}
        used_iterations = []
        for it in self._iterations:
            traj_path = run_layout.sampler_uncertainty_trajectory(
                self._calchome, self._progress, it
            )
            if traj_path is None:
                continue

            depths_all, unc_all = [], []
            for atoms in ase.io.read(traj_path, index=":"):
                symbols = np.asarray(atoms.get_chemical_symbols())
                z = atoms.positions[:, 2]
                interface_mask = np.isin(
                    symbols, list(self._interface_species)
                )
                if not interface_mask.any():
                    continue
                z0 = z[interface_mask].mean()
                depths_all.extend((z - z0).tolist())
                unc_all.extend(
                    np.asarray(atoms.arrays["uncertainty"]).ravel().tolist()
                )

            if not depths_all:
                continue

            depths_arr = np.asarray(depths_all)
            unc_arr = np.asarray(unc_all)
            bin_idx = np.digitize(depths_arr, edges) - 1

            means: list = []
            for b in range(len(centers)):
                mask = bin_idx == b
                means.append(
                    float(unc_arr[mask].mean()) if mask.any() else None
                )

            profiles[str(it)] = means
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
