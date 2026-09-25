"""Where the selector's extracted ROIs actually sit, spatially.

Deliberately never plots the (huge, ~500-600 atom) carved structures
themselves -- just each ROI's center, read from the recorded
``selector_output`` manifest (``run_layout.selector_output_manifest``), as
a 2D scatter colored by depth. Answers "how spread out / clustered were
this iteration's ROIs" at a glance, without needing to open any structure
viewer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard import run_layout

logger = logging.getLogger(__name__)


class ROISpatialData(PanelData):
    """Per-ROI center coordinates for one iteration."""

    x: list[float]
    y: list[float]
    z: list[float]
    box_index: list[int]


class ROISpatialPanel(Entry):
    """2D scatter of ROI centers (x, y), colored by z (depth).

    A 3D scatter was the other option considered, but a static PDF panel
    can't be rotated to resolve depth ambiguity the way an interactive plot
    could -- color-by-z on a 2D projection reads unambiguously on a printed
    page instead. If ROIs cluster tightly, that's directly visible as a
    dense patch; if they spread evenly, that's visible too.

    Args:
        calchome: The campaign's ``calchome`` directory.
        progress: Normalized per-iteration progress list (from
            ``run_layout.load_progress``).
        iteration: Which iteration's ROI manifest to read.
    """

    name = "roi_spatial"
    data_class = ROISpatialData

    def __init__(self, calchome, progress, iteration: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calchome = Path(calchome)
        self._progress = progress
        self._iteration = iteration

    def allocate_panels(self, fig):
        self._fig = fig
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared_ressources: dict) -> ROISpatialData:
        manifest_path = run_layout.selector_output_manifest(
            self._calchome, self._progress, self._iteration
        )
        if manifest_path is None:
            raise ValueError(
                f"No selector manifest for iteration {self._iteration}."
            )
        manifest = json.loads(manifest_path.read_text())
        if not manifest:
            raise ValueError(
                f"Selector manifest for iteration {self._iteration} is "
                "empty (0 ROIs extracted)."
            )
        x, y, z, box_index = [], [], [], []
        for entry in manifest:
            pos = entry["roi_position"]
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            box_index.append(entry["index"])
        logger.info(
            "iteration %d: %d ROI(s) read from %s.",
            self._iteration,
            len(x),
            manifest_path,
        )
        return ROISpatialData(x=x, y=y, z=z, box_index=box_index)

    def plot_data(self):
        d = self.data
        sc = self.ax.scatter(d.x, d.y, c=d.z, cmap="viridis", s=40)
        cbar = self._fig.colorbar(sc, ax=self.ax)
        cbar.set_label("z (Å)")
        self.ax.set_xlabel("x (Å)")
        self.ax.set_ylabel("y (Å)")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title(f"{self.name} ({len(d.x)} ROIs)")
