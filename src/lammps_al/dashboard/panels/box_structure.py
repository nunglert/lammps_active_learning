"""Structural diagnostics for the selector's carved boxes: composition and
partial RDFs, from either the as-carved structures (``selector_output``,
*before* annealing) or the annealed ones (``annealer_output``, *after*) --
see ``run_layout.selector_output_manifest``/``run_layout.annealed_configs``.

Answers "what does this iteration's sampling actually look like" without
opening a structure viewer: how much Cu/H2O ended up in the boxes, and
whether the local structure (metal spacing, water O-O/O-H, metal-water
contact) looks physically sane -- and, via the pre-/post-anneal RDF panels,
whether annealing actually changed that structure. Pure geometry
(``structure_stats``, backed by ``matscipy``) -- no committee, no
re-scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard import run_layout
from lammps_al.dashboard.structure_stats import (
    box_composition,
    compute_partial_rdf,
)

logger = logging.getLogger(__name__)


def _load_pre_anneal_boxes(calchome, progress, iteration):
    """Read every box in one iteration's selector_output as Atoms.

    One ``(box_index, Atoms)`` pair per manifest entry -- box index is
    meaningful here (matches the manifest's own ``"index"``).
    """
    import ase.io

    manifest_path = run_layout.selector_output_manifest(
        calchome, progress, iteration
    )
    if manifest_path is None:
        return []
    manifest = json.loads(manifest_path.read_text())
    boxes = []
    for entry in manifest:
        extxyz_file = entry.get("extxyz_file")
        if not extxyz_file:
            continue
        p = manifest_path.parent / extxyz_file
        if not p.exists():
            logger.debug("Manifest entry %s: %s missing.", entry["index"], p)
            continue
        boxes.append((entry["index"], ase.io.read(p)))
    return boxes


def _load_post_anneal_boxes(calchome, progress, iteration):
    """Read one iteration's annealed boxes as Atoms.

    Unlike the pre-anneal path, this has no manifest to key against --
    ``annealer_output`` is a flat combined extxyz (see
    ``run_layout.annealed_configs``'s own docstring for why frame position
    doesn't reliably map back to a box index when boxes failed), so this
    just uses frame position as a nominal index. Fine for anything that
    pools across boxes (composition histograms, aggregate RDFs) -- which is
    everything both panels below do.
    """
    import ase.io

    path = run_layout.annealed_configs(calchome, progress, iteration)
    if path is None:
        return []
    return list(enumerate(ase.io.read(path, index=":")))


_LOADERS = {
    "pre_anneal": _load_pre_anneal_boxes,
    "post_anneal": _load_post_anneal_boxes,
}


class BoxCompositionData(PanelData):
    """Composition across one iteration's boxes, as histograms."""

    n_atoms: list[int]
    n_cu: list[int]
    n_h2o_approx: list[float]
    n_boxes: int


class BoxCompositionPanel(Entry):
    """Cu / H2O content across one iteration's carved boxes, as
    distributions (not one bar per box -- with dozens of boxes per
    iteration, a histogram over all of them is what's actually readable).

    Args:
        calchome: The campaign's ``calchome`` directory.
        progress: Normalized per-iteration progress list (from
            ``run_layout.load_progress``).
        iteration: Which iteration's carved boxes to read.
        source: ``"pre_anneal"`` (as-carved, default) or ``"post_anneal"``
            -- composition doesn't change during annealing (same atoms,
            just relaxed positions), so this only matters if you want to
            confirm that directly.
    """

    name = "box_composition"
    data_class = BoxCompositionData
    colspan = 2

    def __init__(
        self,
        calchome,
        progress,
        iteration: int,
        *args,
        source: str = "pre_anneal",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calchome = Path(calchome)
        self._progress = progress
        self._iteration = iteration
        self._source = source

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 3)
        self.ax_cu = fig.add_subplot(self.gs[0])
        self.ax_h2o = fig.add_subplot(self.gs[1])
        self.ax_total = fig.add_subplot(self.gs[2])

    def compute_data(self, shared_ressources: dict) -> BoxCompositionData:
        boxes = _LOADERS[self._source](
            self._calchome, self._progress, self._iteration
        )
        if not boxes:
            raise ValueError(
                f"No carved boxes ({self._source}) found for iteration "
                f"{self._iteration}."
            )
        n_atoms, n_cu, n_h2o = [], [], []
        for _idx, atoms in boxes:
            comp = box_composition(atoms)
            n_atoms.append(comp["n_atoms"])
            n_cu.append(comp.get("Cu", 0))
            n_h2o.append(comp["n_h2o_approx"])
        logger.info(
            "iteration %d (%s): composition read for %d box(es).",
            self._iteration,
            self._source,
            len(boxes),
        )
        return BoxCompositionData(
            n_atoms=n_atoms, n_cu=n_cu, n_h2o_approx=n_h2o, n_boxes=len(boxes)
        )

    def plot_data(self):
        d = self.data
        nbins = min(15, d.n_boxes)
        self.ax_cu.hist(d.n_cu, bins=nbins)
        self.ax_cu.set_xlabel("Cu atoms per box")
        self.ax_cu.set_ylabel("number of boxes")
        self.ax_cu.set_title(
            f"{self.name} ({self._source}, {d.n_boxes} boxes)"
        )

        self.ax_h2o.hist(d.n_h2o_approx, bins=nbins)
        self.ax_h2o.set_xlabel("H2O molecules per box (approx)")

        self.ax_total.hist(d.n_atoms, bins=nbins)
        self.ax_total.set_xlabel("atoms per box")


# species_a, species_b, cutoff (A) -- Cu/water interface campaign defaults;
# O-H in particular is a cheap intact-water-molecule sanity check (should
# peak sharply around ~1 A; a smeared/absent peak means annealing broke
# water molecules apart).
_DEFAULT_PAIRS = [
    ("Cu", "Cu", 8.0),
    ("O", "O", 8.0),
    ("O", "H", 3.0),
    ("Cu", "O", 8.0),
]


class BoxRDFData(PanelData):
    """Mean +/- percentile-band partial RDFs across one iteration's boxes."""

    r: dict[str, list[float]]  # keyed by "A-B"
    g_mean: dict[str, list[float]]
    g_low: dict[str, list[float]]
    g_high: dict[str, list[float]]
    n_boxes: dict[str, int]  # boxes actually contributing to this pair


class BoxRDFPanel(Entry):
    """Partial radial distribution functions, averaged across one
    iteration's carved boxes, via ``matscipy``.

    Args:
        calchome: The campaign's ``calchome`` directory.
        progress: Normalized per-iteration progress list (from
            ``run_layout.load_progress``).
        iteration: Which iteration's carved boxes to read.
        source: ``"pre_anneal"`` (as-carved, straight from MD sampling) or
            ``"post_anneal"`` (after the frozen-core anneal -- the
            structures that actually go to DFT). Comparing the two answers
            whether annealing is doing anything structurally: a pre-anneal
            RDF built from a raw MD snapshot can carry sampling noise/local
            defects that the anneal is meant to relax away.
        pairs: ``[(species_a, species_b, cutoff), ...]``; defaults to
            Cu-Cu/O-O/O-H/Cu-O.
        nbins: Histogram bin count per pair.
        low_pct / high_pct: Percentile band to shade.
    """

    name = "box_rdf"
    data_class = BoxRDFData
    colspan = 2

    def __init__(
        self,
        calchome,
        progress,
        iteration: int,
        *args,
        source: str = "pre_anneal",
        pairs=None,
        nbins: int = 80,
        low_pct=10,
        high_pct=90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calchome = Path(calchome)
        self._progress = progress
        self._iteration = iteration
        self._source = source
        self._pairs = pairs or _DEFAULT_PAIRS
        self._nbins = nbins
        self._low_pct = low_pct
        self._high_pct = high_pct

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, len(self._pairs))
        self.axes = [
            fig.add_subplot(self.gs[i]) for i in range(len(self._pairs))
        ]

    def compute_data(self, shared_ressources: dict) -> BoxRDFData:
        boxes = _LOADERS[self._source](
            self._calchome, self._progress, self._iteration
        )
        if not boxes:
            raise ValueError(
                f"No carved boxes ({self._source}) found for iteration "
                f"{self._iteration}."
            )

        r_out, g_mean_out, g_low_out, g_high_out, n_out = {}, {}, {}, {}, {}
        for species_a, species_b, cutoff in self._pairs:
            key = f"{species_a}-{species_b}"
            per_box_g = []
            r_centers = None
            for _idx, atoms in boxes:
                r_centers, g = compute_partial_rdf(
                    atoms, species_a, species_b, cutoff, nbins=self._nbins
                )
                if g is not None:
                    per_box_g.append(g)
            if not per_box_g:
                logger.info(
                    "iteration %d (%s), pair %s: no box had both species "
                    "-- skipping.",
                    self._iteration,
                    self._source,
                    key,
                )
                continue
            stacked = np.vstack(per_box_g)
            r_out[key] = r_centers.tolist()
            g_mean_out[key] = stacked.mean(axis=0).tolist()
            g_low_out[key] = np.percentile(
                stacked, self._low_pct, axis=0
            ).tolist()
            g_high_out[key] = np.percentile(
                stacked, self._high_pct, axis=0
            ).tolist()
            n_out[key] = len(per_box_g)

        if not r_out:
            raise ValueError(
                f"No pair had data in any box for iteration {self._iteration}"
                f" ({self._source})."
            )
        logger.info(
            "iteration %d (%s): RDFs computed for pairs %s (%d box(es) "
            "checked).",
            self._iteration,
            self._source,
            list(r_out.keys()),
            len(boxes),
        )
        return BoxRDFData(
            r=r_out,
            g_mean=g_mean_out,
            g_low=g_low_out,
            g_high=g_high_out,
            n_boxes=n_out,
        )

    def plot_data(self):
        d = self.data
        for ax, (species_a, species_b, _cutoff) in zip(self.axes, self._pairs):
            key = f"{species_a}-{species_b}"
            if key not in d.r:
                ax.set_title(f"{key} (no data)")
                ax.axis("off")
                continue
            ax.plot(d.r[key], d.g_mean[key], lw=1)
            ax.fill_between(d.r[key], d.g_low[key], d.g_high[key], alpha=0.2)
            ax.axhline(1.0, color="k", lw=0.5, ls=":")
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("g(r)")
            ax.set_title(f"{key} ({self._source}, {d.n_boxes[key]} boxes)")
