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
from typing import Literal, Optional

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData

from lammps_al.dashboard.structure_stats import (
    box_composition,
    compute_partial_rdf,
)

logger = logging.getLogger(__name__)


def _load_pre_anneal_boxes(manifest_path):
    """Read every box listed in a selector_output manifest as Atoms.

    One ``(box_index, Atoms)`` pair per manifest entry -- box index is
    meaningful here (matches the manifest's own ``"index"``).
    """
    import ase.io

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


def _load_post_anneal_boxes(path):
    """Read one iteration's annealed boxes (combined extxyz) as Atoms.

    Unlike the pre-anneal path, this has no manifest to key against --
    ``annealer_output`` is a flat combined extxyz (see
    ``run_layout.annealed_configs``'s own docstring for why frame position
    doesn't reliably map back to a box index when boxes failed), so this
    just uses frame position as a nominal index. Fine for anything that
    pools across boxes (composition histograms, aggregate RDFs) -- which is
    everything both panels below do.
    """
    import ase.io

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
        boxes_path: The ``selector_output`` manifest (``source=
            "pre_anneal"``) or the combined annealed extxyz (``source=
            "post_anneal"``).
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
        boxes_path,
        *args,
        source: str = "pre_anneal",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._boxes_path = Path(boxes_path)
        self._source = source

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 3)
        self.ax_cu = fig.add_subplot(self.gs[0])
        self.ax_h2o = fig.add_subplot(self.gs[1])
        self.ax_total = fig.add_subplot(self.gs[2])

    def compute_data(self, shared_ressources: dict) -> BoxCompositionData:
        boxes = _LOADERS[self._source](self._boxes_path)
        if not boxes:
            raise ValueError(
                f"No carved boxes ({self._source}) found in "
                f"{self._boxes_path}."
            )
        n_atoms, n_cu, n_h2o = [], [], []
        for _idx, atoms in boxes:
            comp = box_composition(atoms)
            n_atoms.append(comp["n_atoms"])
            n_cu.append(comp.get("Cu", 0))
            n_h2o.append(comp["n_h2o_approx"])
        logger.info(
            "%s: composition read for %d box(es).",
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
        boxes_path: The ``selector_output`` manifest (``source=
            "pre_anneal"``) or the combined annealed extxyz (``source=
            "post_anneal"``).
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
        boxes_path,
        *args,
        source: str = "pre_anneal",
        pairs=None,
        nbins: int = 80,
        low_pct=10,
        high_pct=90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._boxes_path = Path(boxes_path)
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
        boxes = _LOADERS[self._source](self._boxes_path)
        if not boxes:
            raise ValueError(
                f"No carved boxes ({self._source}) found in "
                f"{self._boxes_path}."
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
                    "%s, pair %s: no box had both species -- skipping.",
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
                f"No pair had data in any box of {self._boxes_path} "
                f"({self._source})."
            )
        logger.info(
            "%s: RDFs computed for pairs %s (%d box(es) checked).",
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


def _boxes_path(spec, ctx):
    """The ``source``'s input path for this page, or ``None`` (logged)."""
    if spec.source == "pre_anneal":
        if ctx.selector_manifest is None:
            return ctx.skip(spec.type, "no selector_output recorded")
        return ctx.selector_manifest
    if ctx.annealed_configs is None:
        return ctx.skip(spec.type, "no annealer_output recorded")
    return ctx.annealed_configs


def _box_name(spec, base):
    """Param-keyed default name -- ``Entry.name`` doubles as the JSON cache
    filename, so a pre- and post-anneal instance must not share one."""
    if spec.name:
        return spec.name
    return base if spec.source == "pre_anneal" else f"{base}_{spec.source}"


class BoxCompositionSpec(PanelSpec):
    type: str = "box_composition"
    source: Literal["pre_anneal", "post_anneal"] = "pre_anneal"


def _build_box_composition(spec, ctx, common):
    if ctx.is_summary:
        return None
    path = _boxes_path(spec, ctx)
    if path is None:
        return None
    return BoxCompositionPanel(
        path,
        source=spec.source,
        name=_box_name(spec, BoxCompositionPanel.name),
        **common,
    )


class BoxRDFSpec(PanelSpec):
    """``pairs``: ``[[species_a, species_b, cutoff], ...]``; unset uses
    Cu-Cu/O-O/O-H/Cu-O."""

    type: str = "box_rdf"
    source: Literal["pre_anneal", "post_anneal"] = "pre_anneal"
    pairs: Optional[list[tuple[str, str, float]]] = None
    nbins: int = 80
    low_pct: float = 10
    high_pct: float = 90


def _build_box_rdf(spec, ctx, common):
    if ctx.is_summary:
        return None
    path = _boxes_path(spec, ctx)
    if path is None:
        return None
    return BoxRDFPanel(
        path,
        source=spec.source,
        pairs=spec.pairs,
        nbins=spec.nbins,
        low_pct=spec.low_pct,
        high_pct=spec.high_pct,
        name=_box_name(spec, BoxRDFPanel.name),
        **common,
    )


register_panel("box_composition", BoxCompositionSpec, _build_box_composition)
register_panel("box_rdf", BoxRDFSpec, _build_box_rdf)
