"""Species-resolved force parity (e.g. Cu/H/O), same committee re-scoring as
``committee_parity``, grouped by chemical species instead of lumped.

A lumped force-parity plot can hide a real fit-quality asymmetry between
species (e.g. metallic Cu vs. molecular H2O in this campaign) -- this panel
makes that visible directly, at essentially no extra cost over the lumped
computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.dashboard import Entry
from alviz.data import PanelData
from alviz.panels.training.parity import parity_scatter

from lammps_al.dashboard.committee_eval import score_with_committee
from lammps_al.dashboard.context import named
from lammps_al.dashboard.panels.committee_parity import _mae, _read_all, _rmse


class SpeciesParityData(PanelData):
    """Per-species force parity (reference vs. committee-predicted)."""

    species: list[str]
    f_obs: dict[str, list[float]]
    f_pred: dict[str, list[float]]
    f_rmse: dict[str, Optional[float]]
    f_mae: dict[str, Optional[float]]


class SpeciesForceParityPanel(Entry):
    """Force parity split by chemical species.

    Args:
        config_files: DFT-labelled extxyz paths (``REF_forces``).
        model_files: Committee-member potential ``.yaml`` paths to score
            with.
        species: The LAMMPS type order to report one subplot per (must match
            the campaign's ``species`` list, e.g. ``["Cu", "H", "O"]``); a
            species with no atoms across all configs gets an empty subplot
            rather than an error.
    """

    name = "species_force_parity"
    data_class = SpeciesParityData
    colspan = 2

    def __init__(self, config_files, model_files, species, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_files = [Path(p) for p in config_files]
        self._model_files = [Path(p) for p in model_files]
        self._species = list(species)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, len(self._species))
        self.axes = [
            fig.add_subplot(self.gs[i]) for i in range(len(self._species))
        ]

    def compute_data(self, shared_ressources: dict) -> SpeciesParityData:
        atoms_list = _read_all(self._config_files)
        if not atoms_list:
            raise ValueError("No configurations to score.")

        _, forces_pred = score_with_committee(
            self._model_files, atoms_list, want_forces=True
        )

        f_obs: dict = {s: [] for s in self._species}
        f_pred: dict = {s: [] for s in self._species}
        for atoms, pred in zip(atoms_list, forces_pred):
            symbols = atoms.get_chemical_symbols()
            ref = np.asarray(atoms.arrays["REF_forces"])
            for sym, ref_row, pred_row in zip(symbols, ref, pred):
                if sym not in f_obs:
                    continue
                f_obs[sym].extend(ref_row.tolist())
                f_pred[sym].extend(pred_row.tolist())

        f_rmse = {
            s: (_rmse(f_obs[s], f_pred[s]) if f_obs[s] else None)
            for s in self._species
        }
        f_mae = {
            s: (_mae(f_obs[s], f_pred[s]) if f_obs[s] else None)
            for s in self._species
        }
        return SpeciesParityData(
            species=self._species,
            f_obs=f_obs,
            f_pred=f_pred,
            f_rmse=f_rmse,
            f_mae=f_mae,
        )

    def plot_data(self):
        d = self.data
        for ax, sym in zip(self.axes, d.species):
            if not d.f_obs[sym]:
                ax.set_title(f"{sym} (no data)")
                ax.axis("off")
                continue
            parity_scatter(
                ax,
                d.f_obs[sym],
                d.f_pred[sym],
                rmse=d.f_rmse[sym],
                mae=d.f_mae[sym],
                xlabel="reference F (eV/Å)",
                ylabel="predicted F (eV/Å)",
                title=sym,
                rasterized=True,
            )


class SpeciesForceParitySpec(PanelSpec):
    """``species`` left unset inherits the run-global ``--species``."""

    type: str = "species_force_parity"
    species: Optional[list[str]] = None


def _build_species_force_parity(spec, ctx, common):
    if ctx.is_summary:
        return None
    if not (ctx.model_files and ctx.config_files):
        return ctx.skip(
            spec.type,
            f"model_files={len(ctx.model_files)}, "
            f"config_files={len(ctx.config_files)}",
        )
    if not spec.species:
        return ctx.skip(spec.type, "no species list")
    return SpeciesForceParityPanel(
        ctx.config_files,
        ctx.model_files,
        spec.species,
        **named(spec),
        **common,
    )


register_panel(
    "species_force_parity", SpeciesForceParitySpec, _build_species_force_parity
)
