"""Structural statistics for a carved box: composition and partial RDFs.

Pure geometry on an already-carved ``Atoms`` object -- no re-scoring, no
committee, nothing remote. Partial RDFs use ``matscipy.neighbours.
neighbour_list`` (a full neighbor list -- confirmed empirically: both
``(i, j)`` and ``(j, i)`` are returned for a pair, so no double-counting
correction is needed below) as the fast pairwise-distance backend, since
matscipy ships no ready-made RDF function itself.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def box_composition(atoms) -> dict:
    """Per-species atom counts for one carved box.

    Returns:
        ``{species: count}``, plus ``"n_atoms"`` (total) and
        ``"n_h2o_approx"`` (``n_H / 2`` -- a cheap approximation, not a real
        bonding-based molecule count, but a reasonable one for this
        campaign's Cu/water system where H is overwhelmingly water
        hydrogen).
    """
    symbols = atoms.get_chemical_symbols()
    counts: dict = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    counts["n_atoms"] = len(symbols)
    counts["n_h2o_approx"] = counts.get("H", 0) / 2.0
    return counts


def compute_partial_rdf(
    atoms, species_a: str, species_b: str, cutoff: float, nbins: int = 100
):
    """Partial radial distribution function g_AB(r) for one structure.

    Args:
        atoms: A periodic ``ase.Atoms`` (uses its own cell/pbc -- carved
            boxes are periodic, see ``geometry.carve_box``).
        species_a / species_b: Reference / neighbor species. Same species
            is fine (e.g. ``"O", "O"``).
        cutoff: Maximum r (Å).
        nbins: Histogram bin count.

    Returns:
        ``(r_centers, g_r)``, or ``(r_centers, None)`` if either species is
        absent from this structure (nothing to normalize by -- not an
        error, just "no data for this pair in this box").
    """
    from matscipy.neighbours import neighbour_list

    symbols = np.asarray(atoms.get_chemical_symbols())
    n_a = int(np.sum(symbols == species_a))
    n_b = int(np.sum(symbols == species_b))
    edges = np.linspace(0.0, cutoff, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if n_a == 0 or n_b == 0:
        logger.debug(
            "compute_partial_rdf(%s-%s): n_%s=%d, n_%s=%d -- skipping "
            "(species absent from this structure).",
            species_a,
            species_b,
            species_a,
            n_a,
            species_b,
            n_b,
        )
        return centers, None

    i, j, d = neighbour_list("ijd", atoms, cutoff=cutoff)
    mask = (symbols[i] == species_a) & (symbols[j] == species_b)
    d_ab = d[mask]

    counts, _ = np.histogram(d_ab, bins=edges)
    shell_volume = 4.0 * np.pi * centers**2 * (edges[1] - edges[0])
    number_density_b = n_b / atoms.get_volume()
    # Every A atom is its own reference point (full list, so each A->B pair
    # is already counted once per A); average over the n_a references.
    g_r = counts / (n_a * shell_volume * number_density_b)
    return centers, g_r
