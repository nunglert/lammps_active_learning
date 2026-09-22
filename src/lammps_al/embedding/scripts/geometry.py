"""ROI clustering and box carving for amorphous matrix embedding.

Pure numpy/ASE -- no OVITO import here, so this module is safe to import on
the host (e.g. from tests) without the ``embedding`` extra installed.
Ports the clustering and carve/rotate/crop logic of the group's
``extract_snapshots.py`` script onto ``ase.neighborlist`` (replacing
``asap3.FullNeighborList``) and a union-find for cluster reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


@dataclass
class BoxDims:
    """Half-extents of the box carved around a region of interest (ROI).

    ``z_below``/``z_above`` are independent so headspace can be added
    asymmetrically above the ROI (e.g. extra solvent above a substrate),
    mirroring the ``boxsize_z``/``boxsize_z_water`` split in the original
    extraction script. ``z_above`` defaults to ``z_below`` (a symmetric box)
    when not given.
    """

    x: float
    y: float
    z_below: float
    z_above: Optional[float] = None

    def __post_init__(self) -> None:
        if self.z_above is None:
            self.z_above = self.z_below


def find_rois(
    atoms: Atoms,
    threshold: float,
    cluster_cutoff: float,
    phase_species: Optional[list[str]] = None,
    phase_cutoff: Optional[float] = None,
) -> list[int]:
    """Cluster high-uncertainty atoms and return one peak index per cluster.

    Selects atoms with ``atoms.arrays["uncertainty"] > threshold``, groups
    them into connected components under ``cluster_cutoff`` (a periodic
    neighbor list restricted to the selected atoms), and returns -- per
    component -- the index of the atom with the highest uncertainty.

    Args:
        atoms: Structure carrying a per-atom ``uncertainty`` array.
        threshold: Minimum uncertainty for an atom to seed a cluster.
        cluster_cutoff: Distance (Angstrom) under which two high-uncertainty
            atoms are considered part of the same region of interest.
        phase_species: When given (together with ``phase_cutoff``),
            restricts candidates to high-uncertainty atoms within
            ``phase_cutoff`` of at least one atom of these species --
            keeps an interface-oriented extraction (carved boxes need a
            condensed-phase core for ``EmbeddingAnnealer``'s frozen-sphere
            anneal) from picking ROIs deep in a bulk single-phase region
            with no such core nearby (confirmed against a real campaign
            trajectory, 2026-09-22: an uncalibrated ``threshold`` let
            through ROIs 50+ A from the nearest condensed-phase atom).
            ``None`` (with ``phase_cutoff``) applies no filter -- the
            right choice for ``topology_mode="bulk"``, where there's no
            separate phase to be near.
        phase_cutoff: Distance (Angstrom) for the ``phase_species`` filter
            above. Required together with ``phase_species``; ignored (with
            a warning) if given alone.

    Returns:
        Sorted list of atom indices (into ``atoms``), one per cluster, each
        the local uncertainty maximum of its cluster.
    """
    uncertainty = atoms.arrays["uncertainty"]
    indices = np.where(uncertainty > threshold)[0]
    if len(indices) == 0:
        return []

    if phase_species is not None and phase_cutoff is not None:
        symbols = np.asarray(atoms.get_chemical_symbols())
        phase_indices = np.where(np.isin(symbols, phase_species))[0]
        if len(phase_indices) == 0:
            return []
        near = _near_any(atoms, indices, phase_indices, phase_cutoff)
        indices = indices[near]
        if len(indices) == 0:
            return []
    elif phase_species is not None or phase_cutoff is not None:
        import warnings

        warnings.warn(
            "find_rois: phase_species and phase_cutoff must be given "
            "together to apply the proximity filter -- got only one, "
            "applying no filter."
        )

    sub = atoms[indices]
    try:
        from matscipy.neighbours import neighbour_list

        i_idx, j_idx = neighbour_list("ij", sub, cluster_cutoff)
    except ImportError:
        i_idx, j_idx = neighbor_list("ij", sub, cluster_cutoff)

    parent = list(range(len(indices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in zip(i_idx, j_idx):
        union(int(a), int(b))

    clusters: dict[int, list[int]] = {}
    for local_idx in range(len(indices)):
        clusters.setdefault(find(local_idx), []).append(local_idx)

    rois = []
    for members in clusters.values():
        local_unc = uncertainty[indices[members]]
        peak_local = members[int(np.argmax(local_unc))]
        rois.append(int(indices[peak_local]))
    return sorted(rois)


def cap_rois_by_uncertainty(
    candidates: list[tuple[int, int, float]],
    max_num_structures: Optional[int],
) -> list[tuple[int, int, float]]:
    """Keep only the ``max_num_structures`` highest-uncertainty ROIs.

    ``candidates`` is ``(frame_index, roi_index, uncertainty)`` per ROI,
    gathered across every frame of a trajectory (one :func:`find_rois` call
    per frame). Ranks by ``uncertainty`` (not frame order or discovery
    order) so a cap keeps the genuinely most informative configs regardless
    of where in the trajectory they happen to fall -- an early-frames-first
    truncation would systematically bias toward whatever a trajectory
    happens to start with. The result is re-sorted back into
    ``(frame_index, roi_index)`` order, matching ``find_rois``' own
    convention, so a capped run's manifest/logging still reads in trajectory
    order rather than uncertainty order.

    ``max_num_structures=None``, or a ``candidates`` list already at or
    under the cap, returns ``candidates`` unchanged (as a new list).
    """
    if max_num_structures is None or len(candidates) <= max_num_structures:
        return list(candidates)
    kept = sorted(candidates, key=lambda c: c[2], reverse=True)
    kept = kept[:max_num_structures]
    return sorted(kept, key=lambda c: (c[0], c[1]))


def _near_any(
    atoms: Atoms,
    candidate_indices: np.ndarray,
    target_indices: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Boolean mask (aligned with ``candidate_indices``): whether each
    candidate atom has a ``target_indices`` atom within ``cutoff``,
    periodic-aware.

    Builds a small combined sub-structure (candidates + targets) rather
    than an all-pairs distance matrix over the full (possibly
    ~200k-atom) ``atoms`` -- the latter is exactly the O(n_atoms^2)
    memory blowup already found and fixed elsewhere in this codebase
    (``committee_selection.aggregate_uncertainties``). A candidate that is
    itself a target atom (e.g. a condensed-phase atom whose own
    uncertainty exceeds threshold) is trivially "near" via the resulting
    zero-distance pair between its two copies in the combined structure --
    no special-casing needed.

    Prefers matscipy's neighbor list (confirmed ~13x faster, and far less
    memory-hungry, than ase's own at real campaign scale -- see
    ``committee_selection._tp_atoms_from_matscipy``'s docstring; confirmed
    directly here too: ase's own neighbor list needed ~100GB/83s for one
    real ~270k-atom candidates+targets structure at cutoff=6.0, vs. a few
    seconds/GB expected from matscipy), falling back to
    ``ase.neighborlist`` when matscipy isn't installed -- unlike the
    ``ace`` backend's own neighbor list, this isn't guaranteed present
    (only via ``[ace_tensorpot_gpu]``, not ``[embedding]``/``[all]``).
    """
    combined = atoms[np.concatenate([candidate_indices, target_indices])]
    n_cand = len(candidate_indices)
    try:
        from matscipy.neighbours import neighbour_list

        i_idx, j_idx = neighbour_list("ij", combined, cutoff)
    except ImportError:
        i_idx, j_idx = neighbor_list("ij", combined, cutoff)

    near = np.zeros(n_cand, dtype=bool)
    from_cand = i_idx < n_cand
    to_target = j_idx >= n_cand
    near[i_idx[from_cand & to_target]] = True
    from_cand = j_idx < n_cand
    to_target = i_idx >= n_cand
    near[j_idx[from_cand & to_target]] = True
    return near


def _periodic_neighborhood(
    atoms: Atoms, center_pos: np.ndarray, reach: float
) -> tuple[np.ndarray, np.ndarray]:
    """Every periodic image of every atom within ``reach`` of ``center_pos``.

    A single ``atoms.wrap()`` only ever yields *one* image per atom (the one
    inside the primary cell) -- fine as long as the region you crop out
    afterwards is no bigger than half the cell in every direction, but once
    the requested box (or its rotation onto the surface normal) needs
    material from beyond that, a plain wrap silently leaves holes: the
    result looks torn along whatever cell edge/rotation angle the missing
    images fall on, rather than raising an error. Building an explicit set
    of periodic replicas wide enough to cover the box's circumscribing
    sphere up front avoids that regardless of ``box_dims``/``normal``.

    Returns unwrapped, real-space (positions, atomic numbers) -- not an
    ``Atoms`` object, since duplicated images cannot coexist as distinct
    atoms in a single non-replicated cell.
    """
    cell = np.asarray(atoms.get_cell())
    pbc = atoms.get_pbc()
    lengths = np.linalg.norm(cell, axis=1)

    reps = [
        int(np.ceil(reach / length)) if periodic and length > 0 else 0
        for length, periodic in zip(lengths, pbc)
    ]
    ranges = [range(-r, r + 1) if r else [0] for r in reps]
    offsets = np.array(
        [
            i * cell[0] + j * cell[1] + k * cell[2]
            for i in ranges[0]
            for j in ranges[1]
            for k in ranges[2]
        ]
    )

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    all_positions = (positions[None, :, :] + offsets[:, None, :]).reshape(
        -1, 3
    )
    all_numbers = np.tile(numbers, len(offsets))

    mask = np.linalg.norm(all_positions - center_pos, axis=1) <= reach
    return all_positions[mask], all_numbers[mask]


def carve_box(
    atoms: Atoms,
    center_index: int,
    box_dims: BoxDims,
    buffer: float = 0.0,
    normal: Optional[np.ndarray] = None,
) -> tuple[Atoms, np.ndarray]:
    """Cut a box out of ``atoms`` centered on ``center_index``.

    Gathers every periodic image within the box's circumscribing sphere of
    the ROI (see :func:`_periodic_neighborhood` -- this is what lets the
    crop below be correct even when ``box_dims`` exceeds half the original
    cell, or after rotation skews which images are needed), optionally
    rotates ``normal`` onto ``+z`` (for ``surface``/``interface`` topology
    modes -- pass ``None`` for ``bulk``), crops to ``box_dims``, and sets a
    new orthorhombic cell of size ``2*box_dims.x + buffer`` x
    ``2*box_dims.y + buffer`` x ``box_dims.z_below + box_dims.z_above +
    buffer``.

    Args:
        atoms: The source structure (not mutated -- a copy is carved).
        center_index: Index of the ROI atom in ``atoms``.
        box_dims: Half-extents of the box to keep around the ROI.
        buffer: Extra padding added to each cell dimension.
        normal: Unit vector (or any nonzero vector) to rotate onto ``+z``
            before cropping. ``None`` skips rotation (bulk mode).

    Returns:
        ``(carved_atoms, roi_position)`` -- the carved structure and the
        ROI atom's position in the new cell. The ROI's position is
        deterministic by construction (the carve is centered on it), so it
        is returned directly rather than requiring index-tracking through
        the atom deletions.
    """
    center_pos = atoms.get_positions()[center_index]
    reach = (
        np.sqrt(
            box_dims.x**2
            + box_dims.y**2
            + max(box_dims.z_below, box_dims.z_above) ** 2
        )
        + buffer
        + 1e-6
    )
    positions, numbers = _periodic_neighborhood(atoms, center_pos, reach)
    atoms = Atoms(numbers=numbers, positions=positions)
    center = center_pos.copy()

    if normal is not None:
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        z_axis = np.array([0.0, 0.0, 1.0])
        cos_angle = np.clip(np.dot(normal, z_axis), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        axis = np.cross(normal, z_axis)
        if np.linalg.norm(axis) > 1e-8:
            atoms.rotate(a=angle, v=axis, center=center)

    shift_z = 0.5 * (box_dims.z_above - box_dims.z_below)
    z_half = 0.5 * (box_dims.z_above + box_dims.z_below)

    positions = atoms.get_positions()
    keep = (
        (np.abs(positions[:, 2] - center[2] - shift_z) < z_half)
        & (np.abs(positions[:, 1] - center[1]) < box_dims.y)
        & (np.abs(positions[:, 0] - center[0]) < box_dims.x)
    )
    del atoms[np.where(~keep)[0]]

    cell_size = np.array(
        [
            2 * box_dims.x + buffer,
            2 * box_dims.y + buffer,
            box_dims.z_below + box_dims.z_above + buffer,
        ]
    )
    shift = cell_size / 2 - center - np.array([0.0, 0.0, shift_z])
    atoms.translate(shift)
    atoms.set_cell(cell_size, scale_atoms=False)
    atoms.set_pbc(True)

    roi_position = cell_size / 2 - np.array([0.0, 0.0, shift_z])
    return atoms, roi_position
