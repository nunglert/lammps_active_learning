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
    atoms: Atoms, threshold: float, cluster_cutoff: float
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

    Returns:
        Sorted list of atom indices (into ``atoms``), one per cluster, each
        the local uncertainty maximum of its cluster.
    """
    uncertainty = atoms.arrays["uncertainty"]
    indices = np.where(uncertainty > threshold)[0]
    if len(indices) == 0:
        return []

    sub = atoms[indices]
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
