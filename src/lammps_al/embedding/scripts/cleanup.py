"""Cleanup strategies for a carved amorphous-matrix box.

Three ``cleanup_mode`` strategies behind one dispatcher (:func:`cleanup`),
all of which respect a ``protect_radius`` around the region of interest
(ROI): atoms inside it are never added, removed, or otherwise touched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atom, Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


def _protected(
    atoms: Atoms, roi_center: np.ndarray, protect_radius: float
) -> np.ndarray:
    """Boolean mask, True for atoms within ``protect_radius`` of the ROI."""
    d = atoms.get_positions() - np.asarray(roi_center)
    return np.linalg.norm(d, axis=1) < protect_radius


def cleanup_none(atoms: Atoms, **_kwargs) -> Atoms:
    """The "basic" mode: return the carved box unchanged."""
    return atoms.copy()


def _bond_cutoff(z1: int, z2: int, tol: float) -> float:
    return tol * (covalent_radii[z1] + covalent_radii[z2])


def _template_topology(template: Atoms) -> Optional[dict]:
    """Classify a solvent template as a star graph: one central atom bonded
    to N equivalent ligand atoms of a single species (e.g. water: O + 2 H).

    Returns a dict describing the topology, or ``None`` if the template is
    not a star graph -- callers then fall back to delete-only handling for
    that fragment.
    """
    n = len(template)
    if n < 2:
        return None
    positions = template.get_positions()
    numbers = template.get_atomic_numbers()
    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    bonded = [
        [
            j
            for j in range(n)
            if j != i
            and dists[i, j] < _bond_cutoff(numbers[i], numbers[j], 1.3)
        ]
        for i in range(n)
    ]
    degrees = [len(b) for b in bonded]
    central_candidates = [i for i in range(n) if degrees[i] == n - 1]
    if len(central_candidates) != 1:
        return None
    central = central_candidates[0]
    ligands = [i for i in range(n) if i != central]
    if not all(degrees[i] == 1 for i in ligands):
        return None
    ligand_symbols = {template[i].symbol for i in ligands}
    if len(ligand_symbols) != 1:
        return None
    ligand_symbol = ligand_symbols.pop()
    bond_length = float(np.mean([dists[central, j] for j in ligands]))
    bond_angle = None
    if len(ligands) >= 2:
        v0 = positions[ligands[0]] - positions[central]
        v1 = positions[ligands[1]] - positions[central]
        cos_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
        bond_angle = float(
            np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        )
    return {
        "central_symbol": template[central].symbol,
        "ligand_symbol": ligand_symbol,
        "coordination": len(ligands),
        "bond_length": bond_length,
        "bond_angle": bond_angle,
    }


def cleanup_molecule(
    atoms: Atoms,
    roi_center: np.ndarray,
    protect_radius: float,
    solvent: Atoms,
    *,
    bond_tol: float = 1.3,
    overlap_tol: float = 0.7,
    max_placement_tries: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> Atoms:
    """Reform molecules broken by the carve, using ``solvent`` as a template.

    Repair is implemented for the "star" solvent topology with exactly two
    ligands (e.g. water: O + 2 H) -- this generalizes the group's original
    water-repair logic (bond length/angle now come from the template itself
    rather than a hardcoded 104.5 deg / 0.96 A) to any two-ligand template.
    Other coordinations, and central atoms missing more than one ligand, are
    out of scope for automatic placement (the geometry is ambiguous) and the
    whole fragment is dropped instead of guessing. Ligand atoms left without
    a bonded central atom are deleted as danglers.

    Atoms within ``protect_radius`` of ``roi_center`` are never modified.

    Args:
        atoms: The carved box to repair (not mutated -- a copy is returned).
        roi_center: Position of the region of interest.
        protect_radius: Radius around ``roi_center`` that is never touched.
        solvent: An ASE ``Atoms`` template for one solvent molecule.
        bond_tol: Multiplier on the sum of covalent radii used as the
            central-ligand bonding cutoff.
        overlap_tol: Multiplier on ``bond_length`` below which a candidate
            placement is rejected as a steric clash.
        max_placement_tries: Random-azimuth attempts before giving up and
            dropping the fragment.
        rng: Optional random generator (for reproducible placement).
    """
    if rng is None:
        rng = np.random.default_rng()
    atoms = atoms.copy()
    topo = _template_topology(solvent)
    protected = _protected(atoms, roi_center, protect_radius)

    if topo is None:
        # Non-star / unsupported topology: nothing to repair automatically.
        return atoms

    central_sym = topo["central_symbol"]
    ligand_sym = topo["ligand_symbol"]
    coordination = topo["coordination"]
    bond_length = topo["bond_length"]
    bond_angle = topo["bond_angle"]

    symbols = np.array(atoms.get_chemical_symbols())
    if not (symbols == central_sym).any():
        return atoms

    cutoff = bond_length * bond_tol
    i_idx, j_idx = neighbor_list("ij", atoms, cutoff)

    central_neighbors: dict[int, list[int]] = {}
    ligand_has_center: set[int] = set()
    for i, j in zip(i_idx, j_idx):
        if symbols[i] == central_sym and symbols[j] == ligand_sym:
            central_neighbors.setdefault(int(i), []).append(int(j))
            ligand_has_center.add(int(j))

    to_delete: set[int] = set()
    to_add: list[tuple[str, np.ndarray]] = []

    central_indices = np.where(symbols == central_sym)[0]
    for c in central_indices:
        c = int(c)
        if protected[c]:
            continue
        ligs = central_neighbors.get(c, [])
        missing = coordination - len(ligs)
        if missing <= 0:
            continue
        if missing != 1 or coordination != 2 or len(ligs) != 1:
            # Ambiguous placement geometry (0 or >=2 missing ligands, or a
            # coordination other than 2) -- drop the whole fragment.
            to_delete.add(c)
            to_delete.update(ligs)
            continue

        delta_vec = atoms.positions[c] - atoms.positions[ligs[0]]
        delta_norm = delta_vec / np.linalg.norm(delta_vec)
        d1 = np.cos(np.deg2rad(180 - bond_angle)) * delta_vec
        perp = np.array([0.0, delta_norm[2], -delta_norm[1]])
        if np.linalg.norm(perp) < 1e-8:
            perp = np.array([delta_norm[2], 0.0, -delta_norm[0]])
        perp = perp / np.linalg.norm(perp)

        placed = False
        for _ in range(max_placement_tries):
            alpha = rng.random() * 360
            d2 = (
                np.sin(np.deg2rad(180 - bond_angle))
                * np.linalg.norm(delta_vec)
                * (
                    delta_norm * np.dot(delta_norm, perp)
                    + np.cos(np.deg2rad(alpha))
                    * np.cross(np.cross(delta_norm, perp), delta_norm)
                    + np.sin(np.deg2rad(alpha)) * np.cross(delta_norm, perp)
                )
            )
            candidate = atoms.positions[c] + d1 + d2
            all_positions = np.vstack(
                [atoms.get_positions()] + ([p for _, p in to_add] or [])
            )
            clashes = np.any(
                np.linalg.norm(all_positions - candidate, axis=1)
                < overlap_tol * bond_length
            )
            if not clashes:
                to_add.append((ligand_sym, candidate))
                placed = True
                break
        if not placed:
            to_delete.add(c)
            to_delete.update(ligs)

    ligand_indices = np.where(symbols == ligand_sym)[0]
    for l in ligand_indices:
        l = int(l)
        if protected[l]:
            continue
        if l not in ligand_has_center:
            to_delete.add(l)

    # Belt-and-braces: never delete a protected atom even if flagged above.
    to_delete = {i for i in to_delete if not protected[i]}

    if to_delete:
        del atoms[sorted(to_delete)]
    for sym, pos in to_add:
        atoms.append(Atom(sym, pos))

    return atoms


def cleanup_stoichiometry(
    atoms: Atoms,
    roi_center: np.ndarray,
    protect_radius: float,
    stoichiometry: dict[str, int],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Atoms:
    """Trim the matrix (atoms outside the ROI) to a target stoichiometry.

    No molecule template is used: atoms outside ``protect_radius`` are
    randomly deleted per-species until the remaining outside-ROI composition
    matches ``stoichiometry`` (e.g. ``{"H": 2, "O": 1}``), limited by the
    scarcest species. Species outside the ROI that are not present in
    ``stoichiometry`` are removed entirely. Atoms within ``protect_radius``
    of ``roi_center`` are always kept and excluded from the accounting.
    """
    if rng is None:
        rng = np.random.default_rng()
    atoms = atoms.copy()
    protected = _protected(atoms, roi_center, protect_radius)
    symbols = np.array(atoms.get_chemical_symbols())

    outside = np.where(~protected)[0]
    ratios = {sym: ratio for sym, ratio in stoichiometry.items() if ratio > 0}
    counts = {sym: int(np.sum(symbols[outside] == sym)) for sym in ratios}
    units = min(counts[sym] // ratio for sym, ratio in ratios.items())

    to_delete: list[int] = []
    for sym, ratio in ratios.items():
        target = units * ratio
        sym_outside = [int(i) for i in outside if symbols[i] == sym]
        excess = len(sym_outside) - target
        if excess > 0:
            chosen = rng.choice(sym_outside, size=excess, replace=False)
            to_delete.extend(int(i) for i in chosen)

    other_species = set(symbols[outside]) - set(ratios)
    for sym in other_species:
        to_delete.extend(int(i) for i in outside if symbols[i] == sym)

    if to_delete:
        del atoms[sorted(set(to_delete))]
    return atoms


def cleanup(
    atoms: Atoms,
    mode: str,
    roi_center: np.ndarray,
    protect_radius: float,
    *,
    solvent: Optional[Atoms] = None,
    stoichiometry: Optional[dict[str, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Atoms:
    """Dispatch to one of the three ``cleanup_mode`` strategies.

    Args:
        atoms: The carved box to clean up.
        mode: One of ``"none"``, ``"molecule"``, ``"stoichiometry"``.
        roi_center: Position of the region of interest.
        protect_radius: Radius around ``roi_center`` that is never touched.
        solvent: Required for ``mode="molecule"``.
        stoichiometry: Required for ``mode="stoichiometry"``.
        rng: Optional random generator.
    """
    if mode == "none":
        return cleanup_none(atoms)
    if mode == "molecule":
        if solvent is None:
            raise ValueError(
                "cleanup_mode='molecule' requires a solvent template"
            )
        return cleanup_molecule(
            atoms, roi_center, protect_radius, solvent, rng=rng
        )
    if mode == "stoichiometry":
        if not stoichiometry:
            raise ValueError(
                "cleanup_mode='stoichiometry' requires a stoichiometry dict"
            )
        return cleanup_stoichiometry(
            atoms, roi_center, protect_radius, stoichiometry, rng=rng
        )
    raise ValueError(f"Unknown cleanup_mode: {mode!r}")
