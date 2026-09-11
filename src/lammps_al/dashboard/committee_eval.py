"""Score a trained ACE committee on reference configurations.

pacemaker produces no portable prediction bundle (only the ``.yaml``
potential per committee member), so parity data must come from *re-scoring*
reference configs with the trained committee, not from reading a
pre-computed bundle. ``pyace.PyACEEnsembleCalculator`` already does exactly
this, and ``lammps_al``'s own sampler already builds one via
``_build_committee_calculator`` for uncertainty scoring -- this module
reuses that directly rather than inventing a second pyace-loading path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _default_committee_calculator(model_files, device: str = "cpu"):
    """Build the same committee calculator lammps_al's sampler uses.

    Lazy import: only pulled in when a caller actually needs pyace, so
    panels that don't touch committee predictions (uncertainty/spatial
    panels) never require it installed.
    """
    from lammps_al.run.scripts.lammps_sampling import (
        _build_committee_calculator,
    )

    return _build_committee_calculator(
        "ace", [str(p) for p in model_files], device
    )


def score_with_committee(
    model_files,
    atoms_list,
    *,
    device: str = "cpu",
    want_forces: bool = True,
    _build_calculator=_default_committee_calculator,
):
    """Committee-predicted energy/forces for each of ``atoms_list``.

    Args:
        model_files: Committee-member potential paths (``.yaml``).
        atoms_list: Configurations to score (unmodified; scored on copies).
        device: Passed through to the calculator builder (unused by ACE,
            kept for signature parity with the MACE/GRACE backends).
        want_forces: Whether to also compute/return forces.
        _build_calculator: Test seam -- patched in tests to avoid needing
            ``pyace`` installed. Defaults to the real committee calculator.

    Returns:
        ``(energies, forces)``: ``energies`` a list of per-config total
        energies (eV, not per-atom); ``forces`` a list of ``(n_atoms, 3)``
        arrays, or ``None`` when ``want_forces`` is False.
    """
    model_files = [Path(p) for p in model_files]
    calc = _build_calculator(model_files, device)

    energies = []
    forces = [] if want_forces else None
    for atoms in atoms_list:
        scored = atoms.copy()
        scored.calc = calc
        energies.append(float(scored.get_potential_energy()))
        if want_forces:
            forces.append(np.asarray(scored.get_forces(), dtype=float))
    return energies, forces
