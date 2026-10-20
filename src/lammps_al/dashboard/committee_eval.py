"""Score a trained ACE committee on reference configurations.

The trainer output the AL loop records is only the ``.yaml`` potential per
committee member, so parity of the *committee* on arbitrary configs must
come from *re-scoring* them, not from reading a pre-computed bundle.
(pacemaker does leave each member's own final-fit predictions on its own
train/test split in the member's calcdir -- see
:func:`read_pacemaker_predictions` -- but those are per member and fixed to
that member's split.) ``pyace.PyACEEnsembleCalculator`` already does exactly
this, and ``lammps_al``'s own committee-selection payload already builds one
via ``_build_committee_calculator`` for uncertainty scoring -- this module
reuses that directly rather than inventing a second pyace-loading path.

pacemaker fits ``energy_corrected`` -- the reference energy minus a
per-element ``reference_energy`` shift -- and the potential predicts on that
scale, so raw committee energies sit a composition-dependent constant away
from ``REF_energy``. :func:`score_with_committee` adds the shift back (read
from the potential's own metadata, see :func:`read_reference_energies`) so
its energies compare directly against ``REF_energy``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _default_committee_calculator(model_files, device: str = "cpu"):
    """Build the same committee calculator lammps_al's sampler uses.

    Lazy import: only pulled in when a caller actually needs pyace, so
    panels that don't touch committee predictions (uncertainty/spatial
    panels) never require it installed.
    """
    from lammps_al.select.scripts.committee_selection import (
        _build_committee_calculator,
    )

    logger.info(
        "Building committee calculator from %d model file(s) (device=%s): %s",
        len(model_files),
        device,
        [str(p) for p in model_files],
    )
    start = time.monotonic()
    calc = _build_committee_calculator(
        "ace", [str(p) for p in model_files], device
    )
    logger.debug(
        "Committee calculator built in %.1fs.", time.monotonic() - start
    )
    return calc


_REFERENCE_ENERGY_LINE = re.compile(r"^\s+reference_energy:\s*(.+?)\s*$")


def _read_reference_energy(path: Path) -> dict:
    """``{element: eV}`` from one potential's ``metadata`` block, or ``{}``
    if the file or the key is absent.

    Only the leading ``metadata:`` block is scanned (the rest of a potential
    can be large); pacemaker stores the value as a JSON string.
    """
    if not path.is_file():
        return {}
    with open(path) as f:
        if f.readline().strip() != "metadata:":
            return {}
        for line in f:
            if line.strip() and not line[0].isspace():
                break  # end of the metadata block
            match = _REFERENCE_ENERGY_LINE.match(line)
            if match:
                value = match.group(1)
                if value.startswith('"'):
                    value = json.loads(value)  # YAML double-quoted string
                elif value.startswith("'"):
                    value = value[1:-1].replace("''", "'")
                return {k: float(v) for k, v in json.loads(value).items()}
    return {}


def read_reference_energies(model_files) -> dict:
    """The per-element ``reference_energy`` shift the committee was fit
    with (``{}`` when the potentials record none).

    Raises:
        ValueError: If committee members record different shifts -- their
            energies would then not share one scale.
    """
    shifts = {Path(p): _read_reference_energy(Path(p)) for p in model_files}
    distinct = {tuple(sorted(s.items())) for s in shifts.values()}
    if len(distinct) > 1:
        raise ValueError(
            "Committee members record different reference energies: "
            + ", ".join(f"{p.name}={s}" for p, s in shifts.items())
        )
    return dict(distinct.pop()) if distinct else {}


def read_pacemaker_metrics(calcdir, *, split: str = "train"):
    """A single committee member's own fit-convergence curve, as pacemaker
    already computed it -- no re-scoring needed.

    Unlike an ensemble parity check (see this module's own ``score_with_
    committee``, and ``committee_parity.py``'s module docstring), this is
    strictly per-member: pacemaker's ``metrics.txt``/``test_metrics.txt``
    log one row per fitting iteration for *that* member's own fit against
    *its own* (possibly bagged/resampled -- confirmed empirically to differ
    member-to-member) training subset, so there is no cross-member
    alignment question here at all, and reading it is exact, not an
    approximation of the re-scored result.

    Args:
        calcdir: One committee member's trainer calcdir (e.g. from
            ``run_layout.trainer_calcdirs_for_iteration``), containing
            pacemaker's raw ``metrics.txt``/``test_metrics.txt``.
        split: ``"train"`` (``metrics.txt``, every fitting iteration) or
            ``"test"`` (``test_metrics.txt``, sparser checkpoints).

    Returns:
        A ``pandas.DataFrame`` with pacemaker's own columns (at least
        ``iter_num``, ``loss``, ``rmse_epa``, ``rmse_f_comp``).

    Raises:
        FileNotFoundError: If this calcdir has no such metrics file (e.g.
            still running, or a fit that failed before pacemaker wrote one).
    """
    import pandas as pd

    calcdir = Path(calcdir)
    filename = "metrics.txt" if split == "train" else f"{split}_metrics.txt"
    path = calcdir / filename
    if not path.exists():
        raise FileNotFoundError(f"No {filename} in {calcdir}.")
    df = pd.read_csv(path, sep=r"\s+")
    logger.debug("Read %d fitting-iteration row(s) from %s.", len(df), path)
    return df


def read_pacemaker_predictions(calcdir, *, split: str = "train"):
    """A single committee member's final-fit predictions on its own
    train/test split, as pacemaker wrote them at the end of the fit -- no
    re-scoring needed.

    Energies are on pacemaker's ``energy_corrected`` scale (reference
    energy minus the per-element reference-energy shift pacemaker fits
    against), which is what ``energy_pred`` predicts; it differs from
    ``REF_energy`` by a per-composition constant.

    Args:
        calcdir: One committee member's trainer calcdir (see
            :func:`read_pacemaker_metrics`).
        split: ``"train"`` or ``"test"`` (``{split}_pred.pckl.gzip``).

    Returns:
        A ``pandas.DataFrame``, one row per structure, with at least
        ``NUMBER_OF_ATOMS``, ``energy_corrected``, ``energy_pred``,
        ``forces`` and ``forces_pred`` (per-structure ``(n, 3)`` arrays).

    Raises:
        FileNotFoundError: If this calcdir has no such file (still running,
            a failed fit, or no held-out split configured for ``"test"``).
    """
    import pandas as pd

    path = Path(calcdir) / f"{split}_pred.pckl.gzip"
    if not path.exists():
        raise FileNotFoundError(f"No {path.name} in {calcdir}.")
    df = pd.read_pickle(path, compression="gzip")
    logger.debug("Read %d %s prediction(s) from %s.", len(df), split, path)
    return df


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
        energies (eV, not per-atom) on the ``REF_energy`` scale, i.e. with
        the potentials' ``reference_energy`` shift added back; ``forces`` a
        list of ``(n_atoms, 3)`` arrays, or ``None`` when ``want_forces``
        is False.
    """
    model_files = [Path(p) for p in model_files]
    reference_energy = read_reference_energies(model_files)
    if reference_energy:
        logger.info(
            "Adding the potentials' per-element reference energies back to "
            "predicted energies: %s",
            reference_energy,
        )
    calc = _build_calculator(model_files, device)

    logger.info(
        "Scoring %d configuration(s) with the committee (forces=%s) -- this "
        "re-runs full committee inference, not a cached read.",
        len(atoms_list),
        want_forces,
    )
    start = time.monotonic()
    energies = []
    forces = [] if want_forces else None
    for atoms in atoms_list:
        scored = atoms.copy()
        scored.calc = calc
        shift = sum(
            reference_energy.get(s, 0.0) for s in scored.get_chemical_symbols()
        )
        energies.append(float(scored.get_potential_energy()) + shift)
        if want_forces:
            forces.append(np.asarray(scored.get_forces(), dtype=float))
    elapsed = time.monotonic() - start
    logger.info(
        "Scored %d configuration(s) in %.1fs (%.2fs/config).",
        len(atoms_list),
        elapsed,
        elapsed / len(atoms_list) if atoms_list else 0.0,
    )
    return energies, forces


def committee_force_spread(
    model_files,
    atoms_list,
    *,
    device: str = "cpu",
    _build_calculator=_default_committee_calculator,
):
    """Committee mean forces *and* their per-atom spread for each config.

    The spread is the ensemble calculator's own per-component force
    standard deviation (``results["forces_std"]``) -- the same quantity
    ``committee_selection`` aggregates into the sampler's ``uncertainty``
    array, so it can be compared against actual force errors directly.

    Args:
        model_files / atoms_list / device / _build_calculator: As
            :func:`score_with_committee`.

    Returns:
        ``(forces, forces_std)``: two lists of ``(n_atoms, 3)`` arrays.
    """
    model_files = [Path(p) for p in model_files]
    calc = _build_calculator(model_files, device)
    logger.info(
        "Scoring %d configuration(s) with the committee (forces + spread).",
        len(atoms_list),
    )
    start = time.monotonic()
    forces, forces_std = [], []
    for atoms in atoms_list:
        scored = atoms.copy()
        scored.calc = calc
        forces.append(np.asarray(scored.get_forces(), dtype=float))
        forces_std.append(
            np.asarray(scored.calc.results["forces_std"], dtype=float)
        )
    logger.info(
        "Scored %d configuration(s) in %.1fs.",
        len(atoms_list),
        time.monotonic() - start,
    )
    return forces, forces_std
