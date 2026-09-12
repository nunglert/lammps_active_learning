"""Cluster-side committee-uncertainty annotation payload.

Scores every frame of an already-MD'd trajectory (produced by
``lammps_al.run.scripts.lammps_sampling``, one LAMMPS run per AL iteration)
against the training committee, and writes the *whole* trajectory back out
with a per-atom ``uncertainty`` array attached to every frame (the input
:class:`lammps_al.embedding.extractor.ROIExtractor` expects). It does not
reduce/select anything -- every frame that went in comes back out, just
annotated.

This is the "annotate" half of what used to be one combined sampler payload
(``lammps_sampling.py``); splitting it into its own remote job (and thus its
own actor, ``CommitteeUncertaintySelector``) mirrors this repo's existing
``ROIExtractor``/``EmbeddingAnnealer`` split. Peak-uncertainty-frame
reduction (:func:`select_max_uncertainty_frames`, still defined here since
it needs no committee/cluster) is a *separate*, optional, host-side step for
campaigns that need it (no remote job -- it only needs the ``uncertainty``
arrays this payload already wrote) -- see
:class:`lammps_al.select.selector.PeakUncertaintySelector`. Campaigns that
do their own downstream spatial selection (``ROIExtractor``) skip that step
entirely and consume this payload's full annotated output directly.

Backend-generic over the committee's potential family
(``CommitteeSelectionInput.backend``: ``"mace"``, ``"ace"``, or ``"grace"``)
-- only the committee-scoring calculator
(:func:`_build_committee_calculator`) is backend-specific. ACE's/GRACE's
ensemble calculators already aggregate a ``forces_std`` across the
committee, unlike MACE's ``MACECalculator`` (``forces_comm``, per-model
forces -- the stdev is computed here).

Shipped and executed on the cluster by
:class:`lammps_al.select.selector.CommitteeUncertaintySelector` -- it is not
imported as a module in the normal flow (``main`` runs it from ``sys.argv``),
but its functions are import-safe so they can be unit-tested. The heavy
``torch``/``mace``/``pyace``/``tensorpotential`` imports are done lazily
inside the functions that need them, so the pure helpers
(``aggregate_uncertainties``, ``select_max_uncertainty_frames``) import with
only ``ase`` + ``numpy`` present -- no committee stack, no GPU.
"""

import logging
import pathlib
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np

# ``CommitteeSelectionInput`` (and thus ``remote_handler``) is imported
# lazily in ``main`` so the pure helpers below import with only ase + numpy.


def aggregate_uncertainties(
    atom_pos: np.ndarray,
    unit_cell: np.ndarray,
    boundary_conditions: np.ndarray,
    atomic_uncertainties: np.ndarray,
    r_cut: float,
) -> np.ndarray:
    """Calculate the locally aggregated uncertainties for a single structure

    Args:
        atom_pos: Atomic positions, shape=(n_atoms, 3)
        unit_cell: Unit cell, shape=(3, 3)
        boundary_conditions: Array of booleans indicating boundary conditions,
            shape=(3)
        atomic_uncertainties: Atomic force uncertainties, shape=(n_atoms, 3)
        r_cut: Cutoff radius defining the aggregation region

    Returns:
        Aggregated uncertainties for each atom, shape=(n_atoms)
    """
    import ase.geometry

    radii = ase.geometry.get_distances(
        atom_pos, cell=unit_cell, pbc=boundary_conditions
    )[1]
    mask = radii < r_cut
    unc_mean = atomic_uncertainties.mean(axis=-1)
    return (mask * unc_mean[np.newaxis, :]).sum(axis=1) / mask.sum(axis=1)


def select_max_uncertainty_frames(structs: list, bin_count: int) -> list:
    """Pick the peak-uncertainty frame from each of ``bin_count`` segments.

    Splits ``structs`` (which must each carry a per-atom ``uncertainty``
    array, written by :func:`annotate_uncertainties`, and share a constant
    atom count) into ``bin_count`` equal contiguous segments and returns,
    for each segment, the frame whose single most uncertain atom is
    largest. ``len(structs)`` must be divisible by ``bin_count`` (the
    caller trims the tail first, e.g.
    :class:`lammps_al.select.selector.PeakUncertaintySelector`).

    Needs no committee backend or cluster job -- pure numpy over
    already-annotated frames -- so it's meant to run host-side as an
    optional, separate reduction step, not inside this remote payload.
    """
    num_atoms = np.array(structs).shape[1]
    uncertainties = np.array([s.arrays["uncertainty"] for s in structs])
    uncertainties = uncertainties.reshape((bin_count, -1, num_atoms))
    segment_size = uncertainties.shape[1]
    offsets = np.arange(0, bin_count * segment_size, segment_size)
    indices = np.argmax(uncertainties.max(axis=2), axis=1) + offsets
    return [structs[i] for i in indices]


def _cuet_available() -> bool:
    """Whether cuequivariance(_torch) is importable (enables cueq in MACE)."""
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401

        return True
    except ImportError:
        return False


def _build_committee_calculator(backend: str, model_files: list, device: str):
    """Build one ASE calculator spanning the whole committee.

    Each backend's ensemble calculator already reports (or, for MACE, makes
    directly computable) a per-atom, per-component force standard deviation
    across its members -- read back by :func:`_committee_force_stdev`. Built
    once per :func:`annotate_uncertainties` call and reused across every
    trajectory frame (re-instantiating a calculator per frame would reload
    every committee member's weights each time).

    Args:
        backend: ``"mace"``, ``"ace"``, or ``"grace"``.
        model_files: Committee member paths (format depends on ``backend``,
            see ``CommitteeSelectionInput.model_files``).
        device: Compute device, where the backend's calculator takes one
            (MACE's torch device; unused by ACE/GRACE).

    Returns:
        The backend's ASE ``Calculator`` instance.
    """
    if backend == "mace":
        from mace.calculators.mace import MACECalculator

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            return MACECalculator(
                model_paths=model_files,
                device=device,
                default_dtype="float32",
                enable_cueq=_cuet_available(),
            )
    elif backend == "ace":
        # python-ace/pacemaker (conda-forge; PyPI's "pyace" is an unrelated
        # package -- do not `pip install pyace`).
        from pyace import PyACEEnsembleCalculator

        return PyACEEnsembleCalculator([str(p) for p in model_files])
    elif backend == "grace":
        # gracemaker / tensorpotential.
        from tensorpotential.calculator.asecalculator import TPCalculator

        return TPCalculator(model=[str(p) for p in model_files])
    else:
        raise ValueError(
            f"Unknown committee backend {backend!r}; expected 'mace', "
            "'ace', or 'grace'."
        )


def _committee_force_stdev(backend: str, atoms, calculator) -> np.ndarray:
    """Per-atom, per-component force standard deviation across the committee.

    Runs ``calculator`` (built once by :func:`_build_committee_calculator`)
    on ``atoms`` and reads back the ensemble spread: MACE's calculator
    reports every member's forces (``forces_comm``), so the stdev is
    computed here; ACE's and GRACE's ensemble calculators already aggregate
    one (``forces_std``).

    Args:
        backend: ``"mace"``, ``"ace"``, or ``"grace"``.
        atoms: Structure to evaluate (mutated: ``atoms.calc`` is set).
        calculator: The committee calculator from
            :func:`_build_committee_calculator`.

    Returns:
        Force standard deviation, shape ``(n_atoms, 3)``.
    """
    atoms.calc = calculator
    atoms.get_potential_energy()
    if backend == "mace":
        forces = atoms.calc.results["forces_comm"]
        return np.std(forces, axis=0, ddof=1)
    return atoms.calc.results["forces_std"]


def annotate_uncertainties(
    trajectory_file: pathlib.Path,
    model_files: list,
    cutoff_radius: int,
    output_file: pathlib.Path,
    logger: logging.Logger,
    device: str = "cuda",
    backend: str = "mace",
) -> None:
    """Score every frame of ``trajectory_file`` against the committee and
    write the whole trajectory back out with a per-atom ``uncertainty``
    array attached to every frame. No reduction/selection -- every frame
    that went in comes back out.

    Args:
        trajectory_file: Path to an extxyz trajectory to score (already
            carrying correct chemical symbols -- see
            ``lammps_al.run.scripts.lammps_sampling``)
        model_files: List of Paths to the committee's model files (format
            depends on ``backend``)
        cutoff_radius: cutoff radius (in Angstrom) for calculating the local
            uncertainties
        output_file: path to write the annotated trajectory to
        logger: logger
        device: compute device for the committee calculator, where the
            backend's calculator takes one ("cuda" or "cpu"; MACE only)
        backend: ``"mace"``, ``"ace"``, or ``"grace"`` -- see the module
            docstring and :func:`_build_committee_calculator`.
    """
    structs = ase.io.read(trajectory_file, index=":")
    n_frames = len(structs)
    logger.info(
        "Annotating %d frames with committee uncertainty (backend=%r, "
        "%d atoms/frame)",
        n_frames,
        backend,
        len(structs[0]) if structs else 0,
    )

    model_calc = _build_committee_calculator(backend, model_files, device)

    start = time.monotonic()
    for i, atoms in enumerate(structs):
        a = atoms.copy()
        stdev = _committee_force_stdev(backend, a, model_calc)
        struct_uncertainties = aggregate_uncertainties(
            atom_pos=a.get_positions(),
            unit_cell=a.cell,
            boundary_conditions=a.pbc,
            atomic_uncertainties=stdev,
            r_cut=cutoff_radius,
        )
        atoms.arrays["uncertainty"] = struct_uncertainties

        elapsed = time.monotonic() - start
        per_frame = elapsed / (i + 1)
        remaining = per_frame * (n_frames - i - 1)
        logger.info(
            "Scored frame %d/%d (%.1fs elapsed, %.1fs/frame, ~%.1fs "
            "remaining)",
            i + 1,
            n_frames,
            elapsed,
            per_frame,
            remaining,
        )

    logger.info(f"Annotated {n_frames} frames with uncertainty")
    ase.io.write(output_file, structs, format="extxyz")


def main(argv: Optional[list] = None) -> None:
    """Cluster-side entry point: run one committee-uncertainty annotation
    job.

    Reads the ``CommitteeSelectionInput`` JSON named by ``argv[1]``, scores
    every frame of the trajectory, and writes the ``is_finished.info``
    success marker that ``SingleScriptTask.finish_check`` polls on the host.
    """
    from remote_handler import configure_logging

    from lammps_al.select.inputs import CommitteeSelectionInput

    configure_logging()

    argv = sys.argv if argv is None else argv

    inputfile = Path(argv[1])
    calcpath = inputfile.parent
    input_config = CommitteeSelectionInput.from_file(inputfile)
    input_config.validate(calcpath)

    # MACE's LAMMPS/ASE calculators default to GPU and raise on CPU tensors
    # ("Set MACE_ALLOW_CPU=true ..."). When the run is CPU-bound, opt in.
    if input_config.device == "cpu":
        import os

        os.environ["MACE_ALLOW_CPU"] = "true"

    outpath = calcpath / "out"
    outpath.mkdir(exist_ok=False)

    trajectory_path = calcpath / input_config.trajectory
    model_paths = [calcpath / p for p in input_config.model_files]

    output_file = outpath / "trajectory_uncertainties.extxyz"

    annotate_uncertainties(
        trajectory_file=trajectory_path,
        model_files=model_paths,
        cutoff_radius=input_config.cutoff_radius,
        output_file=output_file,
        logger=logging.getLogger(),
        device=input_config.device,
        backend=input_config.backend,
    )

    # Success marker checked by SingleScriptTask.finish_check on the host.
    (calcpath / "is_finished.info").write_text(
        "Committee uncertainty annotation finished"
    )


if __name__ == "__main__":
    main()
