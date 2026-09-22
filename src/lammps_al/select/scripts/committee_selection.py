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

``backend="ace"`` is the one exception to "one ASE calculator spanning the
committee": it does NOT use ``pyace.PyACEEnsembleCalculator`` (confirmed,
by reading its source, to be a pure CPU/Cython evaluator with no GPU code
path at all -- unusably slow on a large system, e.g. ~40+ min for one real
206941-atom campaign trajectory and still not finished). Instead it uses
``tensorpotential``'s own re-implementation of ACE (the OLD, pre-GRACE-
rename API, only installed in the ``ace_gpu_train`` env -- see
``pyproject.toml``'s ``[ace_tensorpot_gpu]`` extra comment), which is
GPU-capable, plus ``matscipy`` for the neighbor list (confirmed 13.6x
faster than ASE's own on a real 206941-atom campaign frame, identical pair
count -- and also what the *current*, actively-maintained
``grace-tensorpotential`` package itself uses instead of ASE). Needs the
``[ace_tensorpot_gpu]`` extra (``matscipy`` -- a normal, portable
dependency, declared there directly -- plus the old ``tensorpotential``
API via ``remote_handler``'s own ``[ace_tensorpot_gpu]`` extra, which
genuinely isn't portable: it needs its own dedicated environment, pinned to
Python <=3.10, installed from a specific git snapshot -- see
``pyproject.toml``'s comment on that extra).

Shipped and executed on the cluster by
:class:`lammps_al.select.selector.CommitteeUncertaintySelector` -- it is not
imported as a module in the normal flow (``main`` runs it from ``sys.argv``),
but its functions are import-safe so they can be unit-tested. The heavy
``torch``/``mace``/``pyace``/``tensorpotential``/``matscipy`` imports are
done lazily inside the functions that need them, so the pure helpers
(``aggregate_uncertainties``, ``select_max_uncertainty_frames``) import with
only ``ase`` + ``numpy`` present -- no committee stack, no GPU.
"""

import glob
import logging
import os
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


def _fix_tensorflow_gpu_libs() -> None:
    """Re-exec this process once with ``LD_LIBRARY_PATH`` covering every
    ``site-packages/nvidia/*/lib`` directory, if not already done.

    ``ace_gpu_train``'s TensorFlow can't find its own pip-bundled CUDA
    libraries otherwise: the ``nvidia-cudnn-cu12``/``nvidia-cublas-cu12``
    packages (and their real ``.so`` files) are genuinely installed, but
    ``LD_LIBRARY_PATH`` doesn't include their directories and TF's RPATH-
    based auto-discovery of sibling ``nvidia-*-cu12`` packages isn't
    finding them either in this env -- confirmed by direct testing.
    Without this, TensorFlow silently falls back to CPU
    (``Cannot dlopen some GPU libraries... Skipping registering GPU
    devices``) even inside a real ``--gres=gpu`` SLURM allocation.

    Setting ``os.environ["LD_LIBRARY_PATH"]`` alone, mid-process, does NOT
    work -- confirmed empirically: glibc's dynamic linker reads
    ``LD_LIBRARY_PATH`` once at process exec and does not re-check it for
    a later ``import tensorflow`` further down the same process. Hence the
    re-exec: this re-runs the current script once, with the corrected
    ``LD_LIBRARY_PATH`` actually in place at the new process's exec time.

    No-op (returns immediately) on the re-exec'd process (guarded by a
    sentinel env var), and also if no ``nvidia/*/lib`` directories are
    found at all (e.g. a CPU-only env, or a ``mace``/``grace`` job whose
    env doesn't have this same gap) -- this is a targeted fix for one
    specific environment gap, not a general-purpose CUDA installer. Called
    unconditionally (not gated on ``backend``): harmless everywhere it
    isn't needed, and this same ``LD_LIBRARY_PATH`` gap plausibly affects
    any TensorFlow-based backend in an env that has it -- not just
    ``"ace"``'s ``tensorpotential`` usage.

    Only ever called from the ``if __name__ == "__main__":`` guard at the
    bottom of this file, before :func:`main` -- never from inside
    :func:`main`/:func:`annotate_uncertainties` themselves. ``os.execv``
    replaces the *current* process image outright (it doesn't fork); calling
    it after :func:`main` has already started would (a) re-run whatever
    file-creating side effects already happened (e.g. ``outpath.mkdir
    (exist_ok=False)`` would then raise ``FileExistsError`` on the restarted
    process) and (b) if ``main()`` were ever called in-process (e.g. from a
    test, with a custom ``argv`` that isn't ``sys.argv``), would hijack that
    *entire* process (a test runner included) into re-executing this script
    with the wrong arguments instead.
    """
    if os.environ.get("_TP_GPU_LIBS_FIXED") == "1":
        return

    site_packages = next(
        (p for p in sys.path if p.endswith("site-packages")), None
    )
    if site_packages is None:
        return
    libdirs = sorted(
        glob.glob(os.path.join(site_packages, "nvidia", "*", "lib"))
    )
    if not libdirs:
        return

    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(libdirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["_TP_GPU_LIBS_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


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

    Uses a cutoff-aware neighbor list, not an all-pairs distance matrix.
    """
    from ase import Atoms

    n_atoms = len(atom_pos)
    unc_mean = atomic_uncertainties.mean(axis=-1)

    dummy = Atoms(positions=atom_pos, cell=unit_cell, pbc=boundary_conditions)
    try:
        from matscipy.neighbours import neighbour_list

        ind_i, ind_j = neighbour_list("ij", dummy, r_cut)
    except ImportError:
        from ase.neighborlist import neighbor_list

        ind_i, ind_j = neighbor_list("ij", dummy, r_cut)

    # Neighbor lists exclude self-pairs; the original all-pairs mask always
    # included the atom itself (distance 0 < r_cut), so add it back here.
    counts = np.bincount(ind_i, minlength=n_atoms).astype(float) + 1.0
    sums = (
        np.bincount(ind_i, weights=unc_mean[ind_j], minlength=n_atoms)
        + unc_mean
    )
    return sums / counts


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


def _build_ace_tensorpotential_models(model_files: list, device: str) -> list:
    """One ``TensorPotential`` per ACE committee member -- the GPU-capable
    replacement for ``PyACEEnsembleCalculator`` (see the module docstring).

    Built once per :func:`annotate_uncertainties` call and reused across
    every trajectory frame. ``TensorPotential._evaluate`` is a
    ``tf.function`` with an explicit ``input_signature`` using dynamic
    (``None``) leading dimensions, so it traces its graph once regardless
    of how many atoms/pairs any given frame has -- confirmed: a real
    206941-atom frame's first ``evaluate()`` call took ~26s (incl. one-time
    tracing), a second call ~2.6s. Re-instantiating a ``TensorPotential``
    per frame would pay that tracing cost every frame instead of once.

    Args:
        model_files: Paths to the committee's pacemaker ``.yaml``/``.yace``
            potential files.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        One ``TensorPotential`` instance per committee member, in order.
    """
    from pyace import BBasisConfiguration
    from tensorpotential.potentials.ace import ACE
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.utils.utilities import init_gpu_config

    init_gpu_config({"gpu_index": 0 if device == "cuda" else -1})

    models = []
    for model_file in model_files:
        bbasisconfig = BBasisConfiguration(str(model_file))
        ace_potential = ACE(bbasisconfig)
        models.append(TensorPotential(ace_potential))
    return models


def _tp_atoms_from_matscipy(atoms, cutoff: float) -> dict:
    """Build one frame's neighbor-list data in the shape
    ``tensorpotential.utils.utilities.supply_batch``/``TensorPotential.
    evaluate`` need, using ``matscipy.neighbours.neighbour_list`` instead
    of ``tensorpotential.utils.utilities.generate_tp_atoms``'s ASE-based
    (``ase.neighborlist.NewPrimitiveNeighborList``) neighbor search.

    Confirmed (see RESULTS.md) 13.6x faster on a real 206941-atom frame
    (4.2s vs 57s) with an identical pair count, and is what the current,
    actively-maintained ``grace-tensorpotential`` package itself already
    uses instead of ASE's own neighbor list.

    Unlike ``generate_tp_atoms`` (fitting-data-prep code -- it reads
    ``atoms.get_potential_energy()``/``.get_forces()``), this never reads
    real energy/forces: it fills ``_energy``/``_forces``/``_eweights``/
    ``_fweights`` with placeholder zeros, since ``TensorPotential.
    evaluate()`` never reads them (only ``positions``/``offsets``/
    ``cell``/``ind_i``/``ind_j``/``mu_i``/``mu_j``/``pair_map``/
    ``cell_map``/``e_map`` -- see ``tensorpotential.tensorpot.
    TensorPotential._evaluate``'s ``input_signature``); pure inference, not
    fitting.
    """
    from matscipy.neighbours import neighbour_list

    at_nums = atoms.get_atomic_numbers()
    ind_i, ind_j, offsets = neighbour_list("ijS", atoms, cutoff)
    return {
        "_ind_i": ind_i,
        "_ind_j": ind_j,
        "_mu_i": at_nums[ind_i],
        "_mu_j": at_nums[ind_j],
        "_offsets": offsets.astype(np.float64),
        "_positions": atoms.get_scaled_positions().astype(np.float64),
        "_cell": atoms.get_cell().reshape(1, 3, 3).astype(np.float64),
        "_eweights": np.ones((1, 1)),
        "_fweights": np.ones((len(atoms), 1)),
        "_energy": np.zeros((1, 1)),
        "_forces": np.zeros((len(atoms), 3)),
    }


def _ace_committee_force_stdev(
    models: list, atoms, cutoff_radius: float
) -> np.ndarray:
    """Per-atom, per-component force standard deviation across an ACE
    committee, via ``tensorpotential`` (GPU-capable) instead of ``pyace``
    (CPU-only) -- see the module docstring.

    Args:
        models: Committee members, from :func:`_build_ace_tensorpotential_models`.
        atoms: Structure to evaluate (not mutated -- unlike
            :func:`_committee_force_stdev`'s ASE-``Calculator`` path, this
            never attaches a calculator to ``atoms``).
        cutoff_radius: Neighbor-list cutoff (Angstrom).

    Returns:
        Force standard deviation, shape ``(n_atoms, 3)``.
    """
    from tensorpotential.utils.utilities import supply_batch

    tp_atoms = _tp_atoms_from_matscipy(atoms, cutoff_radius)
    batch = supply_batch([tp_atoms])
    forces = []
    for tp in models:
        _, f, _ = tp.evaluate(batch)
        forces.append(np.asarray(f))
    return np.std(np.stack(forces, axis=0), axis=0, ddof=1)


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
            backend's calculator takes one ("cuda" or "cpu"; MACE and ACE)
        backend: ``"mace"``, ``"ace"``, or ``"grace"`` -- see the module
            docstring and :func:`_build_committee_calculator`. ``"ace"``
            uses a different (GPU-capable) code path,
            :func:`_build_ace_tensorpotential_models` /
            :func:`_ace_committee_force_stdev`, not
            :func:`_build_committee_calculator`.
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

    if backend == "ace":
        models = _build_ace_tensorpotential_models(model_files, device)
    else:
        model_calc = _build_committee_calculator(backend, model_files, device)

    start = time.monotonic()
    for i, atoms in enumerate(structs):
        if backend == "ace":
            stdev = _ace_committee_force_stdev(models, atoms, cutoff_radius)
        else:
            a = atoms.copy()
            stdev = _committee_force_stdev(backend, a, model_calc)
        struct_uncertainties = aggregate_uncertainties(
            atom_pos=atoms.get_positions(),
            unit_cell=atoms.cell,
            boundary_conditions=atoms.pbc,
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
    # Only here, before main() does anything -- see _fix_tensorflow_gpu_libs's
    # own docstring for why this must never run from inside main() itself.
    _fix_tensorflow_gpu_libs()
    main()
