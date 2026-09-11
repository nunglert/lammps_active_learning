"""Cluster-side LAMMPS committee-uncertainty sampling payload.

Runs one LAMMPS MD trajectory driven by the committee's first model, scores
every frame's local force-uncertainty across the committee, and writes the
most-uncertain frame of each trajectory segment to ``out/samples.xyz``.
Every frame also carries a per-atom ``uncertainty`` array (the input
:class:`lammps_al.embedding.extractor.ROIExtractor` expects).

Backend-generic over the committee's potential family
(``LammpsSamplingInput.backend``: ``"mace"``, ``"ace"``, or ``"grace"``) --
only the compile step and the committee-scoring calculator
(:func:`_build_committee_calculator`) are backend-specific; template
rendering, the LAMMPS run, and the uncertainty aggregation/selection logic
are shared. ACE's native ``pair_style pace`` and GRACE's native
``pair_style grace`` read their potential file/directory directly (no
compile step, unlike MACE); ACE/GRACE committee scoring uses their own
ensemble ASE calculators (``pyace``'s ``PyACEEnsembleCalculator``,
``tensorpotential``'s ``TPCalculator``), which already aggregate a
``forces_std`` across the committee, unlike MACE's ``MACECalculator``
(``forces_comm``, per-model forces -- the stdev is computed here).

For ``backend="mace"``, two LAMMPS/MACE coupling routes are chosen per run
via ``LammpsSamplingInput.lammps_pair_style``:

* ``"mliap"`` (**default, production**) -- the ML-IAP Python bridge
  (``pair_style mliap unified``, ``LAMMPS_MLIAP_MACE``, pickled with
  ``torch.save``). This is what the KOKKOS-on-GPU production setup targets.
* ``"mace"`` (**CPU-testable, opt-in**) -- LAMMPS's native ``pair_style mace``
  (the ``ML-MACE`` package, built against LibTorch directly,
  ``LAMMPS_MACE``/TorchScript-compiled). For any MACE model with more than
  one interaction layer (i.e. essentially all real models), the ML-IAP route
  needs LAMMPS's ghost-atom message-passing hooks
  (``forward_exchange``/``reverse_exchange``), which are Kokkos-CUDA-only --
  confirmed by building both the ACEsuit LAMMPS fork and vanilla upstream
  ``develop`` and hitting ``AttributeError: ... no attribute
  'forward_exchange'`` on CPU (Kokkos-OpenMP or no-Kokkos) either way. The
  ``"mace"`` route has no such limitation and is what the CPU-only
  ``lammps-smoke`` CI job (and local `docker run` parity, see
  ``docker/Dockerfile.lammps-mace``) uses -- it is **not** meant to replace
  the production route, only to make the payload testable without a GPU.

Shipped and executed on the cluster by
:class:`lammps_al.run.sampler.LammpsSampler` -- it is not imported as a module
in the normal flow (``main`` runs it from ``sys.argv``), but its functions are
import-safe so they can be unit-tested. The heavy ``torch`` / ``mace`` imports
are done lazily inside the functions that need them, so the pure helpers
(``parse_jinja_template``, ``aggregate_uncertainties``,
``select_max_uncertainty_frames``) import with only ``ase`` + ``numpy`` +
``jinja2`` present -- no MACE stack, no LAMMPS, no GPU.
"""

import logging
import os
import pathlib
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import ase.geometry
import ase.io
import jinja2
import numpy as np

# ``LammpsSamplingInput`` (and thus ``remote_handler``) is imported lazily in
# ``main`` so the pure helpers below import with only ase + numpy + jinja2.

# Default LAMMPS accelerator flags: KOKKOS on one GPU (production default,
# paired with lammps_pair_style="mliap"). CPU runs (e.g. CI, via
# lammps_pair_style="mace") override this via LammpsSamplingInput.lammps_args
# -- "" is the right override there, since `pair_style mace` needs no
# accelerator flags at all on a CPU-only build.
DEFAULT_LAMMPS_ARGS = "-k on g 1 -sf kk -pk kokkos newton on neigh half"


def _cuet_available() -> bool:
    """Whether cuequivariance(_torch) is importable (enables cueq in MACE)."""
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401

        return True
    except ImportError:
        return False


def parse_jinja_template(
    output_file: pathlib.Path,
    template_dir: pathlib.Path,
    template_file_name: str,
    **kwargs,
) -> None:
    """Parse a Jinja2 template and write to file

    Args:
        output_file: Path to file to store the parsed template to
        template_dir: Path to directory containing Jinja2 templates
        template_file_name: File name of the template to use
        **kwargs: Keyword arguments setting template parameters to fill
    """
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
    )
    template = jinja_env.get_template(template_file_name)
    with open(output_file.resolve(), "w") as file:
        file.write(template.render(**kwargs))


def create_lammps_mliap_model(
    model_path: pathlib.Path,
    head: Optional[str] = None,
    float32: bool = False,
    do_return_path: bool = False,
    out_path: Optional[pathlib.Path] = None,
    enable_cueq: bool = True,
) -> Optional[pathlib.Path]:
    """Compile a MACE model for LAMMPS's ML-IAP Python bridge
    (``pair_style mliap unified``) -- the **production** (KOKKOS-on-GPU)
    route. See the module docstring for why the CPU-testable alternative is
    :func:`create_lammps_mace_model`, not this function.

    Args:
        model_path: Path to the trained MACE model
        head: Which model head to select for compilation
        float32: Compile a 32-bit model
        do_return_path: Return a pathlib.Path for the compiled model
        out_path: Path to file to save the compiled model to
        enable_cueq: Convert the model to the cuequivariance (cueq)
            acceleration before compiling (the production default). Set False
            to compile the plain e3nn model -- this drops the ``cuequivariance``
            requirement, so a CPU-only build (e.g. CI) can compile it.

    Returns:
        If requested, path to the compiled model
    """
    from copy import deepcopy

    import torch
    from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE

    if out_path is not None:
        compiled_path = out_path
    else:
        compiled_path = model_path.parent / (
            model_path.name + "-mliap-lammps.pt"
        )
    model = torch.load(
        f=model_path.resolve(),
        map_location=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        weights_only=False,
    )

    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(
            f"Only one head found in the model: {heads[0]}. Skipping selection."
        )
        head = heads[0]
    else:
        head = "Default" if head is None else head

    if float32:
        model = model.float().to("cpu")
    else:
        model = model.double().to("cpu")
    if enable_cueq:
        # Lazy import: cuequivariance is only needed for the cueq conversion.
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

        model = run_e3nn_to_cueq(deepcopy(model))
    model.lammps_mliap = True
    lammps_model = LAMMPS_MLIAP_MACE(model, head=head)
    torch.save(lammps_model, compiled_path.resolve())
    return compiled_path if do_return_path else None


def create_lammps_mace_model(
    model_path: pathlib.Path,
    head: Optional[str] = None,
    float32: bool = False,
    do_return_path: bool = False,
    out_path: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """Compile a MACE model into a TorchScript module for LAMMPS's native
    ``pair_style mace`` (the ``ML-MACE`` package) -- the CPU-testable, opt-in
    ``lammps_pair_style="mace"`` route; see the module docstring.

    Mirrors the ``--format libtorch`` branch of MACE's own
    ``mace.cli.create_lammps_model``: wrap the model in ``LAMMPS_MACE`` and
    compile it with e3nn's ``jit.compile``. Unlike the production
    ``lammps_pair_style="mliap"`` route (:func:`create_lammps_mliap_model`,
    ``LAMMPS_MLIAP_MACE`` pickled with ``torch.save``), this needs no
    ``cuequivariance``/GPU-Kokkos requirement to *run* -- ``pair_style mace``
    works on a plain CPU build.

    Args:
        model_path: Path to the trained MACE model
        head: Which model head to select for compilation
        float32: Compile a 32-bit model
        do_return_path: Return a pathlib.Path for the compiled model
        out_path: Path to file to save the compiled model to

    Returns:
        If requested, path to the compiled model
    """
    import torch
    from e3nn.util import jit
    from mace.calculators import LAMMPS_MACE

    if out_path is not None:
        compiled_path = out_path
    else:
        compiled_path = model_path.parent / (model_path.name + "-lammps.pt")
    model = torch.load(
        f=model_path.resolve(),
        map_location=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        weights_only=False,
    )

    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(
            f"Only one head found in the model: {heads[0]}. Skipping selection."
        )
        head = heads[0]
    else:
        head = "Default" if head is None else head

    if float32:
        model = model.float().to("cpu")
    else:
        model = model.double().to("cpu")
    lammps_model = (
        LAMMPS_MACE(model, head=head)
        if head is not None
        else LAMMPS_MACE(model)
    )
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(compiled_path.resolve())
    return compiled_path if do_return_path else None


def run_lammps_from_template(
    lmp_cmd: str,
    log_file: pathlib.Path,
    input_file: pathlib.Path,
    template_dir: pathlib.Path,
    template_file_name: str,
    do_return_path: bool = False,
    trj_file_key: Any = "trj_file",
    logger: Optional[logging.Logger] = None,
    lammps_args: str = DEFAULT_LAMMPS_ARGS,
    **kwargs,
) -> Optional[pathlib.Path]:
    """Run a LAMMPS MD simulation using an input file template

    Args:
        lmp_cmd: LAMMPS executable command
        log_file: Path to write the LAMMPS log output to
        input_file: Path to write the parsed LAMMPS input file template to
        template_dir: Directory containing template files
        template_file_name: Name of template file to use
        do_return_path: Return the path to the MD run trajectory
        trj_file_key: Name of the kwarg specifying the trajectory
            file to be filled in the LAMMPS template. Needed to return the path
            to the trajectory file. Returns None if 'do_return_path' is True
            but the key is not found in the kwargs dictionary
        logger: Logger to write non-LAMMPS log output to
        lammps_args: Command-line flags inserted between ``-in`` and ``-log``
            (accelerator/package selection). Defaults to KOKKOS-on-GPU; pass a
            CPU variant (or "") for a non-GPU build.
        **kwargs: Keyword arguments passed to the template parser

    Returns:
        Path to the MD trajectory (if requested)
    """
    parse_jinja_template(
        input_file,
        template_dir,
        template_file_name,
        **kwargs,
    )
    command = (
        f"{lmp_cmd} -in {input_file.resolve()} {lammps_args} "
        f"-log {log_file.resolve()}"
    )
    try:
        subprocess.run(
            command,
            check=True,
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as run_error:
        if logger is not None:
            logger.error(
                f"LAMMPS run returned with non-zero exit code "
                f"{run_error.returncode}"
            )
            logger.error(run_error.stderr.decode("utf-8"))
        raise RuntimeError(run_error)
    if do_return_path:
        if trj_file_key in kwargs:
            return pathlib.Path(kwargs[trj_file_key])
        else:
            logger.warning(
                f"'{trj_file_key}' not found in kwargs, returning None"
            )
    return None


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
    radii = ase.geometry.get_distances(
        atom_pos, cell=unit_cell, pbc=boundary_conditions
    )[1]
    mask = radii < r_cut
    unc_mean = atomic_uncertainties.mean(axis=-1)
    return (mask * unc_mean[np.newaxis, :]).sum(axis=1) / mask.sum(axis=1)


def select_max_uncertainty_frames(structs: list, bin_count: int) -> list:
    """Pick the peak-uncertainty frame from each of ``bin_count`` segments.

    Splits ``structs`` (which must each carry a per-atom ``uncertainty``
    array and share a constant atom count) into ``bin_count`` equal contiguous
    segments and returns, for each segment, the frame whose single most
    uncertain atom is largest. ``len(structs)`` must be divisible by
    ``bin_count`` (the caller trims the tail first).

    This is the pure selection core of :func:`get_samples_from_dump`, split out
    so the binning/argmax logic is unit-testable without any committee
    backend or LAMMPS.
    """
    num_atoms = np.array(structs).shape[1]
    uncertainties = np.array([s.arrays["uncertainty"] for s in structs])
    uncertainties = uncertainties.reshape((bin_count, -1, num_atoms))
    segment_size = uncertainties.shape[1]
    offsets = np.arange(0, bin_count * segment_size, segment_size)
    indices = np.argmax(uncertainties.max(axis=2), axis=1) + offsets
    return [structs[i] for i in indices]


def _build_committee_calculator(backend: str, model_files: list, device: str):
    """Build one ASE calculator spanning the whole committee.

    Each backend's ensemble calculator already reports (or, for MACE, makes
    directly computable) a per-atom, per-component force standard deviation
    across its members -- read back by :func:`_committee_force_stdev`. Built
    once per :func:`get_samples_from_dump` call and reused across every
    trajectory frame (re-instantiating a calculator per frame would reload
    every committee member's weights each time).

    Args:
        backend: ``"mace"``, ``"ace"``, or ``"grace"``.
        model_files: Committee member paths (format depends on ``backend``,
            see ``LammpsSamplingInput.model_files``).
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


def get_samples_from_dump(
    trajectory_file: pathlib.Path,
    model_files: list,
    element_mapping: dict,
    cutoff_radius: int,
    bin_count: int,
    samples_file: pathlib.Path,
    logger: logging.Logger,
    output_full_uncertainties_traj: bool = False,
    full_file: pathlib.Path = None,
    device: str = "cuda",
    backend: str = "mace",
) -> None:
    """Sample from the given trajectory by dividing it into segments and
    sampling the structure with the highest individual local uncertainty value
    out of each segment.

    Args:
        trajectory_file: Path to lammps .dump output file to sample from
        model_files: List of Paths to the committee's model files (format
            depends on ``backend``)
        element_mapping: Dict that maps the atom species number from the given
            trajectory file (as strings) to the corresponding element symbol
        cutoff_radius: cutoff radius (in Angstrom) for calculating the local
            uncertainties
        bin_count: number of segments to divide the trajectory into - if the
            length of the trajectory is not evenly divisible by this number,
            configurations will be removed from the end of the trajectory until
            it is
        samples_file: path to write the sampled structures to
        logger: logger
        output_full_uncertainties_traj: determines whether to additionally
            output the full trajectory with attached uncertainties
        full_file: if output_full_uncertainties_traj=True, the full trajectory
            with attached uncertainties is output to this path
        device: compute device for the committee calculator, where the
            backend's calculator takes one ("cuda" or "cpu"; MACE only)
        backend: ``"mace"``, ``"ace"``, or ``"grace"`` -- see the module
            docstring and :func:`_build_committee_calculator`.
    """
    structs = ase.io.read(
        trajectory_file, format="lammps-dump-text", index=":"
    )

    mod = len(structs) % bin_count
    if mod != 0:
        logger.warning(
            f"Input trajectory size of {len(structs)} is not evenly divisible "
            f"by {bin_count} segments., Cutting last {mod} structures from "
            f"trajectory to ensure evenly sized segments."
        )
        structs = structs[:-mod]

    for atoms in structs:
        for atom in atoms:
            atom.symbol = element_mapping[str(atom.number)]

    model_calc = _build_committee_calculator(backend, model_files, device)

    for atoms in structs:
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

    if output_full_uncertainties_traj and full_file is not None:
        ase.io.write(full_file, structs, format="extxyz")

    max_structs = select_max_uncertainty_frames(structs, bin_count)
    ase.io.write(samples_file, max_structs, format="extxyz")


def main(argv: Optional[list] = None) -> None:
    """Cluster-side entry point: run one LAMMPS committee-sampling job.

    Reads the ``LammpsSamplingInput`` JSON named by ``argv[1]``, runs the MD,
    selects candidates, and writes the ``is_finished.info`` success marker that
    ``SingleScriptTask.finish_check`` polls on the host.
    """
    from lammps_al.run.inputs import LammpsSamplingInput

    argv = sys.argv if argv is None else argv

    inputfile = Path(argv[1])
    calcpath = inputfile.parent
    input_config = LammpsSamplingInput.from_file(inputfile)
    input_config.validate(calcpath)

    # MACE's LAMMPS calculators default to GPU and raise on CPU tensors
    # ("Set MACE_ALLOW_CPU=true ..."). When the run is CPU-bound, opt in so
    # the lmp subprocess (which inherits this env) can compute on CPU.
    if input_config.device == "cpu":
        os.environ["MACE_ALLOW_CPU"] = "true"

    outpath = calcpath / "out"
    os.makedirs(outpath, exist_ok=False)

    lammps_path = input_config.lammps_path
    template_file_name = input_config.template_file
    data_path = calcpath / input_config.data_file
    model_paths = [calcpath / p for p in input_config.model_files]

    config_path = outpath / input_config.lammps_config_name
    samples_file = outpath / "samples.xyz"
    full_traj_file = outpath / "traj+uncertainties.xyz"

    if input_config.backend != "mace":
        # ACE's `pair_style pace` and GRACE's `pair_style grace` read their
        # potential file/directory directly -- no compile step.
        compiled_model_path = model_paths[0]
    elif input_config.lammps_pair_style == "mace":
        compiled_model_path = create_lammps_mace_model(
            model_path=model_paths[0],
            do_return_path=True,
        )
    else:
        compiled_model_path = create_lammps_mliap_model(
            model_path=model_paths[0],
            do_return_path=True,
            enable_cueq=input_config.enable_cueq,
        )

    trajectory = run_lammps_from_template(
        lmp_cmd=lammps_path,
        log_file=outpath / "lmp.log",
        input_file=config_path,
        template_dir=calcpath,
        template_file_name=template_file_name,
        logger=logging.getLogger(),
        lammps_args=input_config.lammps_args,
        lammps_pair_style=input_config.lammps_pair_style,
        model_file=compiled_model_path,
        data_file=data_path,
        seed=input_config.seed,
        dump_file=outpath / "md_sampling.dump",
        # Steps between dumped frames; falls back to every step (the old
        # hardcoded template value) when unset -- see
        # LammpsSamplingInput.sampling_frequency's docstring for why this
        # matters (an unthinned dump can reach tens of GB in minutes on a
        # large system).
        dump_frequency=input_config.sampling_frequency or 1,
        final_data_file=outpath / "surface_final.data",
        do_return_path=True,
        trj_file_key="dump_file",
    )

    get_samples_from_dump(
        trajectory_file=trajectory,
        model_files=model_paths,
        element_mapping=input_config.element_mapping,
        cutoff_radius=input_config.cutoff_radius,
        bin_count=input_config.bin_count,
        samples_file=samples_file,
        output_full_uncertainties_traj=True,
        full_file=full_traj_file,
        logger=logging.getLogger(),
        device=input_config.device,
        backend=input_config.backend,
    )

    # Success marker checked by SingleScriptTask.finish_check on the host.
    (calcpath / "is_finished.info").write_text("LAMMPS sampling finished")


if __name__ == "__main__":
    main()
