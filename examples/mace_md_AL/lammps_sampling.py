import json
import logging
import os
import pathlib
import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import ase
import jinja2
import numpy as np
import torch
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from mace.calculators.mace import MACECalculator
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False


calcpath = Path(sys.argv[1]).parent
INPUTFILE = Path(sys.argv[1])
input_args = json.load(open(INPUTFILE, "r"))

outpath = calcpath / "out"
os.makedirs(outpath, exist_ok=False)

lammps_path = Path(input_args["lammps_path"])

template_file_name = input_args["template_file"]
data_path = calcpath / input_args["data_file"]
model_paths = [calcpath / p for p in input_args["model_files"]]

config_path = outpath / input_args["lammps_config_name"]
samples_file = outpath / "samples.xyz"
full_traj_file = outpath / "traj+uncertainties.xyz"

element_mapping = input_args["element_mapping"]
cutoff_radius = input_args["cutoff_radius"]
bin_count = input_args["bin_count"]
seed = input_args["seed"]


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
) -> Optional[pathlib.Path]:
    """Compile a MACE model for use in LAMMPS

    Args:
        model_path: Path to the trained MACE model
        head: Which model head to select for compilation
        float32: Compile a 32-bit model
        do_return_path: Return a pathlib.Path for the compiled model
        out_path: Path to file to save the compiled model to

    Returns:
        If requested, path to the compiled model
    """
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
    head = "Default" if head is None else head
    if float32:
        model = model.float().to("cpu")
    else:
        model = model.double().to("cpu")
    model = run_e3nn_to_cueq(deepcopy(model))
    model.lammps_mliap = True
    lammps_model = LAMMPS_MLIAP_MACE(model, head=head)
    torch.save(lammps_model, compiled_path.resolve())
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
    try:
        subprocess.run(
            f"{lmp_cmd} -in {input_file.resolve()} -k on g 1 -sf kk -pk kokkos newton on neigh half "
            f"-log {log_file.resolve()}",
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


def get_samples_from_dump(
    trajectory_file: pathlib.Path,
    model_files: list[pathlib.Path],
    element_mapping: dict,
    cutoff_radius: int,
    bin_count: int,
    samples_file: pathlib.Path,
    logger: logging.Logger,
    output_full_uncertainties_traj: bool = False,
    full_file: pathlib.Path = None,
) -> None:
    """Sample from the given trajectory by dividing it into segments and sampling
    the structure with the highest individual local uncertainty value out of each
    segement.

    Args:
        trajectory_file: Path to lammps .dump output file to sample from
        model_files: List of Paths to models to calculate committee uncertainties
            with
        element_mapping: Dict that maps the atom species number from the given
            trajectory file (as strings) to the corresponding element symbol
        cutoff_radius: cutoff radius (in Angstrom) for calculating the local
            uncertainties
        bin_count: number of segments to divide the trajectory into - if the length
            of the trajectory is not evenly divisible by this number, configurations
            will be removed from the end of the trajectory until it is
        samples_file: path to write the sampled structures to
        logger: logger
        output_full_uncertainties_traj: determines whether to additionally output
            the full trajectory with attached uncertainties
        full_file: if output_full_uncertainties_traj=True, the full trajectory
            with attached uncertainties is output to this path
    """

    structs = ase.io.read(
        trajectory_file, format="lammps-dump-text", index=":"
    )

    mod = len(structs) % bin_count
    if mod != 0:
        logger.warning(
            f"Input trajectory size of {len(structs)} is not evenly divisible by "
            f"{bin_count} segments., Cutting last {mod} structures from trajectory "
            f"to ensure evenly sized segments."
        )
        structs = structs[:-mod]

    for atoms in structs:
        for atom in atoms:
            atom.symbol = element_mapping[str(atom.number)]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model_calc = MACECalculator(
            model_paths=model_files,
            device="cuda",
            default_dtype="float32",
            enable_cueq=CUET_AVAILABLE,
        )

    for atoms in structs:
        a = atoms.copy()
        a.calc = model_calc
        a.get_potential_energy()
        forces = a.calc.results["forces_comm"]
        stdev = np.std(forces, axis=0, ddof=1)
        struct_uncertainties = aggregate_uncertainties(
            atom_pos=a.get_positions(),
            unit_cell=a.cell,
            boundary_conditions=a.pbc,
            atomic_uncertainties=stdev,
            r_cut=cutoff_radius,
        )
        atoms.arrays["uncertainties"] = struct_uncertainties

    if output_full_uncertainties_traj and full_file is not None:
        ase.io.write(full_file, structs, format="extxyz")

    num_atoms = np.array(structs).shape[1]
    uncertainties = np.array([s.arrays["uncertainties"] for s in structs])
    uncertainties = uncertainties.reshape((bin_count, -1, num_atoms))
    segment_size = uncertainties.shape[1]
    offsets = np.arange(0, bin_count * segment_size, segment_size)
    indices = np.argmax(uncertainties.max(axis=2), axis=1) + offsets
    max_structs = [structs[i] for i in indices]
    ase.io.write(samples_file, max_structs, format="extxyz")


mliap_path = create_lammps_mliap_model(
    model_path=model_paths[0], do_return_path=True
)

trajectory = run_lammps_from_template(
    lmp_cmd=lammps_path,
    log_file=outpath / "lmp.log",
    input_file=config_path,
    template_dir=calcpath,
    template_file_name=template_file_name,
    logger=logging.getLogger(),
    model_file=mliap_path,
    data_file=data_path,
    seed=seed,
    dump_file=outpath / "md_sampling.dump",
    final_data_file=outpath / "surface_final.data",
    do_return_path=True,
    trj_file_key="dump_file",
)

get_samples_from_dump(
    trajectory_file=trajectory,
    model_files=model_paths,
    element_mapping=element_mapping,
    cutoff_radius=cutoff_radius,
    bin_count=bin_count,
    samples_file=samples_file,
    output_full_uncertainties_traj=True,
    full_file=full_traj_file,
    logger=logging.getLogger(),
)
