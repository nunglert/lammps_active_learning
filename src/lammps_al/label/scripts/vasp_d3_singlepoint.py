"""Cluster-side VASP + DFT-D3 single point: VASP for the DFT part, D3 added
on top with simple-dft-d3 (the ``dftd3`` Python package).

The D3 correction is computed here rather than by VASP's own ``IVDW``
because VASP's built-in D3 has no three-body (Axilrod-Teller-Muto) term --
its ``s9`` is a fixed 0 -- while simple-dft-d3 can include it. Labels that
must match a database labelled with simple-dft-d3 including that term need
it computed the same way: leaving it out shifts the energy by a roughly
constant amount per atom and slightly changes the forces.

Input: the same ``input_dft.json`` as remote_handler's ``dft_singlepoint.py``
(``DATABASE``/``launch_command``/``INCAR_kwargs``) plus a ``d3`` block,
``{"method": ..., "damping": "d3zero" | "d3bj", "atm": bool}`` -- written by
:class:`lammps_al.label.evaluator.VaspD3Evaluator`. ``INCAR_kwargs`` must not
enable VASP's own D3 (``IVDW``), or dispersion would be counted twice.

Output, matching ``dft_singlepoint.py``'s contract (so remote_handler's
evaluator collects it unchanged): ``dft_<DATABASE>`` with the total
(DFT + D3) energy/forces plus ``dft_energy``/``d3_energy`` in ``info``, and
the ``is_finished.info`` marker -- only when the SCF converged (see
``remote_handler.scripts.dft_singlepoint.scf_converged``); otherwise an
``scf_unconverged.info`` note and a non-zero exit.

Import-safe (everything runs from :func:`main`). Needs, in the cluster env:
``ase``, ``dftd3`` (lammps_al's ``[label]`` extra), and a remote_handler whose
``dft_singlepoint`` provides ``scf_converged``.
"""

import json
import os
import sys
from pathlib import Path

import ase.io
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp
from ase.units import Bohr, Hartree
from remote_handler.scripts.dft_singlepoint import scf_converged
from remote_handler.scripts.input_models import DftSinglepointInput

_DAMPING_PARAMS = {
    "d3zero": "ZeroDampingParam",
    "d3bj": "RationalDampingParam",
}


def d3_energy_forces(atoms, d3: dict) -> tuple[float, np.ndarray]:
    """D3 dispersion energy (eV) and forces (eV/A) for ``atoms``.

    Args:
        atoms: Structure (cell and ``pbc`` are used as given).
        d3: ``{"method", "damping", "atm"}`` -- ``atm`` switches the
            three-body term; it is required, never left to the library's
            default (which has changed between simple-dft-d3 versions).
    """
    from dftd3 import interface

    param_cls = getattr(interface, _DAMPING_PARAMS[d3["damping"]])
    model = interface.DispersionModel(
        atoms.numbers,
        atoms.positions / Bohr,
        atoms.cell.array / Bohr,
        np.asarray(atoms.pbc, dtype=bool),
    )
    result = model.get_dispersion(
        param_cls(method=d3["method"], atm=bool(d3["atm"])), grad=True
    )
    energy = float(result["energy"]) * Hartree
    forces = -np.asarray(result["gradient"]) * Hartree / Bohr
    return energy, forces


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    calcpath = Path(argv[1])

    raw = json.loads((calcpath / "input_dft.json").read_text())
    d3 = raw.pop("d3")
    if d3.get("damping") not in _DAMPING_PARAMS:
        raise ValueError(
            f"d3.damping must be one of {sorted(_DAMPING_PARAMS)}, got "
            f"{d3.get('damping')!r}"
        )
    input_config = DftSinglepointInput.from_dict(raw)
    input_config.validate(calcpath)
    incar = {k.lower(): v for k, v in input_config.INCAR_kwargs.items()}
    if int(incar.get("ivdw") or 0) != 0:
        raise ValueError(
            "INCAR_kwargs sets IVDW: VASP's D3 would be added on top of this "
            "script's own D3. Leave IVDW unset."
        )

    atoms = ase.io.read(calcpath / input_config.DATABASE)
    # Same rule as dft_singlepoint.py: a readable vasprun.xml means VASP
    # already ran here; a missing or truncated one (crashed run) is redone.
    try:
        ase.io.read(calcpath / "vasprun.xml")
        already_ran = True
    except Exception:
        already_ran = False
    if already_ran:
        raise Exception("Calculation already ran succesfully. Skipping it.")
    print(atoms)

    paw_path = os.environ.get("VASP_PP_PATH", None)
    print(f"Using PAW potentials from: {paw_path}")
    print(f"D3 on top of VASP: {d3}")
    atoms.calc = Vasp(
        directory=calcpath,
        command=input_config.launch_command,
        txt="vasp.out",
        **input_config.INCAR_kwargs,
    )
    dft_energy = atoms.get_potential_energy()
    dft_forces = np.asarray(atoms.calc.results["forces"])

    converged, reason = scf_converged(calcpath)
    if not converged:
        message = f"SCF not converged, not labelling this structure: {reason}"
        (calcpath / "scf_unconverged.info").write_text(message + "\n")
        sys.exit(message)

    d3_energy, d3_forces = d3_energy_forces(atoms, d3)

    atoms.calc = SinglePointCalculator(
        atoms,
        energy=dft_energy + d3_energy,
        forces=dft_forces + d3_forces,
    )
    atoms.info["dft_energy"] = dft_energy
    atoms.info["d3_energy"] = d3_energy
    ase.io.write(calcpath / ("dft_" + input_config.DATABASE), atoms)

    (calcpath / "is_finished.info").write_text("Finished VASP + D3 run")


if __name__ == "__main__":
    main()
