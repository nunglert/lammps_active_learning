import json
import sys
from pathlib import Path

import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_dft.json"
input_args = json.load(open(INPUTFILE, "r"))

DATABASE = input_args["DATABASE"]
STUB = input_args["STUB"]
launch_command = input_args["launch_command"]
VASP_incar_kwargs = input_args["INCAR_kwargs"]

atoms = ase.io.read(calcpath / DATABASE)

try:
    at = ase.io.read(calcpath / "vasprun.xml")
    repeat_calculation = False
    atoms.info["is_finished"] = True
except:
    repeat_calculation = True
    atoms.info["is_finished"] = False


if not repeat_calculation:
    raise Exception("Calculation already ran succesfully. Skipping it.")

print(atoms)

VASP_calculator_args = {
    "command": launch_command,
    "txt": "-",
}

VASP_config = {**VASP_calculator_args, **VASP_incar_kwargs}

Vasp.xc_defaults["r2scan"] = {"metagga": "R2SCAN"}
atoms.calc = Vasp(directory=calcpath, **VASP_config)

if not STUB:
    atoms.calc = SinglePointCalculator(
        atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
    )

atoms.info["is_finished"] = True
ase.io.write(calcpath / ("dft_" + DATABASE), atoms)

with open(calcpath / "is_finished.info", "w") as f:
    f.write(str(True))
