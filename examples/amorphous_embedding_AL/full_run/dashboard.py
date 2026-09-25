"""Build the analysis dashboard for this campaign's ``calchome``.

Thin wrapper -- see ``lammps_al.dashboard.cli`` for the actual
implementation and flags (``--only``/``--exclude``/``--species``/
``--output``). Run as ``python dashboard.py`` from this directory, or
``python experiments/rough_cu_water_al/dashboard.py`` from the repo root.
"""

import sys
from pathlib import Path

from lammps_al.dashboard.cli import main

_calchome = Path(__file__).parent / "calchome"

if __name__ == "__main__":
    sys.exit(main([str(_calchome), *sys.argv[1:]]))
