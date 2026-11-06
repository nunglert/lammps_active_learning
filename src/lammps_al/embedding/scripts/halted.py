"""Recovering a sample from an anneal that the deck's ``fix halt`` stopped.

``embedding_anneal.in.j2`` stops the run (``fix ... halt ... error hard``)
when one mobile atom's kinetic energy or force runs away -- the potential
has hit a configuration it handles badly, which is exactly what the next
training set is missing. Instead of losing the box, ``embedding_anneal.py``
then returns the last dumped frame before the halt: the configuration the
potential failed from, not the blown-up state after it.
"""

import io
import re
from pathlib import Path
from typing import Optional

import ase
import ase.io

_HALT_RE = re.compile(r"Fix halt condition for fix-id \S+ met on step (\d+)")


def halt_step(log_file: Path) -> Optional[int]:
    """The step at which ``fix halt`` stopped the run logged in
    ``log_file``, or ``None`` if it did not."""
    if not Path(log_file).exists():
        return None
    with open(log_file, errors="replace") as f:
        for line in f:
            match = _HALT_RE.search(line)
            if match:
                return int(match.group(1))
    return None


def last_frame_before(
    dump_file: Path, step: int, species: list[str]
) -> Optional[ase.Atoms]:
    """The last frame of a LAMMPS text dump with a timestep below ``step``,
    with chemical symbols from ``species`` (LAMMPS type order), or ``None``
    if there is none.
    """
    if not Path(dump_file).exists():
        return None
    frame, current = None, []
    with open(dump_file) as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                if current:
                    frame = current
                current = [line]
                timestep = int(next(f))
                if timestep >= step:
                    current = []
                    break
                current.append(f"{timestep}\n")
            elif current:
                current.append(line)
    if current:
        frame = current
    if frame is None:
        return None
    atoms = ase.io.read(io.StringIO("".join(frame)), format="lammps-dump-text")
    atoms.set_chemical_symbols([species[t - 1] for t in atoms.numbers])
    return atoms
