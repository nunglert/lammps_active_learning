"""Parse LAMMPS ``log.lammps``-style thermo tables.

Shared by the sampler and annealer diagnostic panels: both read classical-MD
thermo data (step/temp/energy/...) straight from the LAMMPS log the payload
script already wrote, rather than recomputing anything. A single MD run
(sampler) writes one contiguous thermo block; a multi-stage anneal deck
(``EmbeddingAnnealer``'s frozen-core stages) writes one block per ``run``
command, with column sets that can differ between blocks (e.g. an NPT
stage's block adds a ``Volume`` column an NVT stage's doesn't) -- this reads
every block in a log and returns them concatenated, columns unioned, so
callers don't need to know how many stages a given log has.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^\s*Step\s+\S")


def read_lammps_thermo(path):
    """Every thermo block in a LAMMPS log file, concatenated.

    Args:
        path: Path to a LAMMPS log file (e.g. ``out/lmp.log``).

    Returns:
        A ``pandas.DataFrame`` with one row per logged thermo step, an
        added ``run_block`` column (0-indexed, incrementing once per
        ``run``/``minimize`` command's own thermo table), and columns
        unioned across blocks (a column absent from a given block is
        ``NaN`` for that block's rows). Empty (but correctly columned:
        ``["run_block"]``) if the log has no thermo output at all.
    """
    import pandas as pd

    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    blocks = []
    i = 0
    run_block = 0
    while i < len(lines):
        if not _HEADER_RE.match(lines[i]):
            i += 1
            continue
        columns = lines[i].split()
        rows = []
        j = i + 1
        while j < len(lines):
            parts = lines[j].split()
            if len(parts) != len(columns):
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                break
            j += 1
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            df["run_block"] = run_block
            blocks.append(df)
            run_block += 1
        else:
            logger.debug(
                "Thermo header at %s:%d had no parseable data rows "
                "(columns=%s) -- skipping.",
                path,
                i + 1,
                columns,
            )
        i = j

    if not blocks:
        logger.info("No thermo blocks found in %s.", path)
        return pd.DataFrame(columns=["run_block"])

    result = pd.concat(blocks, ignore_index=True, sort=False)
    logger.debug(
        "Read %d thermo block(s), %d total row(s) from %s.",
        run_block,
        len(result),
        path,
    )
    return result
