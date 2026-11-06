"""Stream selected per-atom columns out of a (large) extxyz trajectory.

``ase.io.read``/``iread`` parse every column of every atom and build a full
``Atoms`` per frame -- far more than the uncertainty panels need from a
full-system sampler dump (hundreds of thousands of atoms per frame, GBs per
iteration). This reads each frame's header to locate the requested
``Properties`` columns, then hands only those to pandas' C parser, one
frame at a time.
"""

from __future__ import annotations

import io
from itertools import islice

# extxyz Properties type code -> numpy dtype.
_DTYPES = {"R": float, "I": int, "S": object, "L": object}


def _column_layout(properties: str) -> dict:
    """``{name: (first_column, width, type_code)}`` from a Properties
    string like ``species:S:1:pos:R:3:uncertainty:R:1``."""
    fields = properties.split(":")
    layout, col = {}, 0
    for name, code, width in zip(fields[::3], fields[1::3], fields[2::3]):
        layout[name] = (col, int(width), code)
        col += int(width)
    return layout


def iter_columns(path, columns, *, info_keys=()):
    """Yield ``(info, arrays)`` per frame of the extxyz file at ``path``.

    Args:
        path: The extxyz file.
        columns: Per-atom columns to read: a Properties name (``"species"``,
            ``"uncertainty"``) for the whole column, or ``(name, component)``
            for one component of a vector column (``("pos", 2)`` = z).
        info_keys: Frame-level ``info`` keys to return (missing ones are
            simply absent from ``info``).

    Yields:
        ``info`` (dict of the requested keys) and ``arrays`` (dict mapping
        each entry of ``columns`` to a 1-D numpy array, or 2-D for a whole
        multi-component column).
    """
    import pandas as pd
    from ase.io.extxyz import key_val_str_to_dict

    with open(path, "rb") as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            if not header.strip():
                continue
            n_atoms = int(header)
            comment = key_val_str_to_dict(fh.readline().decode())
            layout = _column_layout(comment["Properties"])

            wanted = []  # (key, [file columns], type code)
            for c in columns:
                name, comp = (c, None) if isinstance(c, str) else c
                if name not in layout:
                    raise KeyError(
                        f"{path}: no {name!r} column (has "
                        f"{sorted(layout)})."
                    )
                first, width, code = layout[name]
                cols = (
                    list(range(first, first + width))
                    if comp is None
                    else [first + comp]
                )
                wanted.append((c, cols, code))
            usecols = sorted({i for _, cols, _ in wanted for i in cols})

            block = b"".join(islice(fh, n_atoms))
            df = pd.read_csv(
                io.BytesIO(block),
                sep=r"\s+",
                header=None,
                usecols=usecols,
                dtype={
                    i: _DTYPES[code] for _, cols, code in wanted for i in cols
                },
                engine="c",
            )
            arrays = {}
            for key, cols, _ in wanted:
                values = df[cols].to_numpy()
                arrays[key] = values[:, 0] if len(cols) == 1 else values
            info = {k: comment[k] for k in info_keys if k in comment}
            yield info, arrays
