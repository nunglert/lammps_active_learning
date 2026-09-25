"""Amorphous matrix embedding -- extraction entry point.

Entry point for remote execution via ``SingleScriptTask``. Reads a JSON
input file (path passed as ``sys.argv[1]``), reads an uncertainty-tagged
trajectory, extracts one carved box per region of interest (ROI) per frame,
and writes each to ``out/`` as both ``lammps-data`` (for the anneal step) and
``extxyz``, alongside a ``manifest.json`` describing every carved box.

Streams the trajectory via ``ase.io.iread`` in two passes rather than
``ase.io.read(index=":")`` once -- the input trajectory is a full-frame-count,
uncertainty-tagged dump rather than a handful of pre-picked peak frames, by
design (see ``CommitteeUncertaintySelector``'s module docstring), so it can
be large enough that loading every frame into memory at once is an avoidable
cost given each phase only ever needs one frame at a time.
Pass 1 (:func:`find_rois` only, no carving) needs every frame exactly once,
in order -- a plain streaming scan. Pass 2 needs only whichever frames
survived ``max_num_structures``' cap, so it re-scans from the start and
skips any frame not in that (typically much smaller) set, rather than
keeping the full trajectory resident between passes. This trades one extra
sequential file read for peak memory bounded by a single frame instead of
the whole trajectory.
"""

import json
import logging
import os
import sys
import time
from itertools import groupby
from pathlib import Path

import ase.io
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

INPUTFILE = Path(sys.argv[1])
calcpath = INPUTFILE.parent

from remote_handler.exceptions import ScriptInputError

from lammps_al.embedding.inputs import MatrixEmbeddingInput
from lammps_al.embedding.scripts.cleanup import cleanup
from lammps_al.embedding.scripts.geometry import (
    BoxDims,
    cap_rois_by_uncertainty,
    carve_box,
    find_rois,
)
from lammps_al.embedding.scripts.surface_mesh import detect_normal

input_config = MatrixEmbeddingInput.from_file(INPUTFILE)
input_config.validate(calcpath)

outpath = calcpath / "out"
# exist_ok so a re-executed/resumed task on a persisted calcdir does not
# abort before doing any work.
os.makedirs(outpath, exist_ok=True)

trajectory_path = calcpath / input_config.trajectory

solvent = None
if input_config.cleanup_mode == "molecule":
    if input_config.solvent_label is not None:
        from ase.build import molecule

        solvent = molecule(input_config.solvent_label)
    else:
        solvent = ase.io.read(calcpath / input_config.solvent_file)

box_dims = BoxDims(
    x=input_config.box_x,
    y=input_config.box_y,
    z_below=input_config.box_z_below,
    z_above=input_config.box_z_above,
)

# Phase 1: find every ROI across the whole trajectory -- cheap (find_rois
# only, no carve/OVITO/cleanup) -- so max_num_structures can cap by global
# uncertainty rank before any of the expensive per-ROI work runs.
logger.info("scanning trajectory %s for ROIs ...", input_config.trajectory)
t_start = time.monotonic()
candidates: list[tuple[int, int, float]] = []
n_frames = 0

for frame_idx, atoms in enumerate(ase.io.iread(trajectory_path, index=":")):
    if "uncertainty" not in atoms.arrays:
        raise ScriptInputError(
            f"Trajectory frame {frame_idx} of "
            f"'{input_config.trajectory}' has no per-atom 'uncertainty' "
            "array; the embedding sampler requires an uncertainty-tagged "
            "trajectory."
        )
    n_frames += 1

    t_frame = time.monotonic()
    roi_indices = find_rois(
        atoms,
        input_config.threshold,
        input_config.cluster_cutoff,
        phase_species=input_config.phase_species,
        phase_cutoff=input_config.phase_cutoff,
    )
    uncertainty = atoms.arrays["uncertainty"]
    candidates.extend(
        (frame_idx, int(roi_index), float(uncertainty[roi_index]))
        for roi_index in roi_indices
    )
    logger.info(
        "frame %d: %d ROI(s) (find_rois %.1fs, %d candidates so far, "
        "%.1fs elapsed)",
        frame_idx,
        len(roi_indices),
        time.monotonic() - t_frame,
        len(candidates),
        time.monotonic() - t_start,
    )

logger.info(
    "found %d candidate ROI(s) across %d frames in %.1fs",
    len(candidates),
    n_frames,
    time.monotonic() - t_start,
)

capped = cap_rois_by_uncertainty(candidates, input_config.max_num_structures)
if len(capped) < len(candidates):
    logger.info(
        "max_num_structures=%d: keeping the %d highest-uncertainty ROI(s), "
        "dropping %d",
        input_config.max_num_structures,
        len(capped),
        len(candidates) - len(capped),
    )

# Phase 2: re-scan the trajectory, carving/cleaning/writing only the frames
# that still have a surviving ROI after the cap. `capped` is already sorted
# by (frame_idx, roi_index) (cap_rois_by_uncertainty's contract), so a
# single second streaming pass -- skipping any frame with nothing to do --
# suffices; no random access into the trajectory is needed.
rois_by_frame = {
    frame_idx: list(group)
    for frame_idx, group in groupby(capped, key=lambda c: c[0])
}

manifest = []
count = 0
t_carve = time.monotonic()

for frame_idx, atoms in enumerate(ase.io.iread(trajectory_path, index=":")):
    frame_rois = rois_by_frame.get(frame_idx)
    if not frame_rois:
        continue

    for _, roi_index, _unc in frame_rois:
        roi_position_orig = atoms.get_positions()[roi_index]

        normal = None
        if input_config.topology_mode in ("surface", "interface"):
            remove_species = (
                input_config.remove_species
                if input_config.topology_mode == "interface"
                else None
            )
            normal = detect_normal(
                atoms,
                roi_position_orig,
                remove_species=remove_species,
                phase_species=input_config.phase_species,
                smoothing_level=input_config.smoothing_level,
                radius=input_config.mesh_radius,
                patch_radius=(
                    input_config.normal_patch_radius
                    if input_config.normal_patch_radius is not None
                    else input_config.protect_radius
                ),
            )

        carved, roi_position = carve_box(
            atoms, roi_index, box_dims, input_config.buffer, normal
        )

        cleaned = cleanup(
            carved,
            input_config.cleanup_mode,
            roi_position,
            input_config.protect_radius,
            solvent=solvent,
            stoichiometry=input_config.stoichiometry,
        )

        data_name = f"embedded_{count}.data"
        extxyz_name = f"embedded_{count}.extxyz"

        write_kwargs = {}
        if input_config.specorder is not None:
            write_kwargs["specorder"] = input_config.specorder
        ase.io.write(
            outpath / data_name,
            cleaned,
            format="lammps-data",
            **write_kwargs,
        )
        ase.io.write(outpath / extxyz_name, cleaned, format="extxyz")

        manifest.append(
            {
                "index": count,
                "frame": frame_idx,
                "roi_index": int(roi_index),
                "roi_position": roi_position.tolist(),
                # Where this ROI actually sat in the source trajectory frame,
                # before carving/re-centering -- distinct from
                # "roi_position" above (the ROI's center *within its own
                # carved box*, always ~box_size/2 regardless of where the
                # ROI came from -- load-bearing for EmbeddingAnnealer's
                # frozen-core placement, so kept as-is rather than repointed
                # at this). Only this field is useful for a spatial map of
                # where ROIs were extracted from across the campaign.
                "roi_position_orig": roi_position_orig.tolist(),
                "normal": normal.tolist() if normal is not None else None,
                "data_file": data_name,
                "extxyz_file": extxyz_name,
            }
        )
        count += 1
        if count % 10 == 0 or count == len(capped):
            logger.info(
                "carved %d/%d boxes (%.1fs elapsed)",
                count,
                len(capped),
                time.monotonic() - t_carve,
            )

with open(outpath / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

logger.info(
    "done: %d boxes extracted from %d frames in %.1fs",
    count,
    n_frames,
    time.monotonic() - t_start,
)

# Success marker checked by SingleScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text(
    f"amorphous matrix embedding finished ({count} boxes extracted)"
)
