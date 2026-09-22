"""Amorphous matrix embedding -- extraction entry point.

Entry point for remote execution via ``SingleScriptTask``. Reads a JSON
input file (path passed as ``sys.argv[1]``), reads an uncertainty-tagged
trajectory, extracts one carved box per region of interest (ROI) per frame,
and writes each to ``out/`` as both ``lammps-data`` (for the anneal step) and
``extxyz``, alongside a ``manifest.json`` describing every carved box.
"""

import json
import os
import sys
from pathlib import Path

import ase.io
import numpy as np

INPUTFILE = Path(sys.argv[1])
calcpath = INPUTFILE.parent

from remote_handler.exceptions import ScriptInputError

from lammps_al.embedding.inputs import MatrixEmbeddingInput
from lammps_al.embedding.scripts.cleanup import cleanup
from lammps_al.embedding.scripts.geometry import BoxDims, carve_box, find_rois
from lammps_al.embedding.scripts.surface_mesh import detect_normal

input_config = MatrixEmbeddingInput.from_file(INPUTFILE)
input_config.validate(calcpath)

outpath = calcpath / "out"
# exist_ok so a re-executed/resumed task on a persisted calcdir does not
# abort before doing any work.
os.makedirs(outpath, exist_ok=True)

trajectory = ase.io.read(calcpath / input_config.trajectory, index=":")

for frame_idx, atoms in enumerate(trajectory):
    if "uncertainty" not in atoms.arrays:
        raise ScriptInputError(
            f"Trajectory frame {frame_idx} of "
            f"'{input_config.trajectory}' has no per-atom 'uncertainty' "
            "array; the embedding sampler requires an uncertainty-tagged "
            "trajectory."
        )

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

manifest = []
count = 0

for frame_idx, atoms in enumerate(trajectory):
    roi_indices = find_rois(
        atoms,
        input_config.threshold,
        input_config.cluster_cutoff,
        phase_species=input_config.phase_species,
        phase_cutoff=input_config.phase_cutoff,
    )
    for roi_index in roi_indices:
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
                "normal": normal.tolist() if normal is not None else None,
                "data_file": data_name,
                "extxyz_file": extxyz_name,
            }
        )
        count += 1

with open(outpath / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Success marker checked by SingleScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text(
    f"amorphous matrix embedding finished ({count} boxes extracted)"
)
