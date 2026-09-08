"""Cluster-side LAMMPS frequency-sampling payload (no uncertainties).

Runs one LAMMPS MD trajectory driven by a MACE model (exactly as
``lammps_sampling.py`` does), but instead of scoring every frame against a
committee it writes every ``sampling_frequency``-th frame of the trajectory
straight to ``out/samples.xyz`` as extxyz.

This is the *selector-agnostic* counterpart to ``lammps_sampling.py``: it does
not embed any uncertainty/selection logic. Selection between candidates is
left to the caller -- the intent is to feed frames to one of
``remote_handler.workflows.selectors`` (e.g. ``StrideSelector``) downstream
rather than doing inline committee-uncertainty selection here.

Shipped and executed on the cluster by a ``LammpsSampler``-style
``SingleScriptTask`` -- it is not imported as a module in the normal flow
(``main`` runs it from ``sys.argv``), but its functions are import-safe so they
can be unit-tested with only ``ase`` + ``numpy`` + ``jinja2`` present.

Reuses the shared LAMMPS-run helpers from ``lammps_sampling.py``
(``parse_jinja_template``, ``create_lammps_mliap_model``/
``create_lammps_mace_model``, ``run_lammps_from_template``), which are
already import-safe (the heavy ``torch``/``mace`` imports are lazy). Like
``lammps_sampling.py``, the driving-model compile route is chosen per run via
``lammps_pair_style`` (``"mliap"`` default/production, ``"mace"``
CPU-testable/opt-in) -- see that module's docstring for why.
"""

import logging
import os
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Optional

import ase.io
import numpy as np  # noqa: F401  (kept for parity/warmup of ase numpy backend)

from lammps_al.run.sampler import remap_elements
from lammps_al.run.scripts.lammps_sampling import (
    create_lammps_mace_model,
    create_lammps_mliap_model,
    run_lammps_from_template,
)

logger = logging.getLogger(__name__)

DEFAULT_LAMMPS_ARGS = "-k on g 1 -sf kk -pk kokkos newton on neigh half"


@dataclass
class DumpSamplingInput:
    """Input configuration for ``lammps_dump_sampling.py``.

    Mirrors ``lammps_al.run.inputs.LammpsSamplingInput`` minus the
    uncertainty-specific fields (``cutoff_radius``, ``bin_count``,
    ``element_mapping``), adding ``sampling_frequency`` in their place.

    NOTE: this duplicates the schema that logically belongs beside
    ``LammpsSamplingInput`` in ``inputs.py``; it lives here so as not to modify
    existing files. See the inconsistency summary in the PR/commit notes.

    Attributes:
        lammps_path: Path to the LAMMPS executable
        template_file: Name of the LAMMPS input template file
        data_file: Name of the initial structure data file
        model_files: List of MACE model files (needed only to run the MD;
            no committee scoring happens here)
        sampling_frequency: Write every Nth trajectory frame to the output
        seed: Random seed for LAMMPS
        lammps_config_name: Name for the generated LAMMPS input file
        lammps_args: LAMMPS accelerator/package flags
        device: torch device for compiling the MACE model
        lammps_pair_style: ``"mliap"`` (default, production) or ``"mace"``
            (CPU-testable, opt-in) -- see ``lammps_sampling.py``.
        enable_cueq: Convert the MACE model to cuequivariance before
            compiling. Only used when ``lammps_pair_style="mliap"``.
    """

    lammps_path: str
    template_file: str
    data_file: str
    model_files: list
    sampling_frequency: int
    seed: int
    lammps_config_name: str = "lammps.in"
    lammps_args: str = DEFAULT_LAMMPS_ARGS
    device: str = "cuda"
    lammps_pair_style: str = "mliap"
    enable_cueq: bool = True

    def validate(self, root: Optional[Path] = None) -> None:
        if self.sampling_frequency <= 0:
            raise ValueError(
                f"sampling_frequency must be positive, got "
                f"{self.sampling_frequency}"
            )
        if self.device not in ("cpu", "cuda"):
            raise ValueError(
                f"device must be 'cpu' or 'cuda', got {self.device!r}"
            )
        if self.lammps_pair_style not in ("mliap", "mace"):
            raise ValueError(
                "lammps_pair_style must be 'mliap' or 'mace', got "
                f"{self.lammps_pair_style!r}"
            )
        if root is not None:
            for name in ("data_file", "template_file"):
                if not (root / getattr(self, name)).exists():
                    raise FileNotFoundError(
                        f"{name} not found under {root}: {getattr(self, name)}"
                    )
            for m in self.model_files:
                if not (root / m).exists():
                    raise FileNotFoundError(
                        f"model file not found under {root}: {m}"
                    )


def load_dump_frames(
    trajectory_file: Path,
) -> list:
    """Read every frame of a LAMMPS dump as a list of ASE Atoms."""
    return list(
        ase.io.read(trajectory_file, format="lammps-dump-text", index=":")
    )


def select_frequency_frames(structs: list, sampling_frequency: int) -> list:
    """Return every ``sampling_frequency``-th frame, preserving order.

    Frame 0 is always included (trajectories are 0-indexed). This is the pure
    selection core of :func:`main`, split out so the stride logic is
    unit-testable without LAMMPS. Downstream, this list is what a
    ``remote_handler`` ``ConfigSelector`` would be handed instead.
    """
    return structs[::sampling_frequency]


def write_samples(
    structs: list, samples_file: Path, element_mapping: dict = None
) -> None:
    """Write sampled structures to an extxyz file."""
    if element_mapping is not None:
        remapped_structs = remap_elements(structs, element_mapping)
    else:
        remapped_structs = structs
    ase.io.write(samples_file, remapped_structs, format="extxyz")


def _load_input(input_file: Path) -> dict:
    """Load the sampling input JSON as a plain dict.

    Mirrors ``SSPDSampler``/its payload: the input is a *generic* document --
    script-backend keys plus any schedule keys (``temperature``, ``timestep``,
    ``n_run``, ...) -- rather than a rigid typed schema, so arbitrary schedule
    parameters reach the LAMMPS template render without a schema change.
    """
    import json

    with open(input_file) as f:
        return json.load(f)


# Keys consumed by this script backend (paths/execution concerns), as opposed to
# keys that are schedule/template parameters forwarded into the LAMMPS input.
_SCRIPT_KEYS = {
    "lammps_path",
    "template_file",
    "lammps_args",
    "device",
    "model_files",
    "data_file",
    "lammps_config_name",
    "sampling_frequency",
    "bin_count",
    "cutoff_radius",
    "element_mapping",
    "lammps_pair_style",
    "enable_cueq",
}


def main(argv: Optional[list] = None) -> None:
    """Cluster-side entry point: run one LAMMPS frequency-sampling job.

    Reads the input JSON named by ``argv[1]``, runs the MD, writes every
    ``sampling_frequency``-th frame to ``samples.xyz``, and writes the
    ``is_finished.info`` success marker that ``SingleScriptTask.finish_check``
    polls on the host.

    Every non-backend key in the JSON (e.g. ``temperature``/``timestep``/
    ``n_run`` from a ``Schedule``) is forwarded into the LAMMPS template
    render, so templates may use ``{{ temp }}``, ``{{ n_run }}``, etc.
    """
    import json
    import sys

    argv = sys.argv if argv is None else argv

    input_file = Path(argv[1])
    calcpath = input_file.parent
    config = _load_input(input_file)

    if config.get("device") == "cpu":
        os.environ["MACE_ALLOW_CPU"] = "true"

    outpath = calcpath / "out"
    os.makedirs(outpath, exist_ok=False)

    data_path = calcpath / config["data_file"]
    model_paths = [calcpath / m for m in config["model_files"]]

    config_path = outpath / config.get("lammps_config_name", "lammps.in")
    samples_file = outpath / "samples.xyz"
    trajectory_file = outpath / "md_sampling.dump"

    pair_style = config.get("lammps_pair_style", "mliap")
    if pair_style == "mace":
        compiled_model_path = create_lammps_mace_model(
            model_path=model_paths[0],
            do_return_path=True,
        )
    else:
        compiled_model_path = create_lammps_mliap_model(
            model_path=model_paths[0],
            do_return_path=True,
            enable_cueq=config.get("enable_cueq", True),
        )

    # Payload-managed render context ...
    render_kwargs = {
        "model_file": compiled_model_path,
        "data_file": data_path,
        "seed": config.get("seed"),
        "dump_file": trajectory_file,
        "final_data_file": outpath / "surface_final.data",
        "lammps_pair_style": pair_style,
    }
    # ... plus every non-backend key from the schedule/input, verbatim.
    for key, value in config.items():
        if key not in _SCRIPT_KEYS:
            render_kwargs[key] = value

    run_lammps_from_template(
        lmp_cmd=config["lammps_path"],
        log_file=outpath / "lmp.log",
        input_file=config_path,
        template_dir=calcpath,
        template_file_name=config["template_file"],
        logger=logger,
        lammps_args=config.get("lammps_args", DEFAULT_LAMMPS_ARGS),
        **render_kwargs,
    )

    structs = load_dump_frames(trajectory_file)
    sampled = select_frequency_frames(
        structs, int(config.get("sampling_frequency", 1))
    )
    write_samples(
        sampled, samples_file, element_mapping=config["element_mapping"]
    )

    # Success marker checked by SingleScriptTask.finish_check on the host.
    (calcpath / "is_finished.info").write_text("LAMMPS sampling finished")


if __name__ == "__main__":
    main()
