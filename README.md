# lammps_active_learning

The **composition root** for LAMMPS active-learning (AL) projects. It owns the
concrete LAMMPS sampler and the cluster-side payload it ships, and builds on
`remote_handler` for everything generic:

```
lammps_active_learning   ← this repo: the LAMMPS MD Sampler + payload + wiring
  └──▶ remote_handler     orchestration loop, Sampler/Evaluator/Trainer ABCs,
                          concrete evaluators (VASP/ORCA) and trainers (MACE),
                          remote-execution machinery (SingleScriptTask, SLURM)
```

`remote_handler` deliberately does **not** contain concrete samplers — those are
a *project* concern. The **LAMMPS sampler lives here** (`lammps_al.run.sampler`),
implementing `remote_handler`'s `Sampler` interface.

## What lives here

- `lammps_al.run.sampler` — `LammpsSampler`, a native `remote_handler.Sampler`.
  Per AL iteration it compiles the committee's MACE model for LAMMPS (MLIAP),
  renders the user's `lammps.in` jinja2 template, runs a LAMMPS MD trajectory on
  the cluster via a `remote_handler` `SingleScriptTask`, scores every frame's
  local force-uncertainty against the committee, and returns the most-uncertain
  frame from each of `num_samples` trajectory segments. Needs the `[run]` extra.
  Supports `stub=True` for cluster-free dry runs.
- `lammps_al.run.inputs.LammpsSamplingInput` — the JSON contract between the
  host-side sampler and the cluster-side payload (a `remote_handler.ScriptInput`
  subclass).
- `lammps_al.run.scripts.lammps_sampling.py` — the cluster-side payload shipped
  and executed by the sampler; `lammps_al.run.scripts.script_dir()` locates it
  from a non-editable install. It runs in the remote conda env (needs a MACE
  stack + a LAMMPS build with the MLIAP/MACE interface), not on the host.

## Uncertainty-driven sampling

The sampler is a MACE **committee** driver: it runs one MD trajectory, evaluates
per-atom force disagreement across the committee, aggregates it locally within
`uncertainty_cutoff_radius`, splits the trajectory into `num_samples` segments,
and emits the peak-uncertainty frame of each segment as a labelling candidate.
Reducing that candidate pool further (diversity/novelty filtering) is a
`remote_handler` *selector* concern, not the sampler's.

## Install (local dev)

`remote_handler` is pinned by SSH git-URL. With an SSH key to the GitLab
instance it installs directly; otherwise install a sibling checkout editable,
then this project with `--no-deps`:

```bash
# Either (SSH key present):
pip install -e ".[run]"

# Or (offline / sibling checkout):
pip install -e ../remote_handler        # provides the [run] dependency
pip install -e ".[dev]" --no-deps
```

The host side needs only `remote_handler` + `ase` + `numpy`. The **cluster**
conda env that runs the payload additionally needs `lammps_al` (for the input
schema), `mace-torch` + `cuequivariance`, `torch`, and a LAMMPS build with the
MLIAP/MACE interface.

Versions come from git tags via `setuptools-scm`: tag `vX.Y.Z` and
`lammps_al.__version__` follows (refreshes on reinstall, not on a bare pull).

## Tests

```bash
pytest                    # everything
pytest -m integration     # the sampler/input tests (need remote_handler)
```

All tests self-skip (`importorskip("remote_handler")`) when the `[run]`
producer is absent; the stub-mode sampler tests otherwise run with no cluster,
SSH, LAMMPS, or MACE.

## Run an AL loop

`examples/mace_md_AL/` is a runnable Cu/Co surface reference (its own cluster
`config/`, committee `inputs/models/`, and `run_al.py`). It bundles hard-coded
paths from a prior user — adapt them before running. Start with `stub=True`.

## Conventions

- Formatting is enforced by pre-commit: **black** and **isort** both at
  `--line-length 79` (isort `--profile black`).
- `remote_handler` labels ground-truth energies/forces with `REF_energy` /
  `REF_forces`; the sampler tags candidates with `max_uncertainty` /
  `mean_uncertainty`.
- The run side needs `REMOTE_HANDLER_CONFIG_DIR` (SSH/cluster config) and, for
  the LAMMPS input template, whatever `template_path` you pass the sampler.
