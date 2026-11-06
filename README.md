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

- `lammps_al.run.sampler` — `LammpsSampler`, a native `remote_handler.Sampler`,
  generic over the committee's potential family via `backend` (`"mace"` default,
  `"ace"`, or `"grace"`). Per AL iteration it compiles the committee's model for
  LAMMPS (`backend="mace"` only -- ACE's/GRACE's native pair styles need no compile
  step), renders the user's `lammps.in` jinja2 template, runs a LAMMPS MD trajectory
  on the cluster via a `remote_handler` `SingleScriptTask`, scores every frame's
  local force-uncertainty against the committee (via the backend's ensemble ASE
  calculator), and returns the most-uncertain frame from each of `num_samples`
  trajectory segments, each tagged with a per-atom `uncertainty` array. Needs
  the `[run]` extra. Supports `stub=True` for cluster-free dry runs.
  For `backend="mace"`, the compile step targets one of two LAMMPS/MACE couplings,
  chosen via `lammps_pair_style` (default `"mliap"`, unchanged production behaviour --
  the ML-IAP `mliap unified` Python bridge, KOKKOS-on-GPU): `"mace"` is an
  opt-in, CPU-testable alternative (LAMMPS's native `pair_style mace`, the
  `ML-MACE` package built against LibTorch directly) used only by the
  `lammps` CI/local test tier, not a production default -- see that
  sampler's module docstring for why. For `backend="ace"`/`"grace"`,
  `lammps_pair_style` is instead the literal pair_style line rendered into the
  template (e.g. `"pace"` or `"grace pad_verbose padding 0.1"`).
- `lammps_al.run.inputs.LammpsSamplingInput` — the JSON contract between the
  host-side sampler and the cluster-side payload (a `remote_handler.ScriptInput`
  subclass).
- `lammps_al.run.scripts.lammps_sampling.py` — the cluster-side payload shipped
  and executed by the sampler; `lammps_al.run.scripts.script_dir()` locates it
  from a non-editable install. It runs in the remote conda env, which needs
  `lammps_al` plus whichever `backend`'s stack: MACE (`mace-torch` + a LAMMPS
  build with ML-IAP + Kokkos-CUDA by default, or the `ML-MACE` package for
  `lammps_pair_style="mace"`), ACE (`pyace`/pacemaker, conda-forge), or GRACE
  (`tensorpotential`/gracemaker) -- not on the host.

- `lammps_al.embedding.sampler` — `AmorphousEmbeddingSampler`, a second
  native `remote_handler.Sampler`. Per AL iteration it extracts regions of
  interest (ROIs) from an uncertainty-tagged trajectory, carves/orients/
  cleans up an amorphous-matrix box around each (optionally oriented to a
  detected surface/interface normal via OVITO), and anneals the mobile shell
  around a frozen ROI core with a multi-stage LAMMPS deck. Two remote stages
  (`lammps_al.embedding.scripts.extract`/`embedding_anneal`), each its own
  `SingleScriptTask`, potentially on different machines/envs. Needs the
  `[run]` extra; `topology_mode` in `{"surface", "interface"}` additionally
  needs the `[embedding]` extra (`ovito`) on the extraction machine. Supports
  `stub=True` for cluster-free dry runs.
- `lammps_al.embedding.inputs` — `MatrixEmbeddingInput`/`EmbeddingAnnealInput`,
  the JSON contracts for the two embedding payloads.
- `lammps_al.embedding.scripts.{extract,embedding_anneal}.py` — the
  cluster-side payloads; `lammps_al.embedding.scripts.script_dir()` locates
  them from a non-editable install. `cleanup.py`/`geometry.py`/
  `surface_mesh.py` hold the importable ROI-clustering/box-carving/
  normal-detection helpers `extract.py` builds on.

## Uncertainty-driven sampling

The sampler is a MACE **committee** driver: it runs one MD trajectory, evaluates
per-atom force disagreement across the committee, aggregates it locally within
`uncertainty_cutoff_radius`, splits the trajectory into `num_samples` segments,
and emits the peak-uncertainty frame of each segment as a labelling candidate.
Reducing that candidate pool further (diversity/novelty filtering) is a
`remote_handler` *selector* concern, not the sampler's.

## Install (local dev)

`remote_handler` is pinned by a public HTTPS git URL on GitHub, so it
installs directly with no SSH key or token. To work against unpushed or local
`remote_handler` changes (or offline), install a sibling checkout editable,
then this project with `--no-deps`:

```bash
# Either (from GitHub):
pip install -e ".[run]"

# Or (offline / sibling checkout):
pip install -e ../remote_handler        # provides the [run] dependency
pip install -e ".[dev]" --no-deps
```

The host side needs only `remote_handler` + `ase` + `numpy`. The **cluster**
conda env that runs the payload additionally needs `lammps_al` (for the input
schema), `mace-torch`, `torch`, and (production default) a LAMMPS build with
ML-IAP + Kokkos-CUDA.

Versions come from git tags via `setuptools-scm`: tag `vX.Y.Z` and
`lammps_al.__version__` follows (refreshes on reinstall, not on a bare pull).

## Tests

```bash
pytest                    # all tiers (each self-skips when its dep is absent)
pytest -m local           # pure payload logic (ase+numpy+jinja2 only)
pytest -m integration     # sampler/input + MACE-committee scoring (needs remote_handler + mace)
pytest -m lammps          # real lmp smoke test (needs a MACE-enabled lmp on PATH)
```

Three tiers cover the sampling machinery at rising cost:

- **`local`** — the payload's pure helpers (jinja render, local-uncertainty
  aggregation, segment/argmax selection) with no producer, MACE, or LAMMPS.
- **`integration`** — the sampler + input schema, and `get_samples_from_dump`
  driven end-to-end on CPU with a real MACE committee over a synthesized LAMMPS
  dump (no `lmp`, no GPU). Self-skips when `remote_handler`/`mace` are absent.
- **`lammps`** — a full end-to-end smoke run against a real MACE-enabled `lmp`
  (compile model → short MD → committee-score → select). Self-skips unless
  such an `lmp` is on PATH; in CI it runs on the prebuilt `lammps-mace` image
  (`docker/Dockerfile.lammps-mace`) -- the same image reproduces this tier
  locally (`docker run`, see that Dockerfile's header). It runs with
  `lammps_pair_style="mace"` (LAMMPS's native `pair_style mace`, CPU-only,
  no Kokkos needed), **not** the production `"mliap"` default -- the ML-IAP
  route's ghost-atom message passing is Kokkos-CUDA-only, so it cannot run
  in this CPU tier. See `lammps_sampling.py`'s module docstring for the full
  story; production is unaffected by any of this.

The payload is import-safe and its accelerator flags / MACE device / pair
style are all configurable (`LammpsSamplingInput.lammps_args` / `device` /
`lammps_pair_style`, defaulting to the production KOKKOS-on-GPU + `"mliap"`
behaviour) — which is what lets the `lammps` tier opt into a CPU-runnable
configuration for CI/local testing without changing what a real deployment
gets by default.

## Run an AL loop

`examples/mace_md_AL/` is a runnable Cu/Co surface reference (its own cluster
`config/`, committee `inputs/models/`, and `run_al.py`). It bundles hard-coded
paths from a prior user — adapt them before running. Start with `stub=True`.

`examples/amorphous_embedding_AL/` has two runnable `AmorphousEmbeddingSampler`
references driving a single sampling step directly against the sampler (not a
full AL loop): `ame1` (Cu/H2O interface, `topology_mode="interface"`,
two-stage anneal) and `ame2` (bulk Si/SiO2, `topology_mode="bulk"`,
single-stage anneal), both driven by a GRACE foundation-model potential. Each
bundles its own `inputs.zip`/`results.zip` and hard-coded paths from a prior
user (conda/LAMMPS/model paths) — adapt `run_remote.py` before running.

## Conventions

- Formatting is enforced by pre-commit: **black** and **isort** both at
  `--line-length 79` (isort `--profile black`).
- `remote_handler` labels ground-truth energies/forces with `REF_energy` /
  `REF_forces`; the sampler tags candidates with `max_uncertainty` /
  `mean_uncertainty`.
- The run side needs `REMOTE_HANDLER_CONFIG_DIR` (SSH/cluster config) and, for
  the LAMMPS input template, whatever `template_path` you pass the sampler.
