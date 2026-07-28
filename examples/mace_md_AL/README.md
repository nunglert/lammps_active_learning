# mace_md_AL — LAMMPS MD active-learning example

A Cu/Co/H/O surface active-learning loop that samples with LAMMPS MD driven by a
MACE committee, labels with VASP DFT, and retrains the committee:

```
LammpsSampler (lammps_al)  ->  VASPEvaluator (remote_handler)  ->  MaceModelTrainer (remote_handler)
        ^                                                                    |
        +--------------------------- next iteration --------------------------+
```

The sampler is this package's `lammps_al.run.sampler.LammpsSampler`; the
evaluator, trainer, and orchestration loop come from `remote_handler`. The
remote payloads (the LAMMPS sampling script, `dft_singlepoint.py`,
`mace_training.py`) are the ones **shipped by the packages** — this example
carries no copies of them.

## Layout

```
run_al.py                         the loop wiring (edit the placeholders)
config/templates/
  lammps.in.template              LAMMPS input (pair_style mliap unified <mace>)
  mace_config_l2.yaml.template    MACE training config (jinja)
  datalab.sh.j2 / dirona.sh.j2    SLURM header templates (per cluster)
  body_single.sh.j2 / body_array.sh.j2   job-body templates
inputs/
  cu+cu_co_db_example.xyz         seed DFT database
  test_fcc100_hollow_2x2.data     MD start structure
  models/8[1-5]_..._stagetwo.model   the seed 5-model committee
```

## Running it

1. Install the run side: `pip install ".[run]"` (or editable sibling checkouts
   of `remote_handler` + this package).
2. Point `REMOTE_HANDLER_SLURM_TEMPLATES` at `config/templates` so the SLURM
   header + body templates resolve:
   ```bash
   export REMOTE_HANDLER_SLURM_TEMPLATES=$(pwd)/config/templates
   ```
3. Edit `run_al.py`: the SSH aliases (`remote_alias=`), `ClusterConfig` fields
   (partition, conda env/source, savepath, header template), the remote LAMMPS
   executable path, and the `remote_mace_train_runner` path are **placeholders
   from a prior user** — set them to your clusters. Remote hosts resolve through
   your `~/.ssh/config`.
4. `python run_al.py`

Cluster configs are built inline in `run_al.py` (the current style; the old
`config/servers/*.json` files are gone). The loop seeds from the committee in
`inputs/models/` + the DFT database and starts at the sampling step.

> The MD run defaults to KOKKOS-on-GPU. For a non-GPU LAMMPS build, pass
> `lammps_args=...` / `device="cpu"` to `LammpsSampler` (see its docstring).
