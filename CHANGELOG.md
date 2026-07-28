# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are cut from git tags via `setuptools-scm` (see the *Install* section
of `README.md`): tag a release with `git tag vX.Y.Z` and the package version
follows automatically. To release, rename the `[Unreleased]` section below to
the new `## [X.Y.Z] — YYYY-MM-DD`, add a fresh empty `[Unreleased]`, then tag.

## [Unreleased]

### Added
- **Tiered testing for the LAMMPS payload, incl. a CI path to a real `lmp`.**
  The cluster payload was made import-safe (`main` behind `__main__`; `torch` /
  `mace` / `LammpsSamplingInput` imported lazily) and its accelerator flags +
  MACE device made configurable (`LammpsSamplingInput.lammps_args` / `device`,
  defaulting to the previous KOKKOS-on-GPU behaviour). On top of that: `local`
  unit tests of the pure sampling logic (jinja/aggregation/selection, no
  MACE/LAMMPS); an `integration` test driving `get_samples_from_dump`
  end-to-end on CPU with a real MACE committee over a synthesized LAMMPS dump
  (no `lmp`, no GPU); and a self-skipping `lammps` smoke test that runs the full
  `main()` against a real MACE-enabled `lmp`. A `docker/Dockerfile.lammps-mace`
  (LAMMPS with ML-IAP + MLIAP-Python + KOKKOS/OpenMP) and a `lammps-smoke` CI
  job provide the automated LAMMPS install for that tier.
- **Initial extraction from `remote_handler`.** `lammps_active_learning` is the
  composition root for LAMMPS active-learning runs. It owns the concrete
  `LammpsSampler` (a native `remote_handler.Sampler` for LAMMPS MD with
  committee-uncertainty sampling), its `LammpsSamplingInput` schema, and the
  cluster-side payload `lammps_sampling.py` — all relocated out of
  `remote_handler`, which keeps only the `Sampler` interface and the
  remote-execution machinery. Install the sampler with the `[run]` extra
  (`pip install ".[run]"` → `remote_handler` + a MACE stack); the host side
  needs only `remote_handler` + `ase` + `numpy`. The `examples/mace_md_AL`
  Cu/Co surface workflow moved over as a runnable reference.

[Unreleased]: https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/lammps_active_learning/-/commits/main
