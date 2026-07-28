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
