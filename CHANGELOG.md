# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are cut from git tags via `setuptools-scm` (see the *Install* section
of `README.md`): tag a release with `git tag vX.Y.Z` and the package version
follows automatically. To release, rename the `[Unreleased]` section below to
the new `## [X.Y.Z] — YYYY-MM-DD`, add a fresh empty `[Unreleased]`, then tag.

## [Unreleased]

### Fixed
- **Gave the LAMMPS payload a CPU-testable coupling route, alongside
  (not instead of) the production one.** The production route
  (`pair_style mliap unified`, ML-IAP) needs LAMMPS's ghost-atom
  message-passing hooks (`forward_exchange`/`reverse_exchange`) for any MACE
  model with more than one interaction layer -- i.e. essentially every real
  model -- and those are Kokkos-CUDA-only: confirmed by building both the
  ACEsuit LAMMPS fork and vanilla upstream `develop` and hitting
  `AttributeError: ... no attribute 'forward_exchange'` on a CPU
  (Kokkos-OpenMP or no-Kokkos) build either way, so the `lammps` CI tier and
  any CPU-only local run could never actually pass. New
  `LammpsSamplingInput.lammps_pair_style` (default `"mliap"`, i.e. **no
  change to production behaviour**) selects the compile route:
  `"mliap"` keeps `create_lammps_mliap_model`/`enable_cueq` exactly as
  before; `"mace"` is new and opt-in, targeting LAMMPS's native
  `pair_style mace` (`ML-MACE` package, LibTorch directly,
  `create_lammps_mace_model`, TorchScript-compiled via
  `mace.cli.create_lammps_model --format libtorch`) -- CPU-only, no Kokkos
  needed. Only the `lammps` CI/local test tier (and
  `docker/Dockerfile.lammps-mace`, rebuilt to a CPU-only `ML-MACE` image) use
  `"mace"`; `LammpsSampler`'s and `LammpsSamplingInput`'s defaults
  (`lammps_args`, `device`, `lammps_pair_style`) are unchanged
  KOKKOS-on-GPU + `"mliap"`. LAMMPS templates
  (`pair_style mliap unified ${model_file} 0` / `pair_coeff * * <species>`)
  now render either route via `{% if lammps_pair_style == "mace" %}`.

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
  and a `lammps-smoke` CI job provide the automated LAMMPS install for that
  tier (see the *Fixed* entry above for how that Dockerfile actually builds
  LAMMPS now).
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
