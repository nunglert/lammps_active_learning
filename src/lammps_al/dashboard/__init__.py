"""Analysis/plotting pipeline for ACE-committee active-learning campaigns.

Mirrors ``jaxrens_al.dashboard``'s split: :mod:`run_layout` is the on-disk
bridge to a ``remote_handler`` campaign directory, :mod:`context` resolves
one page's paths through it (the only caller of :mod:`run_layout` besides
:mod:`cli`), :mod:`committee_eval` wraps this repo's own committee-scoring
calculator, :mod:`panels` holds the ``alviz.Entry`` panel classes, each
registered with an ``alviz.PanelSpec`` + builder, :mod:`panel_specs`
composes those into the default panel list (overridable from a
``dashboard.yaml``), and :mod:`cli` renders it into a multipage dashboard
PDF. A per-campaign entry point is just a thin wrapper
calling :func:`cli.main` with that campaign's ``calchome`` and species list
(campaign working directories, e.g. ``experiments/``, are gitignored --
campaign-specific, not shipped with this repo).

Needs the ``[dashboard]`` extra (``alviz``, ``matscipy``, ``pyyaml``); the
committee/species panels additionally need ``[ace]`` (``pyace``) in the same
environment.
"""
