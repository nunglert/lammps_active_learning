"""Analysis/plotting pipeline for ACE-committee active-learning campaigns.

Mirrors ``jaxrens_al.dashboard``'s split: :mod:`run_layout` is the on-disk
bridge to a ``remote_handler`` campaign directory, :mod:`committee_eval`
wraps this repo's own committee-scoring calculator, :mod:`panels` holds the
``alviz.Entry`` panel classes, and :mod:`cli` assembles them into a
multipage dashboard PDF. See ``experiments/rough_cu_water_al/dashboard.py``
for the per-campaign entry point.

Needs the ``[dashboard]`` extra (``alviz``); the committee/species panels
additionally need ``[ace]`` (``pyace``) in the same environment.
"""
