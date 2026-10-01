"""Composition of the dashboard's panel set (registry glue only).

Each panel registers its own :class:`alviz.PanelSpec` + builder in its own
module (``lammps_al.dashboard.panels.*``), so panel-specific knowledge lives
with the panel; this module composes them: the curated default list, the
config -> spec plumbing, and :func:`assemble`. Mirrors
``jaxrens_al.dashboard.panel_specs``, minus its multiple dashboard flavors.

One default list serves both page kinds -- every builder self-gates on its
view (per-iteration vs. campaign summary) and on its inputs being present in
the :class:`~lammps_al.dashboard.context.RunContext`, so a config's
``add``/``disable``/``panels`` likewise applies to every page.
"""

from __future__ import annotations

from typing import Optional

from alviz import PanelSpec

# Importing the panels package runs every panel module's ``register_panel``
# call, populating the alviz panel registry this module composes over.
import lammps_al.dashboard.panels  # noqa: F401

__all__ = [
    "make_spec",
    "default_specs",
    "parse_panel_config",
    "assemble",
]


def make_spec(type_name: str, **params) -> PanelSpec:
    """Construct a registered panel spec by ``type`` name (+ its params)."""
    from alviz.spec import PANEL_REGISTRY

    return PANEL_REGISTRY[type_name].spec_model(**params)


def default_specs() -> list:
    """The curated default panels, in render order."""
    return [
        make_spec("committee_parity"),
        make_spec("species_force_parity"),
        make_spec("training_convergence"),
        make_spec("roi_spatial"),
        make_spec("box_composition"),
        make_spec("box_rdf"),
        # Explicitly named so `disable: [box_rdf_post_anneal]` can drop just
        # this instance (a bare `box_rdf` drops both).
        make_spec("box_rdf", source="post_anneal", name="box_rdf_post_anneal"),
        make_spec("sampler_thermo"),
        make_spec("annealer_thermo"),
        make_spec("held_out_parity"),
        make_spec("uncertainty_trend"),
        make_spec("uncertainty_depth_profile"),
    ]


def parse_panel_config(raw) -> Optional[dict]:
    """Turn a config's raw ``add``/``disable``/``panels`` mapping into an
    :func:`assemble` ``panel_config``.

    The ``add``/``panels`` lists are validated into :class:`PanelSpec`s
    against the registry (an unknown ``type`` or field raises). Returns
    ``None`` when there is nothing to layer.
    """
    if not raw:
        return None
    from alviz import parse_specs

    out = {}
    if raw.get("disable"):
        out["disable"] = list(raw["disable"])
    if raw.get("add"):
        out["add"] = parse_specs(raw["add"])
    if raw.get("panels"):
        out["panels"] = parse_specs(raw["panels"])
    return out or None


def assemble(ctx, *, datapath, panel_config=None) -> list:
    """The entries of one page: defaults layered with ``panel_config``
    (see ``alviz.resolve_panel_list``), run-global settings filled into
    unset spec fields, then built against ``ctx`` (absent-input /
    wrong-view panels drop out)."""
    from alviz import apply_globals, build_entries, resolve_panel_list

    cfg = panel_config or {}
    specs = resolve_panel_list(
        default_specs(),
        panels=cfg.get("panels"),
        add=cfg.get("add"),
        disable=cfg.get("disable"),
    )
    specs = [apply_globals(s, {"species": ctx.species}) for s in specs]
    return build_entries(specs, ctx, datapath=datapath, force_recompute=False)
