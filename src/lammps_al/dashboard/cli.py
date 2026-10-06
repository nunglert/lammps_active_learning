"""CLI: build a multipage dashboard PDF for one AL campaign's ``calchome``.

One page per iteration with data plus one campaign-summary page. Which
panels appear is spec-driven (:mod:`panel_specs`): a curated default list,
layered with an optional YAML config -- ``--config``, else
``<calchome>/dashboard.yaml`` if present -- of the form::

    species: [Cu, H, O]          # any CLI flag, as a default (flags win)
    ncols: 3                     # page grid: panels per row ...
    width_per_panel: 5.0         # ... and inches per grid cell
    disable: [sampler_thermo]    # drop defaults by type or name
    add:                         # extra panels, rendered last
      - {type: box_rdf, source: post_anneal, name: rdf_long,
         pairs: [[O, O, 12.0]]}
    # panels: [...]              # fully replaces the defaults instead

Mirrors ``jaxrens_al``'s ``jaxrens-al dashboard`` config layer, minus its
multiple dashboard flavors. A per-campaign wrapper is just a thin script
calling :func:`main` with that campaign's ``calchome`` (campaign working
directories are gitignored, not shipped with this repo).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

from alviz.dashboard import Dashboard
from alviz.mpl_utils import load_rc_params
from alviz.pdf import merge_pdfs

from lammps_al.dashboard import run_layout
from lammps_al.dashboard.context import campaign_context, iteration_context
from lammps_al.dashboard.panel_specs import assemble, parse_panel_config

logger = logging.getLogger(__name__)

_CONFIG_NAMES = ("dashboard.yaml", "dashboard.yml")
_PANEL_KEYS = ("add", "disable", "panels")


def _any_entry_has_data(board: Dashboard, *, page_label: str) -> bool:
    """Whether at least one of ``board``'s entries actually computed data.

    ``Dashboard.render`` fails individual entries silently (logs + excludes
    them), so a page can otherwise be "successfully" rendered and saved
    while every one of its panels failed to compute (e.g. no data exists
    yet for that iteration) -- an essentially blank PDF page. Checking this
    lets callers skip saving such a page entirely. Also logs a per-entry
    ok/failed summary for ``page_label`` -- ``Dashboard.render``'s own
    per-entry failure logging already fires (with the exception), but
    nowhere otherwise summarizes which entries on a page actually rendered.
    """
    ok = [e.name for e in board.entries if e.data is not None]
    failed = [e.name for e in board.entries if e.data is None]
    logger.info(
        "%s: %d/%d panel(s) rendered (ok=%s, failed=%s).",
        page_label,
        len(ok),
        len(board.entries),
        ok,
        failed,
    )
    return bool(ok)


def _summary_cache_key(ctx) -> str:
    """Short hash of the artifacts the campaign-summary page resolved.

    alviz reuses an entry's ``<name>.json`` whenever it exists, but the
    summary panels aggregate over a set of iterations that keeps growing
    while the campaign runs. Keying the summary page's data directory on
    what was found on disk (iterations, their trajectories and per-stage
    status) makes every change to that set land in a fresh directory, so
    stale JSON is never reused. Contents changing in place under an
    unchanged path/status are not detected.
    """
    state = [
        sorted(ctx.uncertainty_trajectories.items()),
        [
            (
                c.iteration,
                c.selector_manifest,
                c.annealer_boxes,
                c.dft_calcs,
                c.config_files,
            )
            for c in ctx.iteration_contexts
        ],
    ]
    return hashlib.sha1(repr(state).encode()).hexdigest()[:12]


def _summary_datapath(out_dir: Path, ctx) -> Path:
    """The summary page's keyed data dir; prunes stale sibling keys."""
    root = out_dir / "campaign"
    datapath = root / _summary_cache_key(ctx)
    if root.is_dir():
        for stale in root.iterdir():
            if stale.is_dir() and stale != datapath:
                shutil.rmtree(stale)
    return datapath


def build_dashboard_pdfs(
    calchome,
    out_dir,
    species,
    *,
    only=None,
    exclude=None,
    panel_config=None,
    ncols: int = 2,
    width_per_panel: float = 4.0,
    height_per_panel: float = 4.0,
) -> list:
    """Render one page per iteration with data + one campaign-summary page.

    Returns the ordered list of per-page PDF paths (feed to
    ``alviz.pdf.merge_pdfs``). A page with no renderable entries is skipped
    entirely rather than emitted empty. ``panel_config`` is a parsed
    ``add``/``disable``/``panels`` layer (see
    :func:`panel_specs.parse_panel_config`). ``ncols`` /
    ``width_per_panel`` / ``height_per_panel`` size every page's grid (a
    panel's own ``colspan`` is clamped to ``ncols``).
    """
    calchome = Path(calchome)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = run_layout.load_progress(calchome)

    n_iterations = len(
        run_layout.trained_models_per_iteration(calchome, progress)
    )
    logger.info(
        "Building dashboard for %s: %d iteration(s) with a trained model.",
        calchome,
        n_iterations,
    )
    contexts = [
        iteration_context(calchome, progress, it, species)
        for it in range(n_iterations)
    ]
    contexts.append(
        campaign_context(
            calchome,
            progress,
            range(n_iterations),
            species,
            iteration_contexts=contexts,
        )
    )

    pages = []
    for ctx in contexts:
        page_datapath = (
            _summary_datapath(out_dir, ctx)
            if ctx.is_summary
            else out_dir / f"iter_{ctx.iteration}"
        )
        page_datapath.mkdir(parents=True, exist_ok=True)
        entries = assemble(
            ctx, datapath=page_datapath, panel_config=panel_config
        )
        if not entries:
            logger.info(
                "%s: no panels built at all -- skipping page.",
                ctx.page_label,
            )
            continue
        board = Dashboard.from_entries(
            entries,
            ncols=ncols,
            width_per_panel=width_per_panel,
            height_per_panel=height_per_panel,
            only=only,
            exclude=exclude,
        )
        board.render(suptitle=ctx.page_label)
        if not _any_entry_has_data(board, page_label=ctx.page_label):
            logger.info(
                "%s: every panel failed to compute -- skipping page (see "
                "per-panel errors above).",
                ctx.page_label,
            )
            continue
        page_path = page_datapath / "page.pdf"
        board.savefig(page_path)
        pages.append(page_path)

    logger.info("Built %d dashboard page(s).", len(pages))
    return pages


def load_config(path) -> dict:
    """Read a YAML dashboard config (empty file -> ``{}``)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got "
            f"{type(data).__name__}."
        )
    return data


def _apply_config(parser, argv, args):
    """Fold a config file in as flag defaults; returns ``(args,
    panel_config_raw, config_path)`` (the latter two ``None`` without a
    config).

    The config's flat keys become argparse defaults (so any CLI flag still
    wins); its ``add``/``disable``/``panels`` keys are split off for
    :func:`panel_specs.parse_panel_config`. An unknown flat key raises
    rather than being silently ignored.
    """
    if args.config:
        path = Path(args.config)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        path = next(
            (
                Path(args.calchome) / n
                for n in _CONFIG_NAMES
                if (Path(args.calchome) / n).is_file()
            ),
            None,
        )
    if path is None:
        return args, None, None

    cfg = {str(k).replace("-", "_"): v for k, v in load_config(path).items()}
    panel_raw = {k: cfg.pop(k, None) for k in _PANEL_KEYS}
    valid = {a.dest for a in parser._actions} - {"help", "calchome", "config"}
    unknown = sorted(set(cfg) - valid)
    if unknown:
        raise ValueError(
            f"{path}: unknown config key(s) {unknown}; valid: "
            f"{sorted(valid | set(_PANEL_KEYS))}."
        )
    parser.set_defaults(**cfg)
    return parser.parse_args(argv), panel_raw, path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="lammps-al dashboard",
        description=(
            "Build a parity/uncertainty dashboard PDF for an ACE-committee "
            "active-learning campaign."
        ),
    )
    parser.add_argument("calchome", help="Path to the campaign's calchome dir")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "YAML dashboard config (flag defaults + panel add/disable/"
            "panels). Default: <calchome>/dashboard.yaml if present."
        ),
    )
    parser.add_argument(
        "--output", default="dashboard.pdf", help="Output PDF path"
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["Cu", "H", "O"],
        help="LAMMPS type order (for species-resolved panels)",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, help="Only render these panels"
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None, help="Skip these panels"
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Grid columns per page (wide panels span 2 of them)",
    )
    parser.add_argument(
        "--width-per-panel",
        type=float,
        default=4.0,
        help="Width of one grid cell, in inches",
    )
    parser.add_argument(
        "--height-per-panel",
        type=float,
        default=4.0,
        help="Height of one grid cell, in inches",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Debug-level logging (per-panel skip reasons, calcdir lookups, "
            "..) -- INFO level is the default without this flag."
        ),
    )
    args = parser.parse_args(argv)
    # Before logging setup, so a config's `verbose: true` takes effect.
    args, panel_raw, config_path = _apply_config(parser, argv, args)

    # No caller configures logging for this CLI otherwise -- without a
    # handler, Python's root logger only surfaces WARNING+ (its "handler of
    # last resort"), silently swallowing every INFO/DEBUG log this module
    # and run_layout/committee_eval emit (e.g. exactly why a panel/page was
    # skipped), which is the whole point of having them.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if config_path is not None:
        logger.info("Applied dashboard config %s.", config_path)
    panel_config = parse_panel_config(panel_raw)

    load_rc_params()
    # Resolution of rasterized artists (the parity scatter points) inside
    # the otherwise-vector PDF; matplotlib's default is the figure's 100.
    import matplotlib as mpl

    mpl.rcParams["savefig.dpi"] = 200
    calchome = Path(args.calchome)
    out_dir = calchome / "dashboard_data"
    pages = build_dashboard_pdfs(
        calchome,
        out_dir,
        args.species,
        only=args.only,
        exclude=args.exclude,
        panel_config=panel_config,
        ncols=args.ncols,
        width_per_panel=args.width_per_panel,
        height_per_panel=args.height_per_panel,
    )
    if not pages:
        logger.warning("No dashboard pages produced (no finished iterations?)")
        return 1

    output = merge_pdfs(pages, args.output)
    logger.info("Wrote %s (%d pages)", output, len(pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
