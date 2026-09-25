"""CLI: build a multipage dashboard PDF for one AL campaign's ``calchome``.

Deliberately no panel registry or config-file layer (unlike jaxrens_al's
``jaxrens-al`` CLI, which spans many trainer/sampler backends across a
maintained multi-user tool) -- this is one campaign, one committee family,
single user. A per-campaign wrapper is just a thin script calling
:func:`main` with that campaign's ``calchome`` and species list (campaign
working directories are gitignored, not shipped with this repo).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alviz.dashboard import Dashboard
from alviz.mpl_utils import load_rc_params
from alviz.pdf import merge_pdfs

from lammps_al.dashboard import run_layout
from lammps_al.dashboard.panels.annealer_thermo import AnnealerThermoPanel
from lammps_al.dashboard.panels.box_structure import (
    BoxCompositionPanel,
    BoxRDFPanel,
)
from lammps_al.dashboard.panels.committee_parity import (
    CommitteeParityPanel,
    HeldOutParityPanel,
)
from lammps_al.dashboard.panels.roi_spatial import ROISpatialPanel
from lammps_al.dashboard.panels.sampler_thermo import SamplerThermoPanel
from lammps_al.dashboard.panels.species_parity import SpeciesForceParityPanel
from lammps_al.dashboard.panels.training_fit import TrainingConvergencePanel
from lammps_al.dashboard.panels.uncertainty import (
    UncertaintyDepthProfilePanel,
    UncertaintyTrendPanel,
)

logger = logging.getLogger(__name__)


def build_entries_for_iteration(
    calchome, progress, it: int, species, *, datapath
) -> list:
    """The per-iteration panels: committee / held-out / species parity.

    Returns an empty list when this iteration has no trained model or no
    evaluated configs yet (nothing to plot).
    """
    trained = run_layout.trained_models_per_iteration(calchome, progress)
    evaluated = run_layout.evaluated_configs_per_iteration(calchome, progress)

    model_files = trained[it] if it < len(trained) else []
    config_files = evaluated[it] if it < len(evaluated) else []

    entries = []
    if model_files and config_files:
        entries.append(
            CommitteeParityPanel(config_files, model_files, datapath=datapath)
        )
        entries.append(
            SpeciesForceParityPanel(
                config_files, model_files, species, datapath=datapath
            )
        )
    else:
        logger.info(
            "iteration %d: skipping committee/species parity panels "
            "(model_files=%d, config_files=%d).",
            it,
            len(model_files),
            len(config_files),
        )

    if run_layout.trainer_calcdirs_for_iteration(calchome, it):
        entries.append(
            TrainingConvergencePanel(calchome, progress, it, datapath=datapath)
        )
    else:
        logger.info(
            "iteration %d: skipping training-convergence panel (no trainer "
            "calcdirs found).",
            it,
        )

    if run_layout.selector_output_manifest(calchome, progress, it):
        entries.append(
            ROISpatialPanel(calchome, progress, it, datapath=datapath)
        )
        entries.append(
            BoxCompositionPanel(calchome, progress, it, datapath=datapath)
        )
        entries.append(BoxRDFPanel(calchome, progress, it, datapath=datapath))
    else:
        logger.info(
            "iteration %d: skipping ROI spatial/box composition/box RDF "
            "panels (no selector_output recorded).",
            it,
        )

    if run_layout.annealed_configs(calchome, progress, it):
        # Distinct `name=` from the pre-anneal instance above -- Entry.name
        # doubles as its JSON cache filename, so two panels sharing one
        # would silently clobber each other's cached data.
        entries.append(
            BoxRDFPanel(
                calchome,
                progress,
                it,
                source="post_anneal",
                name="box_rdf_post_anneal",
                datapath=datapath,
            )
        )
    else:
        logger.info(
            "iteration %d: skipping post-anneal box RDF panel (no "
            "annealer_output recorded).",
            it,
        )

    if run_layout.sampler_lmp_log(calchome, it):
        entries.append(SamplerThermoPanel(calchome, it, datapath=datapath))
    else:
        logger.info(
            "iteration %d: skipping sampler thermo panel (no sampler "
            "LAMMPS log found).",
            it,
        )

    if run_layout.annealer_lmp_logs_for_iteration(calchome, it):
        entries.append(AnnealerThermoPanel(calchome, it, datapath=datapath))
    else:
        logger.info(
            "iteration %d: skipping annealer thermo panel (no annealer "
            "LAMMPS logs found).",
            it,
        )

    prev_models = trained[it - 1] if 0 <= it - 1 < len(trained) else []
    if prev_models:
        entries.append(
            HeldOutParityPanel(
                calchome, progress, it, prev_models, datapath=datapath
            )
        )
    else:
        logger.info(
            "iteration %d: skipping held-out parity panel (no trained "
            "model recorded for iteration %d).",
            it,
            it - 1,
        )
    return entries


def build_campaign_entries(calchome, progress, *, datapath) -> list:
    """The whole-campaign panels: uncertainty trend + depth profile."""
    n_iterations = len(
        run_layout.trained_models_per_iteration(calchome, progress)
    )
    iterations = list(range(n_iterations))
    return [
        UncertaintyTrendPanel(
            calchome, progress, iterations, datapath=datapath
        ),
        UncertaintyDepthProfilePanel(
            calchome, progress, iterations, datapath=datapath
        ),
    ]


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


def build_dashboard_pdfs(
    calchome, out_dir, species, *, only=None, exclude=None
) -> list:
    """Render one page per iteration with data + one campaign-summary page.

    Returns the ordered list of per-page PDF paths (feed to
    ``alviz.pdf.merge_pdfs``). A page with no renderable entries is skipped
    entirely rather than emitted empty.
    """
    calchome = Path(calchome)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = run_layout.load_progress(calchome)

    pages = []
    n_iterations = len(
        run_layout.trained_models_per_iteration(calchome, progress)
    )
    logger.info(
        "Building dashboard for %s: %d iteration(s) with a trained model.",
        calchome,
        n_iterations,
    )
    for it in range(n_iterations):
        page_datapath = out_dir / f"iter_{it}"
        page_datapath.mkdir(parents=True, exist_ok=True)
        entries = build_entries_for_iteration(
            calchome, progress, it, species, datapath=page_datapath
        )
        if not entries:
            logger.info(
                "iteration %d: no panels built at all -- skipping page.", it
            )
            continue
        board = Dashboard.from_entries(entries, only=only, exclude=exclude)
        board.render(suptitle=f"iteration {it}")
        if not _any_entry_has_data(board, page_label=f"iteration {it}"):
            logger.info(
                "iteration %d: every panel failed to compute -- skipping "
                "page (see per-panel errors above).",
                it,
            )
            continue
        page_path = page_datapath / "page.pdf"
        board.savefig(page_path)
        pages.append(page_path)

    campaign_datapath = out_dir / "campaign"
    campaign_datapath.mkdir(parents=True, exist_ok=True)
    campaign_entries = build_campaign_entries(
        calchome, progress, datapath=campaign_datapath
    )
    if campaign_entries:
        board = Dashboard.from_entries(
            campaign_entries, only=only, exclude=exclude
        )
        board.render(suptitle="campaign summary")
        if _any_entry_has_data(board, page_label="campaign summary"):
            page_path = campaign_datapath / "page.pdf"
            board.savefig(page_path)
            pages.append(page_path)

    logger.info("Built %d dashboard page(s).", len(pages))
    return pages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a parity/uncertainty dashboard PDF for an ACE-committee "
            "active-learning campaign."
        )
    )
    parser.add_argument("calchome", help="Path to the campaign's calchome dir")
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
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Debug-level logging (per-panel skip reasons, calcdir lookups, "
            "..) -- INFO level is the default without this flag."
        ),
    )
    args = parser.parse_args(argv)

    # No caller configures logging for this CLI otherwise -- without a
    # handler, Python's root logger only surfaces WARNING+ (its "handler of
    # last resort"), silently swallowing every INFO/DEBUG log this module
    # and run_layout/committee_eval emit (e.g. exactly why a panel/page was
    # skipped), which is the whole point of having them.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    load_rc_params()
    calchome = Path(args.calchome)
    out_dir = calchome / "dashboard_data"
    pages = build_dashboard_pdfs(
        calchome, out_dir, args.species, only=args.only, exclude=args.exclude
    )
    if not pages:
        logger.warning("No dashboard pages produced (no finished iterations?)")
        return 1

    output = merge_pdfs(pages, args.output)
    logger.info("Wrote %s (%d pages)", output, len(pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
