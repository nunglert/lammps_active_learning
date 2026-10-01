"""Committee-vs-reference parity: overall fit and the real "is AL working"
held-out diagnostic.

Both panels share the same underlying computation (committee re-score of a
set of ``REF_energy``/``REF_forces``-labelled configs against a given set of
committee-member potentials); they differ only in *which* configs and *which*
committee are compared -- see each class's docstring.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.panels.training.parity import ParityData, ParityPanel

from lammps_al.dashboard.committee_eval import score_with_committee
from lammps_al.dashboard.context import named

logger = logging.getLogger(__name__)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _read_all(config_files):
    import ase.io

    atoms_list = []
    for path in config_files:
        atoms_list.extend(ase.io.read(path, index=":"))
    return atoms_list


def _frames_match(a_frames, b_frames, tol: float = 1e-8) -> bool:
    """Cheap content check between two equal-length frame lists.

    Compares atom count and ``REF_energy`` of the first and last frame only
    -- enough to distinguish "these are the same underlying frames" from
    "these are unrelated data" without reading every array in every frame.
    """
    if len(a_frames) != len(b_frames):
        return False
    for a, b in zip((a_frames[0], a_frames[-1]), (b_frames[0], b_frames[-1])):
        if len(a) != len(b):
            return False
        ea = a.info.get("REF_energy")
        eb = b.info.get("REF_energy")
        if ea is None or eb is None or abs(ea - eb) > tol:
            return False
    return True


def _new_frames(current, previous) -> list:
    """The frames of ``current`` not already in ``previous``.

    Whether an iteration's evaluator output is a fresh delta or
    cumulative-to-date is not a settled property of remote_handler's
    evaluator -- so this auto-detects: if ``current`` is a strict superset
    of ``previous`` (by count, with matching first/last-frame content), only
    the new tail is returned; otherwise all of ``current`` is treated as
    new. Logs which case fired.
    """
    if (
        len(current) > len(previous)
        and previous
        and _frames_match(current[: len(previous)], previous)
    ):
        logger.info(
            "Evaluator output looks cumulative (contains the previous "
            "iteration's %d frames as a prefix); using only the new tail "
            "(%d frames).",
            len(previous),
            len(current) - len(previous),
        )
        return current[len(previous) :]
    if previous:
        logger.info(
            "Evaluator output looks like a delta (does not extend the "
            "previous iteration's as a prefix); using all %d frames.",
            len(current),
        )
    return current


def rasterized_plot_kwargs(plot_kwargs: dict = None) -> dict:
    """``ParityPanel`` ``plot_kwargs`` with the scatter points rasterized.

    Force parity runs to 1e5-1e6 points, which as vector markers bloat the
    PDF and make it crawl in a viewer; axes/labels stay vector. An explicit
    ``rasterized`` in ``plot_kwargs`` wins.
    """
    plot_kwargs = plot_kwargs or {}
    return {
        axis: {"rasterized": True, **plot_kwargs.get(axis, {})}
        for axis in ("energy", "forces")
    }


def _parity_data_from_configs(
    atoms_list, model_files, *, per_atom: bool = True, show_forces: bool = True
) -> ParityData:
    if not atoms_list:
        raise ValueError("No configurations to score.")

    e_obs = []
    f_obs = []
    for atoms in atoms_list:
        norm = len(atoms) if per_atom else 1
        e_obs.append(float(atoms.info["REF_energy"]) / norm)
        if show_forces:
            f_obs.extend(np.asarray(atoms.arrays["REF_forces"]).ravel())

    e_pred_totals, f_pred = score_with_committee(
        model_files, atoms_list, want_forces=show_forces
    )
    e_pred = [
        e / (len(atoms) if per_atom else 1)
        for e, atoms in zip(e_pred_totals, atoms_list)
    ]

    fields = dict(
        e_obs=e_obs,
        e_pred=e_pred,
        e_rmse=_rmse(e_obs, e_pred),
        e_mae=_mae(e_obs, e_pred),
        per_atom=per_atom,
    )
    if show_forces:
        f_pred_flat = (
            np.concatenate([f.ravel() for f in f_pred])
            if f_pred
            else np.asarray([])
        )
        fields.update(
            f_obs=f_obs,
            f_pred=f_pred_flat.tolist(),
            f_rmse=_rmse(f_obs, f_pred_flat),
            f_mae=_mae(f_obs, f_pred_flat),
        )
    return ParityData(**fields)


class CommitteeParityPanel(ParityPanel):
    """Overall fit quality: committee vs. all evaluated reference configs.

    Re-scores every config in ``config_files`` with the mean prediction of
    ``model_files`` (an ACE committee). Config files/model files are read
    lazily inside ``compute_data`` (not at construction), so a cache hit
    never re-reads them.

    Args:
        config_files: DFT-labelled extxyz paths (``REF_energy``/
            ``REF_forces``), e.g. this iteration's evaluated configs.
        model_files: Committee-member potential ``.yaml`` paths to score
            with.
        per_atom: Normalize energies per atom (matches ``ExtxyzParityPanel``'s
            convention).
        show_forces: Include the forces parity subplot.
    """

    name = "committee_parity"

    def __init__(
        self,
        config_files,
        model_files,
        *args,
        per_atom: bool = True,
        show_forces: bool = True,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            show_forces=show_forces,
            plot_kwargs=rasterized_plot_kwargs(plot_kwargs),
            **kwargs,
        )
        self._config_files = [Path(p) for p in config_files]
        self._model_files = [Path(p) for p in model_files]
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> ParityData:
        atoms_list = _read_all(self._config_files)
        return _parity_data_from_configs(
            atoms_list,
            self._model_files,
            per_atom=self._per_atom,
            show_forces=self._show_forces,
        )


class HeldOutParityPanel(ParityPanel):
    """The real "is AL working" diagnostic: out-of-sample committee parity.

    Scores iteration ``iteration``'s *new* evaluator configs (frames the
    committee below never saw during training) with ``model_files`` -- pass
    the *previous* iteration's committee (the one that actually drove that
    iteration's sampling/selection), not the freshly-retrained one, or this
    degenerates into an in-sample check (see the dashboard plan doc for why:
    ``entry_stage="trainer"`` trains iteration N's committee on data through
    evaluator_it{N} itself). This is the panel whose RMSE/MAE trend across
    iterations is the sharpest signal of whether committee-uncertainty-driven
    sampling is finding genuinely informative configs.

    Args:
        config_files: The iteration's evaluated (DFT-labelled) configs.
        previous_config_files: The previous iteration's evaluated configs
            (empty if none) -- frames already in there are dropped when
            ``config_files`` turns out to be cumulative (see
            :func:`_new_frames`).
        model_files: The committee that drove that iteration's sampling
            (the previous iteration's trained models).
    """

    name = "held_out_parity"

    def __init__(
        self,
        config_files,
        previous_config_files,
        model_files,
        *args,
        per_atom: bool = True,
        show_forces: bool = True,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            show_forces=show_forces,
            plot_kwargs=rasterized_plot_kwargs(plot_kwargs),
            **kwargs,
        )
        self._config_files = [Path(p) for p in config_files]
        self._previous_config_files = [Path(p) for p in previous_config_files]
        self._model_files = [Path(p) for p in model_files]
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> ParityData:
        atoms_list = _new_frames(
            _read_all(self._config_files),
            _read_all(self._previous_config_files),
        )
        if not atoms_list:
            raise ValueError("No new evaluator configs; nothing to score.")
        return _parity_data_from_configs(
            atoms_list,
            self._model_files,
            per_atom=self._per_atom,
            show_forces=self._show_forces,
        )


class CommitteeParitySpec(PanelSpec):
    type: str = "committee_parity"
    per_atom: bool = True
    show_forces: bool = True


def _build_committee_parity(spec, ctx, common):
    if ctx.is_summary:
        return None
    if not (ctx.model_files and ctx.config_files):
        return ctx.skip(
            spec.type,
            f"model_files={len(ctx.model_files)}, "
            f"config_files={len(ctx.config_files)}",
        )
    return CommitteeParityPanel(
        ctx.config_files,
        ctx.model_files,
        per_atom=spec.per_atom,
        show_forces=spec.show_forces,
        **named(spec),
        **common,
    )


class HeldOutParitySpec(PanelSpec):
    type: str = "held_out_parity"
    per_atom: bool = True
    show_forces: bool = True


def _build_held_out_parity(spec, ctx, common):
    if ctx.is_summary:
        return None
    if not (ctx.prev_model_files and ctx.config_files):
        return ctx.skip(
            spec.type,
            f"previous iteration's model_files={len(ctx.prev_model_files)},"
            f" config_files={len(ctx.config_files)}",
        )
    return HeldOutParityPanel(
        ctx.config_files,
        ctx.prev_config_files,
        ctx.prev_model_files,
        per_atom=spec.per_atom,
        show_forces=spec.show_forces,
        **named(spec),
        **common,
    )


register_panel(
    "committee_parity", CommitteeParitySpec, _build_committee_parity
)
register_panel("held_out_parity", HeldOutParitySpec, _build_held_out_parity)
