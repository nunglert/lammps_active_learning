"""Committee-vs-reference parity: overall fit and the real "is AL working"
held-out diagnostic.

Both panels share the same underlying computation (committee re-score of a
set of ``REF_energy``/``REF_forces``-labelled configs against a given set of
committee-member potentials); they differ only in *which* configs and *which*
committee are compared -- see each class's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from alviz.panels.training.parity import ParityData, ParityPanel

from lammps_al.dashboard import run_layout
from lammps_al.dashboard.committee_eval import score_with_committee


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
            *args, show_forces=show_forces, plot_kwargs=plot_kwargs, **kwargs
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
        calchome: The campaign's ``calchome`` directory.
        progress: Pre-loaded ``AL_progress.json`` (see
            ``run_layout.load_progress``); loaded internally if ``None``.
        iteration: Which iteration's new evaluator configs to score.
        model_files: The committee that drove that iteration's sampling
            (typically ``trainer_models_for_iteration(iteration - 1)``).
    """

    name = "held_out_parity"

    def __init__(
        self,
        calchome,
        progress,
        iteration: int,
        model_files,
        *args,
        per_atom: bool = True,
        show_forces: bool = True,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args, show_forces=show_forces, plot_kwargs=plot_kwargs, **kwargs
        )
        self._calchome = Path(calchome)
        self._progress = progress
        self._iteration = iteration
        self._model_files = [Path(p) for p in model_files]
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> ParityData:
        progress = self._progress
        if progress is None:
            progress = run_layout.load_progress(self._calchome)
        atoms_list = run_layout.new_evaluator_configs_for_iteration(
            self._calchome, progress, self._iteration
        )
        if not atoms_list:
            raise ValueError(
                f"No new evaluator configs for iteration {self._iteration}; "
                "nothing to score."
            )
        return _parity_data_from_configs(
            atoms_list,
            self._model_files,
            per_atom=self._per_atom,
            show_forces=self._show_forces,
        )
