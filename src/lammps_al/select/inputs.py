"""Input schema for the committee-uncertainty selection remote payload.

``CommitteeSelectionInput`` is the JSON contract between the host-side
:class:`lammps_al.select.selector.CommitteeUncertaintySelector` (which writes
it) and the cluster-side payload ``scripts/committee_selection.py`` (which
loads and validates it). It subclasses ``remote_handler``'s ``ScriptInput``,
mirroring :class:`lammps_al.run.inputs.LammpsSamplingInput`.

Kept in this package (not in ``remote_handler``) for the same reason as
``LammpsSamplingInput``: ``remote_handler`` owns the ``Selector`` interface
and the remote-execution machinery, while the concrete committee-scoring
selector -- and the input contract it speaks -- belongs here.

These fields (``model_files``, ``cutoff_radius``, ``device``, ``backend``)
previously lived on ``LammpsSamplingInput`` before committee scoring was
split out of the MD-running sampler into this separate actor. Deliberately
has no ``bin_count``/reduction field -- this payload only annotates, it
does not select; see ``committee_selection.py``'s module docstring and
:class:`lammps_al.select.selector.PeakUncertaintySelector` for the
separate, optional, host-side peak-reduction step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import ScriptInput, _require_path


@dataclass
class CommitteeSelectionInput(ScriptInput):
    """Input configuration for ``scripts/committee_selection.py``.

    Attributes:
        trajectory: Name of the extxyz trajectory to score (produced by
            ``LammpsSampler``'s payload -- already has correct chemical
            symbols, so this selector needs no ``element_mapping``).
        model_files: Paths to every committee member's model file. Format
            depends on ``backend``: MACE ``.model`` files, an ACE
            pacemaker ``.yaml``/``.yace`` file per member, or a GRACE
            gracemaker model directory per member.
        cutoff_radius: Cutoff radius for local uncertainty aggregation
            (Angstrom)
        device: compute device for the committee calculator, where the
            backend's calculator takes one ("cuda" or "cpu"; MACE only)
        backend: ``"mace"``, ``"ace"``, or ``"grace"`` -- see
            ``committee_selection.py``'s module docstring for the
            calculator dispatch.

    Example:
        >>> config = CommitteeSelectionInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    _path_fields: ClassVar[tuple[str, ...]] = ("trajectory",)

    trajectory: Path
    model_files: list[str]
    cutoff_radius: float
    device: str = "cuda"
    backend: str = "mace"

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.model_files:
            raise ScriptInputError("model_files cannot be empty")

        if self.cutoff_radius <= 0:
            raise ScriptInputError(
                f"cutoff_radius must be positive, got {self.cutoff_radius}"
            )

        if self.device not in ("cpu", "cuda"):
            raise ScriptInputError(
                f"device must be 'cpu' or 'cuda', got {self.device!r}"
            )

        if self.backend not in ("mace", "ace", "grace"):
            raise ScriptInputError(
                "backend must be 'mace', 'ace', or 'grace', got "
                f"{self.backend!r}"
            )

        if root is not None:
            _require_path(root, str(self.trajectory), "trajectory")
            for m in self.model_files:
                _require_path(root, m, "model file")
