"""VASP evaluator with DFT-D3 from simple-dft-d3 instead of VASP's ``IVDW``.

VASP's built-in D3 (``IVDW=11``/``12``) has no three-body
(Axilrod-Teller-Muto) term, so labels it produces differ from labels made
with simple-dft-d3 including that term: by a roughly constant energy per
atom plus small forces. :class:`VaspD3Evaluator` keeps remote_handler's
``VASPEvaluator`` machinery unchanged and only swaps the single-point
payload (``scripts/vasp_d3_singlepoint.py``), which runs VASP without D3 and
adds simple-dft-d3's energy and forces on top -- the same route as a
database labelled with ASE's ``dftd3`` calculator.
"""

from __future__ import annotations

from remote_handler.workflows.evaluators import VASPEvaluator

from lammps_al.label.scripts import script_dir

_D3_KEYS = ("method", "damping", "atm")
_D3_DAMPINGS = ("d3zero", "d3bj")


class VaspD3Evaluator(VASPEvaluator):
    """``VASPEvaluator`` whose labels add simple-dft-d3 dispersion.

    Args:
        d3: ``{"method": str, "damping": "d3zero" | "d3bj", "atm": bool}``
            -- e.g. ``{"method": "RPBE", "damping": "d3zero", "atm": True}``.
            All three keys are required: ``atm`` (the three-body term) in
            particular is never left to simple-dft-d3's default, which has
            changed between its versions.
        **kwargs: Passed to ``VASPEvaluator``. ``incar_kwargs`` must not
            set ``IVDW`` (dispersion would be counted twice);
            ``singlepoint_script`` defaults to this package's
            ``vasp_d3_singlepoint.py``.

    Raises:
        ValueError: If ``d3`` is incomplete/invalid or ``IVDW`` is set.

    The cluster env running the payload needs the ``[label]`` extra
    (simple-dft-d3's ``dftd3`` package) and a remote_handler whose
    ``scripts.dft_singlepoint`` provides ``scf_converged``.
    """

    _calc_config_keys = VASPEvaluator._calc_config_keys + ("d3",)

    def __init__(self, *, d3: dict, **kwargs):
        missing = [k for k in _D3_KEYS if k not in d3]
        if missing:
            raise ValueError(f"d3 is missing {missing} (needs {_D3_KEYS})")
        if d3["damping"] not in _D3_DAMPINGS:
            raise ValueError(
                f"d3['damping'] must be one of {_D3_DAMPINGS}, got "
                f"{d3['damping']!r}"
            )
        # Set before VASPEvaluator.__init__, which builds calc_config.
        self.d3 = {k: d3[k] for k in _D3_KEYS}
        kwargs.setdefault(
            "singlepoint_script", str(script_dir() / "vasp_d3_singlepoint.py")
        )
        super().__init__(**kwargs)
        incar = {k.lower(): v for k, v in self.incar_kwargs.items()}
        if int(incar.get("ivdw") or 0) != 0:
            raise ValueError(
                "incar_kwargs sets IVDW: VASP's own D3 would be added on top "
                "of simple-dft-d3's. Leave IVDW unset with VaspD3Evaluator."
            )

    def _build_calc_config(self) -> dict:
        return {**super()._build_calc_config(), "d3": dict(self.d3)}
