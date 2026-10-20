"""Ground-truth labelling actors specific to this project's campaigns.

:class:`lammps_al.label.evaluator.VaspD3Evaluator` -- remote_handler's
``VASPEvaluator`` with DFT-D3 computed by simple-dft-d3 on top of VASP
instead of VASP's own ``IVDW`` (which lacks the three-body term). Not
imported here, so importing ``lammps_al`` does not pull in remote_handler.
"""
