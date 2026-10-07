"""ACE-committee dashboard panels (``alviz.Entry`` subclasses).

Each panel module registers its own ``alviz.PanelSpec`` + builder (see
:mod:`lammps_al.dashboard.panel_specs`); importing this package runs every
registration.
"""

from lammps_al.dashboard.panels import (  # noqa: F401
    annealer_thermo,
    box_structure,
    committee_parity,
    cross_iteration,
    error_analysis,
    pipeline,
    roi_spatial,
    sampler_thermo,
    species_parity,
    training_fit,
    uncertainty,
)
