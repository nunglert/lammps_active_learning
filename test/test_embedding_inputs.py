"""Tests for the amorphous-embedding input schemas (MatrixEmbeddingInput,
EmbeddingAnnealInput).

Needs the [run] extra (remote_handler provides the ``ScriptInput`` base); the
module self-skips otherwise.
"""

import pytest

pytest.importorskip("remote_handler")
from remote_handler.exceptions import ScriptInputError  # noqa: E402

from lammps_al.embedding.inputs import (  # noqa: E402
    EmbeddingAnnealInput,
    MatrixEmbeddingInput,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# MatrixEmbeddingInput
# ---------------------------------------------------------------------------


def test_matrix_embedding_input_bulk_defaults_valid():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
    )
    cfg.validate()  # should not raise


def test_matrix_embedding_input_normal_patch_radius_must_be_positive():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        normal_patch_radius=0.0,
    )
    with pytest.raises(ScriptInputError, match="normal_patch_radius"):
        cfg.validate()


def test_matrix_embedding_input_phase_cutoff_must_be_positive():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        phase_species=["Cu"],
        phase_cutoff=0.0,
    )
    with pytest.raises(ScriptInputError, match="phase_cutoff"):
        cfg.validate()


def test_matrix_embedding_input_phase_cutoff_requires_phase_species():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        phase_cutoff=5.0,
    )
    with pytest.raises(ScriptInputError, match="phase_species"):
        cfg.validate()


def test_matrix_embedding_input_max_num_structures_must_be_positive():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        max_num_structures=0,
    )
    with pytest.raises(ScriptInputError, match="max_num_structures"):
        cfg.validate()


def test_matrix_embedding_input_max_num_structures_none_is_valid():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        max_num_structures=None,
    )
    cfg.validate()  # should not raise


def test_matrix_embedding_input_interface_requires_remove_species():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        topology_mode="interface",
    )
    with pytest.raises(ScriptInputError, match="remove_species"):
        cfg.validate()


def test_matrix_embedding_input_molecule_requires_exactly_one_solvent_source():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        cleanup_mode="molecule",
    )
    with pytest.raises(ScriptInputError, match="solvent"):
        cfg.validate()

    cfg2 = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        cleanup_mode="molecule",
        solvent_label="H2O",
        solvent_file="x.xyz",
    )
    with pytest.raises(ScriptInputError, match="solvent"):
        cfg2.validate()


def test_matrix_embedding_input_stoichiometry_requires_ratio_dict():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        cleanup_mode="stoichiometry",
    )
    with pytest.raises(ScriptInputError, match="stoichiometry"):
        cfg.validate()


def test_matrix_embedding_input_invalid_topology_mode():
    cfg = MatrixEmbeddingInput(
        trajectory="traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        topology_mode="not-a-mode",
    )
    with pytest.raises(ScriptInputError, match="topology_mode"):
        cfg.validate()


# ---------------------------------------------------------------------------
# EmbeddingAnnealInput
# ---------------------------------------------------------------------------


def _one_stage_kwargs():
    return dict(
        stages=[
            {
                "species": ["Cu"],
                "temperatures": [300.0, 600.0, 300.0],
                "run_steps": [50000, 100000],
                "pressure_xy": 1e4,
            },
        ],
    )


def test_embedding_anneal_input_bulk_defaults_valid(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="embedded_0.data",
        lammps_path=lmp,
        template_file="embedding_anneal.in.j2",
        cutoff=4.0,
        roi_x=1.0,
        roi_y=1.0,
        roi_z=1.0,
        pair_style="grace",
        pair_coeff_path="model",
        species=["Cu", "H", "O"],
        masses=[63.5, 1.0, 16.0],
        **_one_stage_kwargs(),
    )
    cfg.validate()


def test_embedding_anneal_input_n_processors_must_be_positive(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="embedded_0.data",
        lammps_path=lmp,
        template_file="embedding_anneal.in.j2",
        cutoff=4.0,
        roi_x=1.0,
        roi_y=1.0,
        roi_z=1.0,
        pair_style="grace",
        pair_coeff_path="model",
        species=["Cu", "H", "O"],
        masses=[63.5, 1.0, 16.0],
        n_processors=0,
        **_one_stage_kwargs(),
    )
    with pytest.raises(ScriptInputError, match="n_processors"):
        cfg.validate()


def test_embedding_anneal_input_arbitrary_segment_count_valid(tmp_path):
    """A stage isn't limited to a 2-segment start/mid/end ramp -- any
    number of temperature waypoints (and therefore segments) works."""
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu"],
        masses=[63.5],
        stages=[
            {
                "species": ["Cu"],
                "temperatures": [300, 500, 700, 500, 300],
                "run_steps": [10000, 10000, 10000, 10000],
                "pressure_xy": 1.0,
            },
        ],
    )
    cfg.validate()


def test_embedding_anneal_input_stage_missing_required_key(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu"],
        masses=[63.5],
        stages=[{"species": ["Cu"], "temperatures": [300.0, 300.0]}],
    )
    with pytest.raises(ScriptInputError, match="run_steps"):
        cfg.validate()


def test_embedding_anneal_input_run_steps_must_match_segment_count(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu"],
        masses=[63.5],
        stages=[
            {
                "species": ["Cu"],
                "temperatures": [300.0, 600.0, 300.0],  # 2 segments
                "run_steps": [50000],  # only 1
                "pressure_xy": 1.0,
            },
        ],
    )
    with pytest.raises(ScriptInputError, match="run_steps"):
        cfg.validate()


def test_embedding_anneal_input_stages_cannot_share_species(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu", "H", "O"],
        masses=[63.5, 1.0, 16.0],
        stages=[
            {
                "species": ["Cu", "H"],
                "temperatures": [300.0, 300.0],
                "run_steps": [1000],
            },
            {
                "species": ["H", "O"],
                "temperatures": [300.0, 300.0],
                "run_steps": [1000],
            },
        ],
    )
    with pytest.raises(ScriptInputError, match="not share species"):
        cfg.validate()


def test_embedding_anneal_input_stage_species_must_be_known_species(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu"],
        masses=[63.5],
        stages=[
            {
                "species": ["Xx"],
                "temperatures": [300.0, 300.0],
                "run_steps": [1000],
            },
        ],
    )
    with pytest.raises(ScriptInputError, match="not in 'species'"):
        cfg.validate()


def test_embedding_anneal_input_species_masses_length_mismatch(tmp_path):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu", "O"],
        masses=[63.5],
        **_one_stage_kwargs(),
    )
    with pytest.raises(ScriptInputError, match="species and masses"):
        cfg.validate()


def test_embedding_anneal_input_missing_lammps_executable(tmp_path):
    cfg = EmbeddingAnnealInput(
        data_file="d.data",
        lammps_path=tmp_path / "does_not_exist",
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu"],
        masses=[63.5],
        **_one_stage_kwargs(),
    )
    with pytest.raises(ScriptInputError, match="LAMMPS executable"):
        cfg.validate()


def _valid_anneal_cfg(tmp_path, **overrides):
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n")
    kwargs = dict(
        data_file="d.data",
        lammps_path=lmp,
        template_file="t.j2",
        cutoff=4.0,
        roi_x=1,
        roi_y=1,
        roi_z=1,
        pair_style="grace",
        pair_coeff_path="m",
        species=["Cu"],
        masses=[63.5],
        **_one_stage_kwargs(),
    )
    kwargs.update(overrides)
    return EmbeddingAnnealInput(**kwargs)


def test_embedding_anneal_input_roi_repel_margin_negative(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, roi_repel_margin=-0.1)
    with pytest.raises(ScriptInputError, match="roi_repel_margin"):
        cfg.validate()


def test_embedding_anneal_input_roi_repel_margin_not_smaller_than_cutoff(
    tmp_path,
):
    cfg = _valid_anneal_cfg(tmp_path, cutoff=4.0, roi_repel_margin=4.0)
    with pytest.raises(ScriptInputError, match="roi_repel_margin"):
        cfg.validate()


def test_embedding_anneal_input_roi_repel_epsilon_negative(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, roi_repel_epsilon=-1.0)
    with pytest.raises(ScriptInputError, match="roi_repel_epsilon"):
        cfg.validate()


def test_embedding_anneal_input_roi_repel_wall_cutoff_not_positive(
    tmp_path,
):
    cfg = _valid_anneal_cfg(tmp_path, roi_repel_wall_cutoff=0.0)
    with pytest.raises(ScriptInputError, match="roi_repel_wall_cutoff"):
        cfg.validate()


def test_embedding_anneal_input_z_wall_margin_negative(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, z_wall_margin=-1.0)
    with pytest.raises(ScriptInputError, match="z_wall_margin"):
        cfg.validate()


def test_embedding_anneal_input_z_wall_epsilon_negative(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, z_wall_epsilon=-1.0)
    with pytest.raises(ScriptInputError, match="z_wall_epsilon"):
        cfg.validate()


def test_embedding_anneal_input_z_wall_cutoff_not_positive(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, z_wall_cutoff=0.0)
    with pytest.raises(ScriptInputError, match="z_wall_cutoff"):
        cfg.validate()


def test_embedding_anneal_input_repel_and_zwall_defaults_valid(tmp_path):
    """Defaults alone (no explicit repel/z-wall kwargs) must validate --
    every existing caller that hasn't been touched keeps working."""
    cfg = _valid_anneal_cfg(tmp_path)
    cfg.validate()


def _stage_kwargs(**stage_overrides):
    stage = {
        "species": ["Cu"],
        "temperatures": [300.0, 600.0, 300.0],
        "run_steps": [50000, 100000],
        "pressure_xy": 1e4,
    }
    stage.update(stage_overrides)
    return dict(stages=[stage])


def test_embedding_anneal_input_stage_z_wall_margin_negative(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, **_stage_kwargs(z_wall_margin=-1.0))
    with pytest.raises(ScriptInputError, match=r"stages\[0\].*z_wall_margin"):
        cfg.validate()


def test_embedding_anneal_input_stage_z_wall_epsilon_negative(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, **_stage_kwargs(z_wall_epsilon=-1.0))
    with pytest.raises(ScriptInputError, match=r"stages\[0\].*z_wall_epsilon"):
        cfg.validate()


def test_embedding_anneal_input_stage_z_wall_cutoff_not_positive(tmp_path):
    cfg = _valid_anneal_cfg(tmp_path, **_stage_kwargs(z_wall_cutoff=0.0))
    with pytest.raises(ScriptInputError, match=r"stages\[0\].*z_wall_cutoff"):
        cfg.validate()


def test_embedding_anneal_input_stage_zwall_overrides_valid(tmp_path):
    """A stage overriding every z_wall_* key with valid values must
    validate cleanly."""
    cfg = _valid_anneal_cfg(
        tmp_path,
        **_stage_kwargs(
            z_wall_margin=2.0,
            z_wall_epsilon=5.0,
            z_wall_cutoff=0.5,
        ),
    )
    cfg.validate()
