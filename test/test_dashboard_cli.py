"""End-to-end tests for ``lammps_al.dashboard.cli`` (plus the context /
panel-spec / YAML-config layers it drives) against a fully synthetic
``calchome`` tree (no cluster/pyace/LAMMPS needed -- committee scoring is
patched at the panel-module import point, same technique as
``test_dashboard_panels.py``).
"""

import json

import numpy as np
import pytest
from ase import Atoms

ase_io = pytest.importorskip("ase.io")
# Not "alviz" bare -- see test_dashboard_panels.py's comment on why.
pytest.importorskip("alviz.dashboard")
pypdf = pytest.importorskip("pypdf")

from lammps_al.dashboard import cli  # noqa: E402
from lammps_al.dashboard import run_layout  # noqa: E402
from lammps_al.dashboard.context import (  # noqa: E402
    campaign_context,
    iteration_context,
)
from lammps_al.dashboard.panel_specs import (  # noqa: E402
    assemble,
    parse_panel_config,
)

pytestmark = pytest.mark.local


def _make_config(rng, e_ref, n_atoms=3):
    atoms = Atoms(
        ["Cu", "H", "O"][:n_atoms] if n_atoms <= 3 else ["Cu"] * n_atoms,
        positions=rng.uniform(0, 5, size=(n_atoms, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    atoms.info["REF_energy"] = e_ref
    atoms.arrays["REF_forces"] = rng.normal(size=(n_atoms, 3))
    return atoms


def _fake_score_with_committee(model_files, atoms_list, **kwargs):
    energies = [float(a.info["REF_energy"]) + 0.1 for a in atoms_list]
    forces = None
    if kwargs.get("want_forces", True):
        forces = [
            np.asarray(a.arrays["REF_forces"]) + 0.01 for a in atoms_list
        ]
    return energies, forces


def _fake_committee_force_spread(model_files, atoms_list, **kwargs):
    forces = [np.asarray(a.arrays["REF_forces"]) + 0.01 for a in atoms_list]
    spread = [np.full((len(a), 3), 0.05) for a in atoms_list]
    return forces, spread


@pytest.fixture(autouse=True)
def _patch_committee_scoring(monkeypatch):
    """No real pyace anywhere in this file -- every panel module that
    scores with the committee gets a fake."""
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.error_analysis.committee_force_spread",
        _fake_committee_force_spread,
    )
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.committee_parity.score_with_committee",
        _fake_score_with_committee,
    )
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.species_parity.score_with_committee",
        _fake_score_with_committee,
    )
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.cross_iteration.score_with_committee",
        _fake_score_with_committee,
    )
    monkeypatch.setattr(
        "lammps_al.dashboard.panels.test_set.score_with_committee",
        _fake_score_with_committee,
    )


@pytest.fixture
def two_iteration_calchome(tmp_path):
    """A synthetic 2-iteration campaign: iteration 0 bootstrap-trains only
    (entry_stage="trainer" semantics -- no evaluator/sampler output yet),
    iteration 1 has a full ring's worth of recorded + raw artifacts."""
    calchome = tmp_path / "calchome"
    (calchome / "trainer_output").mkdir(parents=True)
    (calchome / "evaluator_output").mkdir(parents=True)

    rng = np.random.default_rng(0)

    # Iteration 0: bootstrap train only, 2 committee members.
    for i in range(2):
        (calchome / "trainer_output" / f"trainer_it0_{i}.yaml").write_text(
            "potential"
        )

    # Iteration 1: 2 more committee members + evaluated configs + a
    # recorded, annotated (not reduced) uncertainty trajectory. Only
    # evaluator_it1 is recorded in AL_progress.json (evaluator_it0 sits on
    # disk unrecorded), so the held-out panel scores all of it -- its
    # cumulative/delta diff is covered in test_dashboard_panels.py.
    for i in range(2):
        (calchome / "trainer_output" / f"trainer_it1_{i}.yaml").write_text(
            "potential"
        )
    configs0 = [_make_config(rng, e_ref=-1.0), _make_config(rng, e_ref=-2.0)]
    configs1 = configs0 + [
        _make_config(rng, e_ref=-3.0),
        _make_config(rng, e_ref=-4.0),
    ]
    ase_io.write(
        calchome / "evaluator_output" / "evaluator_it0_0.extxyz", configs0
    )
    ase_io.write(
        calchome / "evaluator_output" / "evaluator_it1_0.extxyz", configs1
    )

    (calchome / "uncertainty_select_output").mkdir(parents=True)
    traj_atoms = []
    for z in (0.0, 1.0, 2.0):
        atoms = Atoms(
            "CuHO",
            positions=[[0.0, 0.0, z], [0.0, 1.0, z], [0.0, 2.0, z]],
            cell=np.eye(3) * 20,
            pbc=True,
        )
        atoms.arrays["uncertainty"] = np.array([0.1, 0.1, 0.1])
        traj_atoms.append(atoms)
    ase_io.write(
        calchome / "uncertainty_select_output" / "u_it1_0.extxyz",
        traj_atoms,
    )

    (calchome / "AL_progress.json").write_text(
        json.dumps(
            {
                "stages": [
                    "sampler",
                    "uncertainty_select",
                    "selector",
                    "annealer",
                    "evaluator",
                    "trainer",
                ],
                "outputs": {
                    "trainer": {
                        "0": [
                            "trainer_output/trainer_it0_0.yaml",
                            "trainer_output/trainer_it0_1.yaml",
                        ],
                        "1": [
                            "trainer_output/trainer_it1_0.yaml",
                            "trainer_output/trainer_it1_1.yaml",
                        ],
                    },
                    "evaluator": {
                        "1": ["evaluator_output/evaluator_it1_0.extxyz"]
                    },
                    "uncertainty_select": {
                        "1": ["uncertainty_select_output/u_it1_0.extxyz"]
                    },
                },
            }
        )
    )
    return calchome


def test_build_dashboard_pdfs_produces_pages(tmp_path, two_iteration_calchome):
    out_dir = tmp_path / "dashboard_data"
    pages = cli.build_dashboard_pdfs(
        two_iteration_calchome, out_dir, ["Cu", "H", "O"]
    )

    # Iteration 0 has a trained model but no evaluated configs yet -> no
    # per-iteration page. Iteration 1 has both -> one page. Plus one
    # campaign-summary page (uncertainty trend/depth, from iteration 1's
    # raw trajectory).
    assert len(pages) == 2
    for page in pages:
        assert page.exists() and page.stat().st_size > 0


def test_build_dashboard_pdfs_only_filters_panels(
    tmp_path, two_iteration_calchome
):
    out_dir = tmp_path / "dashboard_data"
    pages = cli.build_dashboard_pdfs(
        two_iteration_calchome,
        out_dir,
        ["Cu", "H", "O"],
        only=["committee_parity"],
    )
    # Only the per-iteration page has a "committee_parity"-named entry; the
    # campaign page's entries are all excluded by `only` and so produces no
    # page at all.
    assert len(pages) == 1


def test_summary_page_data_keyed_on_resolved_artifacts(
    tmp_path, two_iteration_calchome
):
    out_dir = tmp_path / "dashboard_data"
    species = ["Cu", "H", "O"]
    cli.build_dashboard_pdfs(two_iteration_calchome, out_dir, species)
    keyed = list((out_dir / "campaign").iterdir())
    assert len(keyed) == 1 and (keyed[0] / "page.pdf").exists()

    # Unchanged calchome -> same key, cached JSON reused.
    cli.build_dashboard_pdfs(two_iteration_calchome, out_dir, species)
    assert list((out_dir / "campaign").iterdir()) == keyed

    # A newly finished iteration changes the key; the stale dir is pruned.
    progress = run_layout.load_progress(two_iteration_calchome)
    ctx = campaign_context(two_iteration_calchome, progress, range(2), species)
    grown = campaign_context(
        two_iteration_calchome, progress, range(2), species
    )
    grown.uncertainty_trajectories[2] = tmp_path / "it2.extxyz"
    assert cli._summary_cache_key(ctx) == keyed[0].name
    assert cli._summary_cache_key(grown) != keyed[0].name
    (keyed[0].parent / "deadbeef0000").mkdir()
    datapath = cli._summary_datapath(out_dir, grown)
    assert not keyed[0].exists()
    assert not (keyed[0].parent / "deadbeef0000").exists()
    assert datapath.name == cli._summary_cache_key(grown)


def test_main_writes_merged_pdf(tmp_path, two_iteration_calchome):
    output = tmp_path / "dashboard.pdf"
    rc = cli.main([str(two_iteration_calchome), "--output", str(output)])

    assert rc == 0
    assert output.exists() and output.stat().st_size > 0


def test_main_returns_nonzero_when_no_pages(tmp_path):
    calchome = tmp_path / "empty_calchome"
    calchome.mkdir()
    (calchome / "AL_progress.json").write_text(
        json.dumps({"stages": [], "outputs": {}})
    )
    output = tmp_path / "dashboard.pdf"

    rc = cli.main([str(calchome), "--output", str(output)])

    assert rc == 1
    assert not output.exists()


def _names(entries):
    return [e.name for e in entries]


def test_iteration_page_empty_without_artifacts(tmp_path):
    calchome = tmp_path / "calchome"
    calchome.mkdir()
    (calchome / "AL_progress.json").write_text(
        json.dumps({"stages": [], "outputs": {}})
    )
    progress = run_layout.load_progress(calchome)

    ctx = iteration_context(calchome, progress, 0, ["Cu", "H", "O"])
    assert assemble(ctx, datapath=tmp_path) == []


def test_default_specs_gate_on_view_and_inputs(
    tmp_path, two_iteration_calchome
):
    progress = run_layout.load_progress(two_iteration_calchome)
    species = ["Cu", "H", "O"]

    # Iteration 0: trained models but no evaluated configs -> nothing.
    ctx0 = iteration_context(two_iteration_calchome, progress, 0, species)
    assert assemble(ctx0, datapath=tmp_path) == []

    # Iteration 1: models + configs + a previous committee + an uncertainty
    # trajectory; no selector/annealer/LAMMPS artifacts.
    ctx1 = iteration_context(two_iteration_calchome, progress, 1, species)
    assert _names(assemble(ctx1, datapath=tmp_path)) == [
        "committee_parity",
        "species_force_parity",
        "uncertainty_timeline",
        "held_out_parity",
        "uncertainty_calibration",
    ]

    summary = campaign_context(
        two_iteration_calchome, progress, range(2), species
    )
    assert _names(assemble(summary, datapath=tmp_path)) == [
        "pipeline_attrition",
        "cross_iteration_errors",
        "uncertainty_trend",
        "uncertainty_depth_profile",
    ]


def test_campaign_context_skips_unfinished_iterations(two_iteration_calchome):
    progress = run_layout.load_progress(two_iteration_calchome)
    ctx = campaign_context(
        two_iteration_calchome, progress, range(3), ["Cu", "H", "O"]
    )
    assert list(ctx.uncertainty_trajectories) == [1]


def test_panel_config_disable_add_and_species_override(
    tmp_path, two_iteration_calchome
):
    progress = run_layout.load_progress(two_iteration_calchome)
    ctx = iteration_context(
        two_iteration_calchome, progress, 1, ["Cu", "H", "O"]
    )
    panel_config = parse_panel_config(
        {
            "disable": ["held_out_parity"],
            "add": [
                {
                    "type": "species_force_parity",
                    "name": "species_force_parity_cu",
                    "species": ["Cu"],
                }
            ],
        }
    )

    entries = assemble(ctx, datapath=tmp_path, panel_config=panel_config)

    assert _names(entries) == [
        "committee_parity",
        "species_force_parity",
        "uncertainty_timeline",
        "uncertainty_calibration",
        "species_force_parity_cu",
    ]
    # Unset `species` inherits the run-global list; an explicit one wins.
    assert entries[1]._species == ["Cu", "H", "O"]
    assert entries[-1]._species == ["Cu"]


def test_test_set_errors_panel_needs_an_existing_test_set(
    tmp_path, two_iteration_calchome
):
    progress = run_layout.load_progress(two_iteration_calchome)
    summary = campaign_context(
        two_iteration_calchome, progress, range(2), ["Cu", "H", "O"]
    )
    test_set = two_iteration_calchome / "test.xyz"
    ase_io.write(test_set, [_make_config(np.random.default_rng(0), -1.0)])

    def names(spec):
        config = parse_panel_config({"panels": [spec]})
        return _names(
            assemble(summary, datapath=tmp_path, panel_config=config)
        )

    assert names({"type": "test_set_errors"}) == []
    assert names({"type": "test_set_errors", "test_set": "missing.xyz"}) == []
    assert names({"type": "test_set_errors", "test_set": "test.xyz"}) == [
        "test_set_errors"
    ]


def test_panel_config_panels_replaces_defaults(
    tmp_path, two_iteration_calchome
):
    progress = run_layout.load_progress(two_iteration_calchome)
    ctx = iteration_context(
        two_iteration_calchome, progress, 1, ["Cu", "H", "O"]
    )
    panel_config = parse_panel_config(
        {"panels": [{"type": "committee_parity", "per_atom": False}]}
    )
    entries = assemble(ctx, datapath=tmp_path, panel_config=panel_config)
    assert _names(entries) == ["committee_parity"]
    assert entries[0]._per_atom is False


@pytest.mark.parametrize(
    "raw",
    [
        {"add": [{"type": "no_such_panel"}]},
        {"add": [{"type": "box_rdf", "no_such_field": 1}]},
        {"add": [{"type": "box_rdf", "source": "mid_anneal"}]},
    ],
)
def test_parse_panel_config_rejects_invalid_specs(raw):
    with pytest.raises(ValueError):
        parse_panel_config(raw)


def test_main_applies_yaml_config(tmp_path, two_iteration_calchome):
    pytest.importorskip("yaml")
    (two_iteration_calchome / "dashboard.yaml").write_text(
        "only: [committee_parity]\n"
    )
    output = tmp_path / "dashboard.pdf"

    rc = cli.main([str(two_iteration_calchome), "--output", str(output)])

    # Same as test_build_dashboard_pdfs_only_filters_panels: one page.
    assert rc == 0
    assert len(pypdf.PdfReader(output).pages) == 1


def test_main_cli_flag_overrides_yaml_config(tmp_path, two_iteration_calchome):
    pytest.importorskip("yaml")
    config = tmp_path / "custom.yaml"
    config.write_text("only: [committee_parity]\n")
    output = tmp_path / "dashboard.pdf"

    rc = cli.main(
        [
            str(two_iteration_calchome),
            "--config",
            str(config),
            "--only",
            "uncertainty_trend",
            "--output",
            str(output),
        ]
    )

    # Only the campaign page has an uncertainty_trend entry.
    assert rc == 0
    assert len(pypdf.PdfReader(output).pages) == 1


def test_main_rejects_unknown_config_key(tmp_path, two_iteration_calchome):
    pytest.importorskip("yaml")
    (two_iteration_calchome / "dashboard.yaml").write_text("specis: [Cu]\n")
    with pytest.raises(ValueError, match="specis"):
        cli.main([str(two_iteration_calchome)])


def test_main_passes_grid_sizing_from_config(
    monkeypatch, tmp_path, two_iteration_calchome
):
    pytest.importorskip("yaml")
    (two_iteration_calchome / "dashboard.yaml").write_text(
        "ncols: 3\nwidth-per-panel: 5.5\n"
    )
    seen = []
    real = cli.Dashboard.from_entries

    def spy(entries, **kwargs):
        seen.append(kwargs)
        return real(entries, **kwargs)

    monkeypatch.setattr(cli.Dashboard, "from_entries", spy)
    output = tmp_path / "dashboard.pdf"

    assert (
        cli.main([str(two_iteration_calchome), "--output", str(output)]) == 0
    )
    assert seen
    for kwargs in seen:
        assert kwargs["ncols"] == 3
        assert kwargs["width_per_panel"] == 5.5
        assert kwargs["height_per_panel"] == 4.0
