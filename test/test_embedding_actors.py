"""Tests for the amorphous-embedding host-side actors (ROIExtractor,
EmbeddingAnnealer) in stub mode.

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode tests run with NO cluster, SSH, or LAMMPS -- only remote_handler +
ase + numpy on the host.
"""

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402
from remote_handler.exceptions import WorkflowError  # noqa: E402

from lammps_al.embedding.annealer import EmbeddingAnnealer  # noqa: E402
from lammps_al.embedding.extractor import ROIExtractor  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_remote_fs():
    mock_fs = MagicMock()
    mock_fs.slurm_config = {"savepath": "/remote/path"}
    return mock_fs


@pytest.fixture
def mock_cluster_config():
    return ClusterConfig(
        remote_machine="mock",
        partition="default",
        conda_env="test",
        conda_source="/opt/conda",
        savepath="/tmp/remote_handler",
        template="mock.sh.j2",
    )


def _extractor_kwargs(mock_remote_fs, mock_cluster_config, **overrides):
    kwargs = dict(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        specorder=["Cu", "H", "O"],
        stub=True,
    )
    kwargs.update(overrides)
    return kwargs


def _annealer_kwargs(mock_remote_fs, mock_cluster_config, **overrides):
    kwargs = dict(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        lammps_path="/opt/lammps/lmp",
        pair_style="grace",
        pair_coeff_path="/opt/models/model",
        species=["Cu", "H", "O"],
        masses=[63.5, 1.0, 16.0],
        anneal_stages=[
            {
                "species": ["Cu"],
                "temperatures": [300.0, 600.0, 300.0],
                "run_steps": [50000, 100000],
                "pressure_xy": 1e4,
            },
        ],
        anneal_cutoff=4.0,
        stub=True,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# ROIExtractor
# ---------------------------------------------------------------------------


def test_roi_extractor_stub_mode(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    extractor = ROIExtractor(
        **_extractor_kwargs(mock_remote_fs, mock_cluster_config)
    )
    extractor.update(tmp_path, tmp_path / "traj.xyz")
    out_dir = extractor.get_selected_configs()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 3
    for entry in manifest:
        assert (out_dir / entry["data_file"]).exists()


def test_roi_extractor_writes_specorder_to_extract_input(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """The extract input must carry the caller's ``specorder`` verbatim so
    per-box type numbering stays consistent with the downstream anneal
    deck's mass/pair_coeff/Z_of_type lines."""
    monkeypatch.setattr(
        "lammps_al.embedding.extractor.SingleScriptTask", MagicMock()
    )
    specorder = ["Cu", "H", "O"]
    extractor = ROIExtractor(
        **_extractor_kwargs(
            mock_remote_fs,
            mock_cluster_config,
            specorder=specorder,
            stub=False,
        )
    )

    extractor.update(tmp_path, tmp_path / "traj.xyz")

    written = json.loads(extractor.extract_input_file.read_text())
    assert written["specorder"] == specorder


def test_roi_extractor_writes_phase_cutoff(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """phase_cutoff must reach the extract input JSON so extract.py's
    find_rois() call can apply the phase-proximity filter."""
    monkeypatch.setattr(
        "lammps_al.embedding.extractor.SingleScriptTask", MagicMock()
    )
    extractor = ROIExtractor(
        **_extractor_kwargs(
            mock_remote_fs,
            mock_cluster_config,
            phase_species=["Cu"],
            phase_cutoff=6.0,
            stub=False,
        )
    )

    extractor.update(tmp_path, tmp_path / "traj.xyz")

    written = json.loads(extractor.extract_input_file.read_text())
    assert written["phase_species"] == ["Cu"]
    assert written["phase_cutoff"] == 6.0


def test_roi_extractor_writes_normal_patch_radius(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """normal_patch_radius must reach the extract input JSON so extract.py
    can default it to protect_radius on the remote side."""
    monkeypatch.setattr(
        "lammps_al.embedding.extractor.SingleScriptTask", MagicMock()
    )
    extractor = ROIExtractor(
        **_extractor_kwargs(
            mock_remote_fs,
            mock_cluster_config,
            normal_patch_radius=6.0,
            stub=False,
        )
    )

    extractor.update(tmp_path, tmp_path / "traj.xyz")

    written = json.loads(extractor.extract_input_file.read_text())
    assert written["normal_patch_radius"] == 6.0


def test_roi_extractor_molecule_requires_valid_solvent_type(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    extractor = ROIExtractor(
        **_extractor_kwargs(
            mock_remote_fs,
            mock_cluster_config,
            cleanup_mode="molecule",
            solvent=12345,  # not an Atoms or str
            stub=False,
        )
    )

    with pytest.raises(WorkflowError):
        extractor.update(tmp_path, tmp_path / "traj.xyz")


# ---------------------------------------------------------------------------
# EmbeddingAnnealer
# ---------------------------------------------------------------------------


def test_embedding_annealer_stub_mode(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(mock_remote_fs, mock_cluster_config)
    )
    # a hand-written manifest stands in for a real ROIExtractor's output.
    extract_out = tmp_path / "extract_out"
    extract_out.mkdir()
    manifest = [
        {"index": i, "roi_position": [1.0, 1.0, 1.0], "data_file": "x.data"}
        for i in range(2)
    ]
    (extract_out / "manifest.json").write_text(json.dumps(manifest))

    annealer.update(tmp_path, extract_out)
    output_path = annealer.get_sampled_configs()

    assert output_path.exists()
    configs = ase.io.read(output_path, index=":")
    assert len(configs) == 2
    for atoms in configs:
        assert atoms.info["topology_mode"] == "bulk"


@pytest.mark.parametrize(
    "topology_mode, expected_epsilon",
    [
        ("bulk", 0.0),
        ("surface", 10.0),
        ("interface", 10.0),
    ],
)
def test_embedding_annealer_z_wall_epsilon_auto_from_topology(
    mock_remote_fs, mock_cluster_config, topology_mode, expected_epsilon
):
    """z_wall_epsilon left unset auto-decides from topology_mode: a "bulk"
    box is fully periodic in z (nothing to drift into, so the wall would be
    inert overhead), while "surface"/"interface" boxes have a free surface
    where mobile atoms could otherwise drift into vacuum/solvent
    headspace."""
    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(
            mock_remote_fs, mock_cluster_config, topology_mode=topology_mode
        )
    )
    assert annealer.z_wall_epsilon == expected_epsilon


def test_embedding_annealer_z_wall_epsilon_explicit_overrides_auto(
    mock_remote_fs, mock_cluster_config
):
    """An explicit z_wall_epsilon always wins, regardless of topology_mode."""
    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(
            mock_remote_fs,
            mock_cluster_config,
            topology_mode="bulk",
            z_wall_epsilon=7.0,
        )
    )
    assert annealer.z_wall_epsilon == 7.0


# ---------------------------------------------------------------------------
# ROIExtractor -> EmbeddingAnnealer (chained)
# ---------------------------------------------------------------------------


def test_roi_extractor_then_embedding_annealer_chain_stub_mode(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    """The two stages actually pass data forward: however many ROIs the
    extractor reports is exactly how many boxes the annealer anneals --
    even in stub mode, unlike the old combined sampler's stub path, which
    fabricated a hardcoded box count independent of extraction."""
    extractor = ROIExtractor(
        **_extractor_kwargs(mock_remote_fs, mock_cluster_config)
    )
    extractor.update(tmp_path / "0", tmp_path / "traj.xyz")
    extracted = extractor.get_selected_configs()

    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(mock_remote_fs, mock_cluster_config)
    )
    annealer.update(tmp_path / "0", extracted)
    output_path = annealer.get_sampled_configs()

    manifest = json.loads((extracted / "manifest.json").read_text())
    configs = ase.io.read(output_path, index=":")
    assert len(configs) == len(manifest)
