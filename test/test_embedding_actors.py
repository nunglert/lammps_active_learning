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
from ase import Atoms  # noqa: E402
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


def test_roi_extractor_writes_max_num_structures(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """max_num_structures must reach the extract input JSON so extract.py
    can cap the number of carved boxes by uncertainty rank."""
    monkeypatch.setattr(
        "lammps_al.embedding.extractor.SingleScriptTask", MagicMock()
    )
    extractor = ROIExtractor(
        **_extractor_kwargs(
            mock_remote_fs,
            mock_cluster_config,
            max_num_structures=100,
            stub=False,
        )
    )

    extractor.update(tmp_path, tmp_path / "traj.xyz")

    written = json.loads(extractor.extract_input_file.read_text())
    assert written["max_num_structures"] == 100


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


def test_embedding_annealer_array_job_builds_index_aligned_resources(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """array_job=True must build one config + one structure Writeable per
    box, index-aligned with the manifest (ArrayScriptTask writes
    individual_ressources[k][i] into calcdir/i for every resource kind k),
    upload the template/potential once as shared resources, and never touch
    SingleScriptTask."""
    array_task_cls = MagicMock()
    single_task_cls = MagicMock()
    monkeypatch.setattr(
        "lammps_al.embedding.annealer.ArrayScriptTask", array_task_cls
    )
    monkeypatch.setattr(
        "lammps_al.embedding.annealer.SingleScriptTask", single_task_cls
    )

    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(
            mock_remote_fs, mock_cluster_config, array_job=True, stub=False
        )
    )
    extract_out = tmp_path / "extract_out"
    extract_out.mkdir()
    manifest = [
        {
            "index": i,
            "roi_position": [1.0, 1.0, 1.0],
            "data_file": f"box_{i}.data",
        }
        for i in range(3)
    ]
    (extract_out / "manifest.json").write_text(json.dumps(manifest))
    for entry in manifest:
        (extract_out / entry["data_file"]).write_text("fake lammps-data")

    annealer.update(tmp_path, extract_out)

    single_task_cls.assert_not_called()
    array_task_cls.assert_called_once()
    _, kwargs = array_task_cls.call_args
    configs, structures = kwargs["individual_ressources"]
    assert [c.content["data_file"] for c in configs] == [
        "box_0.data",
        "box_1.data",
        "box_2.data",
    ]
    assert all(c.content["template_file"].startswith("../") for c in configs)
    assert all(c.content["pair_coeff_path"].startswith("../") for c in configs)
    assert [s.filename for s in structures] == [
        "box_0.data",
        "box_1.data",
        "box_2.data",
    ]
    assert [s.content for s in structures] == [
        extract_out / e["data_file"] for e in manifest
    ]

    array_task = array_task_cls.return_value
    array_task.allocate_data.assert_called_once()
    array_task.copy_path.assert_any_call(annealer.anneal_template_path)
    array_task.copy_path.assert_any_call(annealer.pair_coeff_path)


def test_embedding_annealer_array_job_get_sampled_configs_combines_boxes(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    """get_sampled_configs()'s array-job path must read each box's own
    out/samples.extxyz (by manifest position) and combine them."""
    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(
            mock_remote_fs, mock_cluster_config, array_job=True, stub=False
        )
    )
    annealer.anneal_dir = tmp_path / "anneal"
    annealer.anneal_dir.mkdir()
    annealer._array_manifest = [{"index": 0}, {"index": 1}, {"index": 2}]

    array_root = tmp_path / "array_local"
    for i, symbol in enumerate(["Cu", "H", "O"]):
        box_out = array_root / str(i) / "out"
        box_out.mkdir(parents=True)
        atoms = Atoms(
            symbol, positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True
        )
        ase.io.write(box_out / "samples.extxyz", atoms)

    annealer.array_task = MagicMock()
    annealer.array_task.local_savepath = array_root

    output_path = annealer.get_sampled_configs()

    configs = ase.io.read(output_path, index=":")
    assert len(configs) == 3
    annealer.array_task.submit.assert_called_once()
    annealer.array_task.collect_results.assert_called_once()


def test_embedding_annealer_array_job_tolerates_partial_failure(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    """A box with no out/samples.extxyz (crashed anneal) is skipped, not
    fatal, as long as at least one other box succeeded."""
    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(
            mock_remote_fs, mock_cluster_config, array_job=True, stub=False
        )
    )
    annealer.anneal_dir = tmp_path / "anneal"
    annealer.anneal_dir.mkdir()
    annealer._array_manifest = [{"index": 0}, {"index": 1}]

    array_root = tmp_path / "array_local"
    box_out = array_root / "0" / "out"
    box_out.mkdir(parents=True)
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    ase.io.write(box_out / "samples.extxyz", atoms)
    # box 1: no output at all -- simulates a crashed/non-converged anneal.
    (array_root / "1" / "out").mkdir(parents=True)

    annealer.array_task = MagicMock()
    annealer.array_task.local_savepath = array_root

    output_path = annealer.get_sampled_configs()
    configs = ase.io.read(output_path, index=":")
    assert len(configs) == 1


def test_embedding_annealer_array_job_raises_if_all_boxes_fail(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    annealer = EmbeddingAnnealer(
        **_annealer_kwargs(
            mock_remote_fs, mock_cluster_config, array_job=True, stub=False
        )
    )
    annealer.anneal_dir = tmp_path / "anneal"
    annealer.anneal_dir.mkdir()
    annealer._array_manifest = [{"index": 0}]

    array_root = tmp_path / "array_local"
    (array_root / "0" / "out").mkdir(parents=True)

    annealer.array_task = MagicMock()
    annealer.array_task.local_savepath = array_root

    with pytest.raises(WorkflowError, match="none of the"):
        annealer.get_sampled_configs()


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
