"""
Tests for amorphous matrix embedding components.

Covers the ROI clustering/box-carving geometry, the three cleanup
strategies, interface-normal detection (fallback heuristic + pure mesh math;
the real OVITO path is exercised only when the ``ovito`` package is
importable), the two input schemas, the anneal deck template, and
``AmorphousEmbeddingSampler`` in stub mode.

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode sampler tests run with NO cluster, SSH, or LAMMPS -- only
remote_handler + ase + numpy on the host.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import jinja2
import numpy as np
import pytest

pytest.importorskip("remote_handler")
from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402
from remote_handler.exceptions import ScriptInputError  # noqa: E402

from lammps_al.embedding.inputs import (  # noqa: E402
    EmbeddingAnnealInput,
    MatrixEmbeddingInput,
)
from lammps_al.embedding.sampler import AmorphousEmbeddingSampler  # noqa: E402
from lammps_al.embedding.scripts.cleanup import (  # noqa: E402
    _template_topology,
    cleanup,
    cleanup_molecule,
    cleanup_none,
    cleanup_stoichiometry,
)
from lammps_al.embedding.scripts.geometry import (  # noqa: E402
    BoxDims,
    carve_box,
    find_rois,
)
from lammps_al.embedding.scripts.surface_mesh import (  # noqa: E402
    _fallback_normal,
    _normal_from_mesh,
    detect_normal,
)

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration

OVITO_AVAILABLE = importlib.util.find_spec("ovito") is not None

_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lammps_al"
    / "embedding"
    / "scripts"
    / "templates"
)


# ---------------------------------------------------------------------------
# geometry.py
# ---------------------------------------------------------------------------


def _tagged_atoms_two_clusters():
    """Cu box with two well-separated high-uncertainty clusters."""
    rng = np.random.default_rng(0)
    n = 100
    positions = rng.uniform(0, 20, (n, 3))
    atoms = Atoms(["Cu"] * n, positions=positions, cell=[20, 20, 20], pbc=True)
    unc = np.zeros(n)

    # cluster A: peak at index 11
    atoms.positions[10] = [5.0, 5.0, 5.0]
    atoms.positions[11] = [5.5, 5.0, 5.0]
    atoms.positions[12] = [5.0, 5.5, 5.0]
    unc[[10, 11, 12]] = [0.5, 0.9, 0.3]

    # cluster B: peak at index 50
    atoms.positions[50] = [15.0, 15.0, 15.0]
    atoms.positions[51] = [15.3, 15.0, 15.0]
    unc[[50, 51]] = [0.7, 0.6]

    atoms.arrays["uncertainty"] = unc
    return atoms


def test_find_rois_returns_one_peak_per_cluster():
    atoms = _tagged_atoms_two_clusters()
    rois = find_rois(atoms, threshold=0.2, cluster_cutoff=1.0)
    assert rois == [11, 50]


def test_find_rois_no_atoms_above_threshold():
    atoms = _tagged_atoms_two_clusters()
    assert find_rois(atoms, threshold=10.0, cluster_cutoff=1.0) == []


def test_carve_box_centers_and_sizes_correctly():
    atoms = _tagged_atoms_two_clusters()
    box = BoxDims(x=4, y=4, z_below=4, z_above=6)
    carved, roi_pos = carve_box(atoms, 11, box, buffer=1.0)

    expected_cell = np.array([9.0, 9.0, 11.0])
    assert np.allclose(carved.cell.array.diagonal(), expected_cell)

    d = np.linalg.norm(carved.get_positions() - roi_pos, axis=1)
    assert d.min() < 1e-8  # the ROI atom itself survives the carve

    # roi_pos deterministic: x/y centered, z offset by the asymmetric headspace
    assert np.allclose(roi_pos, [4.5, 4.5, 4.5])


def test_carve_box_rotates_normal_onto_z():
    # A Cu plane normal to +x with an H ROI just outside it. Rotating +x onto
    # +z must make every Cu atom have the same z coordinate.
    cu_positions = [
        [4.0, y, z] for y in (2.0, 4.0, 6.0, 8.0) for z in (2.0, 4.0, 6.0, 8.0)
    ]
    atoms = Atoms(
        ["Cu"] * len(cu_positions) + ["H"],
        positions=cu_positions + [[5.0, 5.0, 5.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )
    carved, roi_pos = carve_box(
        atoms,
        len(atoms) - 1,
        BoxDims(x=5, y=5, z_below=5),
        buffer=0.0,
        normal=np.array([1.0, 0.0, 0.0]),
    )
    symbols = np.array(carved.get_chemical_symbols())
    cu = carved.positions[symbols == "Cu"]
    assert np.ptp(cu[:, 2]) < 1e-10
    assert np.ptp(cu[:, 0]) > 5.0
    assert np.ptp(cu[:, 1]) > 5.0
    assert np.allclose(roi_pos, [5.0, 5.0, 5.0])


def test_box_dims_symmetric_default():
    box = BoxDims(x=1, y=2, z_below=3)
    assert box.z_above == 3


# ---------------------------------------------------------------------------
# cleanup.py
# ---------------------------------------------------------------------------


def test_cleanup_none_returns_unchanged_copy():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    out = cleanup_none(atoms)
    assert out is not atoms
    assert out.get_chemical_formula() == atoms.get_chemical_formula()


def test_template_topology_water_is_star_graph():
    water = molecule("H2O")
    topo = _template_topology(water)
    assert topo is not None
    assert topo["central_symbol"] == "O"
    assert topo["ligand_symbol"] == "H"
    assert topo["coordination"] == 2
    assert abs(topo["bond_angle"] - 104.5) < 3.0


def test_template_topology_rejects_non_star():
    # Two disconnected atoms: no central atom bonded to all others.
    dimer = Atoms("H2", positions=[[0, 0, 0], [20, 0, 0]])
    assert _template_topology(dimer) is None


def _broken_water_box():
    """Intact water far away, a broken (1-H) water, and a lone dangling H."""
    water = molecule("H2O")
    atoms = water.copy()
    atoms.translate([10, 10, 10])

    o_pos = np.array([2.0, 2.0, 2.0])
    h_pos = o_pos + np.array([0.96, 0.0, 0.0])
    atoms += Atoms("O", positions=[o_pos])
    atoms += Atoms("H", positions=[h_pos])
    atoms += Atoms("H", positions=[[15.0, 2.0, 2.0]])  # dangler

    atoms.set_cell([20, 20, 20])
    atoms.set_pbc(True)
    return atoms


def test_cleanup_molecule_repairs_broken_water_and_drops_danglers():
    atoms = _broken_water_box()
    water = molecule("H2O")
    rng = np.random.default_rng(1)

    repaired = cleanup_molecule(
        atoms,
        roi_center=np.array([0.0, 0.0, 0.0]),
        protect_radius=0.1,
        solvent=water,
        rng=rng,
    )

    symbols = np.array(repaired.get_chemical_symbols())
    o_indices = np.where(symbols == "O")[0]
    assert len(o_indices) == 2  # both waters kept (repaired, not dropped)

    for o_idx in o_indices:
        d = np.linalg.norm(
            repaired.get_positions() - repaired.get_positions()[o_idx], axis=1
        )
        n_h_neighbors = int(np.sum((symbols == "H") & (d < 1.3) & (d > 0.01)))
        assert n_h_neighbors == 2

    # the dangling H (no bonded O) must have been deleted
    assert len(repaired) == 6  # 2 intact waters worth of atoms, no dangler


def test_cleanup_molecule_respects_protect_radius():
    atoms = _broken_water_box()
    water = molecule("H2O")
    rng = np.random.default_rng(1)

    # Protect everything -- nothing should change.
    repaired = cleanup_molecule(
        atoms,
        roi_center=np.array([2.0, 2.0, 2.0]),
        protect_radius=1000.0,
        solvent=water,
        rng=rng,
    )
    assert len(repaired) == len(atoms)


def test_cleanup_stoichiometry_trims_excess():
    rng = np.random.default_rng(2)
    positions = rng.uniform(0, 10, (35, 3))
    symbols = ["H"] * 25 + ["O"] * 10
    atoms = Atoms(symbols, positions=positions, cell=[10, 10, 10], pbc=True)

    trimmed = cleanup_stoichiometry(
        atoms,
        roi_center=np.array([100, 100, 100]),
        protect_radius=0.1,
        stoichiometry={"H": 2, "O": 1},
        rng=rng,
    )
    counts = trimmed.get_chemical_symbols()
    assert counts.count("H") == 20
    assert counts.count("O") == 10


def test_cleanup_stoichiometry_never_touches_protected_atoms():
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    atoms = Atoms(
        ["H", "H", "H"], positions=positions, cell=[10, 10, 10], pbc=True
    )
    trimmed = cleanup_stoichiometry(
        atoms,
        roi_center=np.array([0.0, 0.0, 0.0]),
        protect_radius=5.0,
        stoichiometry={"H": 1},
        rng=np.random.default_rng(0),
    )
    assert len(trimmed) == 3  # all three within protect_radius, none removed


def test_cleanup_dispatcher_errors_on_missing_args():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        cleanup(atoms, "molecule", np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        cleanup(atoms, "stoichiometry", np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        cleanup(atoms, "not-a-mode", np.zeros(3), 1.0)


def test_cleanup_dispatcher_none_mode():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    out = cleanup(atoms, "none", np.zeros(3), 1.0)
    assert len(out) == len(atoms)


# ---------------------------------------------------------------------------
# surface_mesh.py
# ---------------------------------------------------------------------------


def test_fallback_normal_points_away_from_phase():
    # Cu atoms all on the +z side of the ROI -> the outward normal points -z
    # (away from the Cu phase, toward the exposed side).
    positions = np.array([[0, 0, 2], [1, 0, 3], [-1, 0, 3], [0, 0, 5]])
    atoms = Atoms(["Cu", "Cu", "Cu", "O"], positions=positions)
    roi = np.array([0.0, 0.0, 0.0])
    normal = _fallback_normal(atoms, roi, phase_species=["Cu"])
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert normal[2] < 0


def test_fallback_normal_raises_on_symmetric_degenerate_case():
    # Cu atoms symmetric around the ROI -> vector sum cancels -> degenerate.
    positions = np.array([[1, 0, 0], [-1, 0, 0]])
    atoms = Atoms(["Cu", "Cu"], positions=positions)
    with pytest.raises(ValueError):
        _fallback_normal(atoms, np.zeros(3), phase_species=["Cu"])


def test_fallback_normal_raises_when_no_matching_species():
    atoms = Atoms(["Cu"], positions=[[0, 0, 0]])
    with pytest.raises(ValueError):
        _fallback_normal(atoms, np.zeros(3), phase_species=["Xx"])


def test_fallback_normal_uses_minimum_image_convention():
    """A phase atom near the *opposite* edge of a periodic cell must be
    treated as nearby (via its periodic image), not as a huge raw-distance
    outlier that swamps the true local signal -- otherwise a ROI sitting
    near a cell edge gets a direction dominated by box geometry rather than
    the actual nearby condensed phase (this is exactly what produced wildly
    inconsistent normals for ROIs scattered across a large periodic Cu
    slab)."""
    cell = [30.0, 30.0, 30.0]
    # Three Cu atoms directly "below" the ROI (true local signal: -z).
    near = [[0.1, 15.0, 10.0]] * 3
    # One Cu atom at the same z as the ROI, but across the periodic
    # boundary in x -- its minimum-image offset is tiny (~-0.2 in x), but
    # its *raw* offset is ~29.8 in x, large enough to corrupt the naive sum.
    far = [[29.9, 15.0, 20.0]]
    atoms = Atoms(
        ["Cu"] * 4,
        positions=near + far,
        cell=cell,
        pbc=True,
    )
    roi = np.array([0.1, 15.0, 20.0])
    normal = _fallback_normal(atoms, roi, phase_species=["Cu"])
    # Correct answer: Cu is really only below the ROI -> outward is +z,
    # negligible x/y. (Without minimum-image correction this comes out as
    # a corrupted ~45-degree mix of +x/-z instead.)
    assert normal[2] == pytest.approx(1.0, abs=0.02)
    assert abs(normal[0]) < 0.02
    assert abs(normal[1]) < 1e-8


def test_normal_from_mesh_picks_nearest_triangle():
    # A simple two-triangle "roof": one facing +z, one facing +x, offset.
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],  # triangle 0: in xy-plane, normal +/-z
            [5, 0, 0],
            [5, 1, 0],
            [5, 0, 1],  # triangle 1: constant-x plane at x=5, normal +/-x
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])

    roi_0 = np.array([0.2, 0.2, 0.0])
    normal_near_0 = _normal_from_mesh(vertices, faces, roi_center=roi_0)
    assert abs(normal_near_0[2]) == pytest.approx(1.0, abs=1e-6)

    roi_1 = np.array([5.0, 0.2, 0.2])
    normal_near_1 = _normal_from_mesh(vertices, faces, roi_center=roi_1)
    assert abs(normal_near_1[0]) == pytest.approx(1.0, abs=1e-6)


def test_normal_from_mesh_raises_on_empty_faces():
    with pytest.raises(ValueError):
        _normal_from_mesh(
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros(3)
        )


def _rough_patch_mesh():
    """Two flat faces connected to one tiny, tilted anchor face."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0, 0.02],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],  # flat, normal +z, twice-area 1.0
            [1, 3, 2],  # flat, normal +z, twice-area 1.0
            [0, 4, 1],  # tilted, normal +y, twice-area 0.02
        ]
    )
    roi_center = np.array([0.5, 0.0, 0.01])
    return vertices, faces, roi_center


def test_normal_from_mesh_default_still_picks_nearest_triangle():
    """patch_radius=None uses the exact closest surface facet."""
    vertices, faces, roi_center = _rough_patch_mesh()
    normal = _normal_from_mesh(vertices, faces, roi_center)
    assert normal == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_normal_from_mesh_patch_radius_averages_out_local_roughness():
    """Connected facets are averaged in proportion to their true area."""
    vertices, faces, roi_center = _rough_patch_mesh()
    normal = _normal_from_mesh(vertices, faces, roi_center, patch_radius=1.0)
    assert normal[2] == pytest.approx(1.0, abs=1e-3)
    assert abs(normal[1]) < 0.02


def test_normal_from_mesh_patch_radius_too_small_falls_back_to_nearest():
    """A radius that reaches no neighbor still returns the anchor facet."""
    vertices, faces, roi_center = _rough_patch_mesh()
    normal = _normal_from_mesh(vertices, faces, roi_center, patch_radius=1e-6)
    assert normal == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_normal_from_mesh_uses_point_to_triangle_not_centroid_distance():
    # The large horizontal facet contains the closest surface point even
    # though its centroid is far away. A centroid search would incorrectly
    # choose the compact vertical facet near the ROI.
    vertices = np.array(
        [
            [0, 0, 0],
            [100, 0, 0],
            [0, 100, 0],
            [0.8, 0, 0],
            [0.8, 1, 0],
            [0.8, 0, 1],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    roi = np.array([0.1, 0.1, 0.2])
    normal = _normal_from_mesh(vertices, faces, roi)
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


def test_normal_from_mesh_excludes_disconnected_nearby_surface():
    # A large vertical sheet lies within the Euclidean patch radius but does
    # not share an edge with the horizontal anchor surface. It must not enter
    # the connected/geodesic average.
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.3, -2, -2],
            [0.3, 2, -2],
            [0.3, -2, 2],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    roi = np.array([0.1, 0.1, 0.1])
    normal = _normal_from_mesh(vertices, faces, roi, patch_radius=10.0)
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


def test_normal_from_mesh_unwraps_periodic_triangle_edges():
    # This +z triangle crosses x=0. Stored raw, its 9.8 -> 0.2 edge looks
    # -9.6 A long and reverses the apparent normal; its true MIC edge is
    # +0.4 A. The ROI lies directly on the correctly unwrapped facet.
    vertices = np.array([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0], [9.8, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    roi = np.array([9.9, 0.2, 0.0])
    normal = _normal_from_mesh(
        vertices,
        faces,
        roi,
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=np.array([True, True, True]),
    )
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


def test_detect_normal_falls_back_without_ovito_atoms(monkeypatch):
    """When OVITO mesh construction fails, detect_normal degrades to the
    species-COM heuristic rather than raising."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    atoms = Atoms(["Cu", "Cu", "Cu"], positions=positions)
    roi = np.array([0.0, 0.0, 5.0])
    # remove_species that deletes the whole structure -> forces the internal
    # ValueError -> exercises the fallback path regardless of whether ovito
    # is actually installed in this environment.
    normal = detect_normal(
        atoms, roi, remove_species=["Cu"], phase_species=["Cu"]
    )
    assert np.linalg.norm(normal) == pytest.approx(1.0)


def test_detect_normal_flips_inward_mesh_normal_outward(monkeypatch):
    """The mesh path is forced to the same outward convention as the fallback
    (away from the phase), regardless of OVITO's facet winding (finding #2)."""
    import lammps_al.embedding.scripts.surface_mesh as sm

    # Cu phase on the +z side of the ROI. A mesh normal pointing +z (toward
    # the phase) must be flipped to -z (outward).
    positions = np.array([[0, 0, 2], [1, 0, 3], [-1, 0, 3]])
    atoms = Atoms(["Cu", "Cu", "Cu"], positions=positions)
    roi = np.array([0.0, 0.0, 0.0])
    monkeypatch.setattr(
        sm, "_ovito_normal", lambda *a, **k: np.array([0.0, 0.0, 1.0])
    )
    normal = detect_normal(atoms, roi, phase_species=["Cu"])
    assert normal[2] < 0


def test_detect_normal_keeps_outward_mesh_normal(monkeypatch):
    """An already-outward mesh normal is left untouched."""
    import lammps_al.embedding.scripts.surface_mesh as sm

    positions = np.array([[0, 0, 2], [1, 0, 3], [-1, 0, 3]])
    atoms = Atoms(["Cu", "Cu", "Cu"], positions=positions)
    roi = np.array([0.0, 0.0, 0.0])
    monkeypatch.setattr(
        sm, "_ovito_normal", lambda *a, **k: np.array([0.0, 0.0, -1.0])
    )
    normal = detect_normal(atoms, roi, phase_species=["Cu"])
    assert normal[2] < 0


def test_detect_normal_forwards_patch_radius(monkeypatch):
    """detect_normal must pass patch_radius through to _ovito_normal (and
    therefore _normal_from_mesh) rather than silently dropping it."""
    import lammps_al.embedding.scripts.surface_mesh as sm

    positions = np.array([[0, 0, 2], [1, 0, 3], [-1, 0, 3]])
    atoms = Atoms(["Cu", "Cu", "Cu"], positions=positions)
    roi = np.array([0.0, 0.0, 0.0])

    captured = {}

    def fake_ovito_normal(
        atoms,
        roi_center,
        remove_species,
        smoothing_level,
        radius,
        patch_radius=None,
    ):
        captured["patch_radius"] = patch_radius
        return np.array([0.0, 0.0, -1.0])

    monkeypatch.setattr(sm, "_ovito_normal", fake_ovito_normal)
    detect_normal(atoms, roi, phase_species=["Cu"], patch_radius=4.0)
    assert captured["patch_radius"] == 4.0


@pytest.mark.skipif(not OVITO_AVAILABLE, reason="needs the 'ovito' extra")
def test_detect_normal_surface_mode_with_real_ovito():
    from ase.build import bulk, surface

    cu = bulk("Cu", "fcc", a=3.6)
    slab = surface(cu, (1, 1, 1), 8, vacuum=0)
    slab = slab.repeat((6, 6, 1))
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    zmax = slab.positions[:, 2].max()
    cell = slab.cell.array.copy()
    cell[2, :] = [0, 0, zmax + 15.0]
    slab.set_cell(cell, scale_atoms=False)
    slab.set_pbc((True, True, True))

    roi = np.array([slab.cell[0, 0] / 2, slab.cell[1, 1] / 2, zmax])
    normal = detect_normal(slab, roi, radius=2.0)
    assert normal[2] == pytest.approx(1.0, abs=2e-3)


# ---------------------------------------------------------------------------
# input_models.py
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


# ---------------------------------------------------------------------------
# embedding_anneal.in.j2 (rendering)
# ---------------------------------------------------------------------------


def _rendered_stage(**overrides):
    """A pre-expanded stage dict as embedding_anneal.py builds it (segments
    already zipped from temperature waypoints) -- what the template
    actually consumes."""
    stage = dict(
        index=0,
        type_list="1",
        temp_start=300,
        segments=[(300, 600, 50000), (600, 300, 100000)],
        pressure_xy=1e4,
        pressure_z=None,
        z_wall_margin=1.0,
        z_wall_epsilon=10.0,
        z_wall_cutoff=1.0,
    )
    stage.update(overrides)
    return stage


def _render_anneal_template(**overrides):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(_TEMPLATE_DIR))
    template = env.get_template("embedding_anneal.in.j2")
    common = dict(
        data_file="embedded.data",
        pair_style="grace",
        pair_coeff_path="/path/to/model",
        species=["Cu", "H", "O"],
        mass_lines="mass 1 63.546\nmass 2 1.008\nmass 3 15.999",
        cutoff=4.0,
        roi_x=6.5,
        roi_y=6.5,
        roi_z=6.5,
        timestep=0.0005,
        seed=42,
        dump_every=1000,
        dump_file="anneal.dump",
        final_data_file="embedded_out.data",
        stages=[_rendered_stage()],
        # cutoff - roi_repel_margin, matching what embedding_anneal.py
        # computes -- kept as a literal here since the template only
        # consumes the already-subtracted radius. Deck-wide (not
        # per-stage): z_wall_margin/epsilon/cutoff are resolved per-stage
        # instead (see _rendered_stage's defaults).
        roi_repel_radius=3.5,
        roi_repel_epsilon=10.0,
        roi_repel_wall_cutoff=1.0,
    )
    common.update(overrides)
    return template.render(**common)


def test_anneal_template_renders_without_leftover_jinja():
    stage_lists = [
        [_rendered_stage()],
        [_rendered_stage(pressure_xy=None, pressure_z=1.0)],
        [
            _rendered_stage(
                index=0, type_list="1", pressure_xy=1e4, pressure_z=None
            ),
            _rendered_stage(
                index=1,
                type_list="2",
                temp_start=300,
                segments=[(300, 300, 50000)],
                pressure_xy=None,
                pressure_z=1.0,
            ),
        ],
        [_rendered_stage(pressure_xy=None, pressure_z=None)],  # NVT fallback
        [
            _rendered_stage(index=0, type_list="1"),
            _rendered_stage(index=1, type_list="2"),
            _rendered_stage(index=2, type_list="3"),
        ],
        # arbitrary (4-segment) ramp within a single stage.
        [
            _rendered_stage(
                segments=[
                    (300, 500, 10000),
                    (500, 700, 10000),
                    (700, 500, 10000),
                    (500, 300, 10000),
                ]
            )
        ],
    ]
    for stages in stage_lists:
        out = _render_anneal_template(stages=stages)
        assert "{{" not in out
        assert "{%" not in out


def test_anneal_template_arbitrary_segment_count_all_present():
    """Every segment in a stage's list must produce its own run block, not
    just the first/last (this is exactly what was missing before -- a
    stage was hardcoded to exactly two segments)."""
    out = _render_anneal_template(
        stages=[
            _rendered_stage(
                segments=[
                    (300, 500, 11111),
                    (500, 700, 22222),
                    (700, 500, 33333),
                    (500, 300, 44444),
                ]
            )
        ]
    )
    for steps in (11111, 22222, 33333, 44444):
        assert f"run {steps}" in out
    assert out.count("fix 1 phase0_move npt") == 4
    assert out.count("unfix 1") == 4


def test_anneal_template_two_stage_freezes_the_other_phase():
    out = _render_anneal_template(
        stages=[
            _rendered_stage(
                index=0, type_list="1", pressure_xy=1e4, pressure_z=None
            ),
            _rendered_stage(
                index=1,
                type_list="2",
                temp_start=300,
                segments=[(300, 300, 50000)],
                pressure_xy=None,
                pressure_z=1.0,
            ),
        ]
    )
    assert "fix freeze_1_during_0 phase1 setforce" in out
    assert "fix freeze_0_during_1 phase0 setforce" in out
    assert "group phase1_move subtract phase1 fixed" in out
    assert "unfix freeze_1_during_0" in out
    assert "unfix freeze_0_during_1" in out


def test_anneal_template_single_stage_has_no_freeze_of_others():
    out = _render_anneal_template(stages=[_rendered_stage()])
    assert "freeze_" not in out
    assert "group moving union phase0_move" in out


def test_anneal_template_three_stages_each_freeze_the_other_two():
    out = _render_anneal_template(
        stages=[
            _rendered_stage(index=0, type_list="1"),
            _rendered_stage(index=1, type_list="2"),
            _rendered_stage(index=2, type_list="3"),
        ]
    )
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            assert f"fix freeze_{j}_during_{i} phase{j} setforce" in out


def test_anneal_template_pressure_axes_selected_per_stage():
    # xy-only stage: an x/y-coupled npt line, no bare "z " pressure coupling.
    out_xy = _render_anneal_template(
        stages=[_rendered_stage(pressure_xy=1e4, pressure_z=None)]
    )
    assert "x 10000.0 10000.0" in out_xy
    assert "z " not in out_xy.split("dilate moving")[0].split("\n")[-1]

    # z-only stage.
    out_z = _render_anneal_template(
        stages=[_rendered_stage(pressure_xy=None, pressure_z=1.0)]
    )
    assert "z 1.0 1.0" in out_z
    assert " x " not in out_z.split("dilate moving")[0].split("\n")[-1]

    # both axes -> full npt.
    out_both = _render_anneal_template(
        stages=[_rendered_stage(pressure_xy=1e4, pressure_z=1.0)]
    )
    assert "x 10000.0 10000.0" in out_both
    assert "z 1.0 1.0" in out_both

    # neither axis -> nvt, no barostat.
    out_none = _render_anneal_template(
        stages=[_rendered_stage(pressure_xy=None, pressure_z=None)]
    )
    assert "fix 1 phase0_move nvt" in out_none
    assert "npt" not in out_none


def test_anneal_template_fixsphere_radius_matches_cutoff():
    out = _render_anneal_template(cutoff=7.5, roi_x=1.0, roi_y=2.0, roi_z=3.0)
    assert "region fixSphere sphere 1.0 2.0 3.0 7.5" in out


def test_anneal_template_roi_repel_wall_present_once():
    """The ROI repulsive wall is set up once, around the whole `moving`
    group, not per-stage -- it must protect the frozen core throughout the
    entire anneal, whichever phase is currently mobile, uniformly for every
    species (unlike the z-wall, which is legitimately tunable per stage)."""
    out = _render_anneal_template(
        roi_x=1.0,
        roi_y=2.0,
        roi_z=3.0,
        roi_repel_radius=3.5,
        roi_repel_epsilon=12.0,
        roi_repel_wall_cutoff=0.8,
        stages=[
            _rendered_stage(index=0, type_list="1"),
            _rendered_stage(index=1, type_list="2"),
        ],
    )
    assert "region repSphere sphere 1.0 2.0 3.0 3.5 side out" in out
    assert out.count("fix repel moving wall/region repSphere harmonic") == 1
    assert "wall/region repSphere harmonic 12.0 0.0 0.8" in out
    # Defined (and the wall created) before any stage's mobile group can
    # possibly have moved, and never unfixed -- active for the whole run.
    assert out.index("fix repel") < out.index("fix 1 phase0_move")
    assert "unfix repel" not in out


def test_anneal_template_z_wall_skipped_when_pressure_z_set():
    """A stage that already barostats z must not also get a static z-wall
    -- the two would fight each other over the box's z rescaling."""
    out = _render_anneal_template(
        stages=[_rendered_stage(pressure_xy=None, pressure_z=1.0)]
    )
    assert "wall/harmonic zhi" not in out
    assert "wall/harmonic zlo" not in out


def test_anneal_template_z_wall_present_when_pressure_z_unset():
    out = _render_anneal_template(
        stages=[
            _rendered_stage(
                pressure_xy=1e4,
                pressure_z=None,
                z_wall_margin=2.0,
                z_wall_epsilon=15.0,
                z_wall_cutoff=0.5,
            )
        ],
    )
    assert "compute maxZ0 phase0_move reduce max z" in out
    assert "compute minZ0 phase0_move reduce min z" in out
    assert 'variable zmax0 equal "c_maxZ0+2.0"' in out
    assert 'variable zmin0 equal "c_minZ0-2.0"' in out
    assert (
        "fix zwall_hi_0 phase0_move wall/harmonic zhi ${zmax0} 15.0 0.0 0.5 "
        "pbc yes" in out
    )
    assert (
        "fix zwall_lo_0 phase0_move wall/harmonic zlo ${zmin0} 15.0 0.0 0.5 "
        "pbc yes" in out
    )
    assert "unfix zwall_hi_0" in out
    assert "unfix zwall_lo_0" in out


def test_anneal_template_z_wall_per_stage_mixed_pressure_z():
    """Two stages, only one barostats z -- only the other gets a z-wall."""
    out = _render_anneal_template(
        stages=[
            _rendered_stage(
                index=0, type_list="1", pressure_xy=1e4, pressure_z=None
            ),
            _rendered_stage(
                index=1,
                type_list="2",
                temp_start=300,
                segments=[(300, 300, 50000)],
                pressure_xy=None,
                pressure_z=1.0,
            ),
        ]
    )
    assert "zwall_hi_0" in out and "zwall_lo_0" in out
    assert "zwall_hi_1" not in out and "zwall_lo_1" not in out


def test_anneal_template_z_wall_strength_differs_per_stage():
    """Two stages that both qualify for a z-wall (neither barostats z) can
    each be tuned independently -- e.g. a stiffer wall for a rigid metal
    lattice than for a mobile solvent stage."""
    out = _render_anneal_template(
        stages=[
            _rendered_stage(
                index=0,
                type_list="1",
                pressure_xy=1e4,
                pressure_z=None,
                z_wall_margin=1.0,
                z_wall_epsilon=20.0,
                z_wall_cutoff=1.0,
            ),
            _rendered_stage(
                index=1,
                type_list="2",
                temp_start=300,
                segments=[(300, 300, 50000)],
                pressure_xy=None,
                pressure_z=None,
                z_wall_margin=3.0,
                z_wall_epsilon=2.0,
                z_wall_cutoff=1.5,
            ),
        ]
    )
    assert (
        "fix zwall_hi_0 phase0_move wall/harmonic zhi ${zmax0} 20.0 0.0 1.0"
        in out
    )
    assert (
        "fix zwall_hi_1 phase1_move wall/harmonic zhi ${zmax1} 2.0 0.0 1.5"
        in out
    )
    assert 'variable zmax0 equal "c_maxZ0+1.0"' in out
    assert 'variable zmax1 equal "c_maxZ1+3.0"' in out


def test_anneal_template_z_wall_skipped_when_stage_epsilon_zero():
    """A stage can opt out of the z-wall entirely (epsilon=0) even though
    it has no pressure_z and would otherwise qualify."""
    out = _render_anneal_template(
        stages=[
            _rendered_stage(
                pressure_xy=1e4,
                pressure_z=None,
                z_wall_epsilon=0.0,
            )
        ]
    )
    assert "wall/harmonic zhi" not in out
    assert "wall/harmonic zlo" not in out
    assert "compute maxZ0" not in out


# ---------------------------------------------------------------------------
# AmorphousEmbeddingSampler (stub mode)
# ---------------------------------------------------------------------------


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


def test_amorphous_embedding_sampler_stub_mode(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    sampler = AmorphousEmbeddingSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        trajectory_path=tmp_path / "traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
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
        stub=True,
    )

    sampler.update(tmp_path)
    output_path = sampler.get_sampled_configs()

    assert output_path.exists()
    import ase.io

    configs = ase.io.read(output_path, index=":")
    assert len(configs) == 3
    for atoms in configs:
        assert atoms.info["topology_mode"] == "bulk"


def _minimal_sampler_kwargs(
    tmp_path, mock_remote_fs, mock_cluster_config, **overrides
):
    kwargs = dict(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        trajectory_path=tmp_path / "traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
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
        stub=True,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "topology_mode, expected_epsilon",
    [
        ("bulk", 0.0),
        ("surface", 10.0),
        ("interface", 10.0),
    ],
)
def test_amorphous_embedding_sampler_z_wall_epsilon_auto_from_topology(
    tmp_path,
    mock_remote_fs,
    mock_cluster_config,
    topology_mode,
    expected_epsilon,
):
    """z_wall_epsilon left unset auto-decides from topology_mode: a "bulk"
    box is fully periodic in z (nothing to drift into, so the wall would be
    inert overhead), while "surface"/"interface" boxes have a free surface
    where mobile atoms could otherwise drift into vacuum/solvent
    headspace."""
    sampler = AmorphousEmbeddingSampler(
        **_minimal_sampler_kwargs(
            tmp_path,
            mock_remote_fs,
            mock_cluster_config,
            topology_mode=topology_mode,
        )
    )
    assert sampler.z_wall_epsilon == expected_epsilon


def test_amorphous_embedding_sampler_z_wall_epsilon_explicit_overrides_auto(
    tmp_path,
    mock_remote_fs,
    mock_cluster_config,
):
    """An explicit z_wall_epsilon always wins, regardless of topology_mode."""
    sampler = AmorphousEmbeddingSampler(
        **_minimal_sampler_kwargs(
            tmp_path,
            mock_remote_fs,
            mock_cluster_config,
            topology_mode="bulk",
            z_wall_epsilon=7.0,
        )
    )
    assert sampler.z_wall_epsilon == 7.0


def test_amorphous_embedding_sampler_defaults_specorder_to_species(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """The extract input must pin ``specorder`` to the annealed ``species``
    order so per-box type numbering stays consistent with the anneal deck's
    mass/pair_coeff/Z_of_type lines (finding #1)."""
    monkeypatch.setattr(
        "lammps_al.embedding.sampler.SingleScriptTask", MagicMock()
    )
    species = ["Cu", "H", "O"]
    sampler = AmorphousEmbeddingSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        trajectory_path=tmp_path / "traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
        lammps_path="/opt/lammps/lmp",
        pair_style="grace",
        pair_coeff_path="/opt/models/model",
        species=species,
        masses=[63.5, 1.0, 16.0],
        anneal_stages=[
            {
                "species": ["Cu"],
                "temperatures": [300.0, 600.0, 300.0],
                "run_steps": [50000, 100000],
                "pressure_xy": 1e4,
            },
        ],
        stub=False,
    )

    sampler.update(tmp_path)

    written = json.loads(sampler.extract_input_file.read_text())
    assert written["specorder"] == species


def test_amorphous_embedding_sampler_writes_normal_patch_radius(
    tmp_path, mock_remote_fs, mock_cluster_config, monkeypatch
):
    """normal_patch_radius must reach the extract input JSON so extract.py
    can default it to protect_radius on the remote side."""
    monkeypatch.setattr(
        "lammps_al.embedding.sampler.SingleScriptTask", MagicMock()
    )
    sampler = AmorphousEmbeddingSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        trajectory_path=tmp_path / "traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
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
        normal_patch_radius=6.0,
        stub=False,
    )

    sampler.update(tmp_path)

    written = json.loads(sampler.extract_input_file.read_text())
    assert written["normal_patch_radius"] == 6.0


def test_amorphous_embedding_sampler_molecule_requires_valid_solvent_type(
    tmp_path, mock_remote_fs, mock_cluster_config
):
    sampler = AmorphousEmbeddingSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        trajectory_path=tmp_path / "traj.xyz",
        threshold=0.1,
        cluster_cutoff=3.0,
        box_x=8,
        box_y=8,
        box_z_below=8,
        protect_radius=4.0,
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
        cleanup_mode="molecule",
        solvent=12345,  # not an Atoms or str
        stub=False,
    )
    from remote_handler.exceptions import WorkflowError

    with pytest.raises(WorkflowError):
        sampler.update(tmp_path)
