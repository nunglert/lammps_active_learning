"""Interface/surface-normal detection via OVITO's surface-mesh construction.

Lazy-imports OVITO so the host and any non-OVITO remote environment can still
import :mod:`lammps_al.embedding.scripts` without the ``embedding``
extra installed -- only :func:`detect_normal` (and, transitively,
:func:`_ovito_normal`) touches ``ovito``.

Mesh introspection uses ``SurfaceMesh.get_vertices()``/``get_face_vertices()``
(OVITO >= 3.7.9), isolated in :func:`_extract_mesh_arrays` in case that API
shifts across releases -- everything else, including the actual "nearest
triangle to the ROI" math in :func:`_normal_from_mesh`, is plain numpy and
independently testable with synthetic mesh arrays.
"""

from __future__ import annotations

import heapq
import logging
from typing import Optional

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

logger = logging.getLogger(__name__)


def _phase_direction(
    atoms: Atoms,
    roi_center: np.ndarray,
    phase_species: Optional[list[str]],
) -> Optional[np.ndarray]:
    """Unnormalized vector from ``roi_center`` toward the condensed-phase COM.

    Sums the offsets of the phase atoms (``phase_species``, or all atoms when
    ``None``) relative to the ROI. Offsets are minimum-image-corrected
    (:func:`ase.geometry.find_mic`) against ``atoms.cell``/``atoms.pbc``
    before summing -- for a periodic system spanning the full cell (e.g. a
    slab that's periodic in-plane), the *raw* offset to every phase atom
    sums to something dominated by wherever the ROI happens to sit relative
    to the (arbitrary) periodic cell origin, not by the true local
    direction toward the condensed phase; a ROI near a cell edge would get
    a vector pointing mostly *across* the cell rather than through the
    phase. ``find_mic`` is a no-op wherever ``pbc`` is unset, so this is
    unchanged for a finite/non-periodic ``atoms``.

    Returns ``None`` if no atom matches or the configuration is symmetric
    (the sum cancels to ~0), so callers can decide whether that is an error
    (fallback estimate) or simply "no sign hint" (orienting the mesh
    normal).
    """
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    mask = (
        np.isin(symbols, list(phase_species))
        if phase_species
        else np.ones(len(atoms), dtype=bool)
    )
    if not mask.any():
        return None
    diffs = positions[mask] - np.asarray(roi_center)
    diffs, _ = find_mic(diffs, atoms.cell, atoms.get_pbc())
    vec = diffs.sum(axis=0)
    if np.linalg.norm(vec) < 1e-8:
        return None
    return vec


def _fallback_normal(
    atoms: Atoms,
    roi_center: np.ndarray,
    phase_species: Optional[list[str]],
) -> np.ndarray:
    """Species center-of-mass heuristic (the original ``copper_vec`` logic).

    Used when OVITO is unavailable or mesh construction fails. Cruder than
    the surface-mesh normal, but degrades gracefully. Points *away* from the
    condensed phase (outward), matching the OVITO surface-mesh convention --
    see :func:`detect_normal`.
    """
    vec = _phase_direction(atoms, roi_center, phase_species)
    if vec is None:
        raise ValueError(
            "Cannot estimate interface normal: no matching phase atoms, or a "
            "symmetric/degenerate configuration around the ROI"
        )
    outward = -vec
    return outward / np.linalg.norm(outward)


def _closest_points_on_triangles(
    triangles: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest point and squared distance for every triangle.

    The plane projection is used when it lies inside the triangle. Outside
    the triangle (and for degenerate triangles), the closest of the three
    edge-segment points is used instead. This is the actual point-to-triangle
    distance, unlike a triangle-centroid distance, which can select the wrong
    facet when mesh triangles have different sizes or aspect ratios.
    """
    triangles = np.asarray(triangles, dtype=float)
    point = np.asarray(point, dtype=float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab = b - a
    ac = c - a

    normal = np.cross(ab, ac)
    normal_sq = np.einsum("ij,ij->i", normal, normal)
    safe_normal_sq = np.where(normal_sq > 1e-24, normal_sq, 1.0)
    signed_scale = np.einsum("ij,ij->i", point - a, normal) / safe_normal_sq
    projected = point - signed_scale[:, None] * normal

    ap = projected - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denom = d00 * d11 - d01 * d01
    safe_denom = np.where(np.abs(denom) > 1e-24, denom, 1.0)
    bary_b = (d11 * d20 - d01 * d21) / safe_denom
    bary_c = (d00 * d21 - d01 * d20) / safe_denom
    bary_a = 1.0 - bary_b - bary_c
    inside = (
        (normal_sq > 1e-24)
        & (np.abs(denom) > 1e-24)
        & (bary_a >= -1e-12)
        & (bary_b >= -1e-12)
        & (bary_c >= -1e-12)
    )

    def closest_on_segments(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        edge = end - start
        edge_sq = np.einsum("ij,ij->i", edge, edge)
        numerator = np.einsum("ij,ij->i", point - start, edge)
        parameter = np.divide(
            numerator,
            edge_sq,
            out=np.zeros_like(numerator),
            where=edge_sq > 1e-24,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        return start + parameter[:, None] * edge

    edge_points = np.stack(
        [
            closest_on_segments(a, b),
            closest_on_segments(b, c),
            closest_on_segments(c, a),
        ],
        axis=1,
    )
    edge_dist_sq = np.sum((edge_points - point) ** 2, axis=2)
    closest_edge = edge_points[
        np.arange(len(triangles)), np.argmin(edge_dist_sq, axis=1)
    ]

    closest = closest_edge
    closest[inside] = projected[inside]
    distance_sq = np.sum((closest - point) ** 2, axis=1)
    return closest, distance_sq


def _face_adjacency(faces: np.ndarray) -> list[set[int]]:
    """Build triangle adjacency from shared mesh edges."""
    adjacency = [set() for _ in range(len(faces))]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            neighbors = edge_faces.setdefault(edge, [])
            for neighbor in neighbors:
                adjacency[face_index].add(neighbor)
                adjacency[neighbor].add(face_index)
            neighbors.append(face_index)
    return adjacency


def _connected_face_patch(
    faces: np.ndarray,
    centers: np.ndarray,
    anchor_face: int,
    patch_radius: Optional[float],
    cell: Optional[np.ndarray],
    pbc: Optional[np.ndarray],
) -> np.ndarray:
    """Faces reachable from ``anchor_face`` within a geodesic radius.

    Geodesic distance is approximated by shortest paths between centroids of
    edge-adjacent triangles. This deliberately cannot jump to a different
    surface sheet, pit wall, or cavity merely because it is Euclidean-close
    to the ROI.
    """
    selected = np.zeros(len(faces), dtype=bool)
    selected[anchor_face] = True
    if patch_radius is None or patch_radius <= 0:
        return selected

    adjacency = _face_adjacency(faces)
    distances = np.full(len(faces), np.inf)
    distances[anchor_face] = 0.0
    queue: list[tuple[float, int]] = [(0.0, anchor_face)]

    while queue:
        distance, face_index = heapq.heappop(queue)
        if distance > distances[face_index] or distance > patch_radius:
            continue
        selected[face_index] = True
        for neighbor in adjacency[face_index]:
            offset = centers[neighbor] - centers[face_index]
            if cell is not None and pbc is not None and np.any(pbc):
                offset, _ = find_mic(offset, cell, pbc)
            candidate = distance + float(np.linalg.norm(offset))
            if candidate <= patch_radius and candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return selected


def _normal_from_mesh(
    vertex_positions: np.ndarray,
    faces: np.ndarray,
    roi_center: np.ndarray,
    patch_radius: Optional[float] = None,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Area-weighted normal of the connected surface nearest ``roi_center``.

    The anchor facet is selected by exact point-to-triangle distance. From
    there, only edge-connected facets within ``patch_radius`` geodesic
    distance are included. This prevents a spatially nearby but disconnected
    surface sheet, cavity, or overhang from contaminating the average. Faces
    are already consistently wound on a connected OVITO surface mesh, so
    summing their raw (un-normalized) cross products -- whose magnitude is
    ``2 * area`` -- gives the requested area-weighted average.

    ``patch_radius=None`` uses only the closest facet, as does a radius
    smaller than the distance to the first edge-connected neighbor.

    Args:
        vertex_positions: ``(n_vertices, 3)`` mesh vertex positions.
        faces: ``(n_faces, 3)`` triangle vertex-index triples.
        roi_center: Point to search near.
        patch_radius: Approximate geodesic radius along edge-connected mesh
            facets. ``None`` uses only the closest facet.
        cell: Simulation cell, for minimum-image-corrected closest-point and
            adjacent-face distances. ``None`` skips periodic correction.
        pbc: Periodicity per axis, paired with ``cell``.

    Returns:
        Unit normal, area-weighted-averaged over the local patch.
    """
    if len(faces) == 0:
        raise ValueError("Mesh has no faces")
    faces = np.asarray(faces, dtype=int)
    tri = np.asarray(vertex_positions, dtype=float)[faces]
    roi_center = np.asarray(roi_center, dtype=float)
    if cell is not None and pbc is not None and np.any(pbc):
        # OVITO mesh faces that cross a periodic boundary may reference
        # vertices stored on opposite sides of the primary cell. First put
        # vertex 0 in its image nearest the ROI, then minimum-image unwrap
        # both triangle edges relative to it. Merely translating the complete
        # raw triangle would leave those edges spuriously cell-sized, which
        # corrupts its closest point, centroid, area, and normal.
        first_offset, _ = find_mic(tri[:, 0] - roi_center, cell, pbc)
        first = roi_center + first_offset
        edge_01, _ = find_mic(tri[:, 1] - tri[:, 0], cell, pbc)
        edge_02, _ = find_mic(tri[:, 2] - tri[:, 0], cell, pbc)
        tri = np.stack([first, first + edge_01, first + edge_02], axis=1)
    centers = tri.mean(axis=1)

    closest_points, distance_sq = _closest_points_on_triangles(tri, roi_center)
    anchor_face = int(np.argmin(distance_sq))
    mask = _connected_face_patch(
        faces, centers, anchor_face, patch_radius, cell, pbc
    )

    # Un-normalized per-face normals; magnitude = 2 * triangle area, so
    # summing them directly is an area-weighted average.
    raw_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    normal = raw_normals[mask].sum(axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError(
            "Degenerate or mutually-cancelling mesh faces near ROI (zero "
            "net normal)"
        )
    normal = normal / norm

    # The closest surface point gives a strictly local sign hint whenever
    # the ROI is not exactly on the surface. The broader phase-COM check in
    # detect_normal remains as a fallback for an on-surface ROI.
    toward_roi = roi_center - closest_points[anchor_face]
    if (
        np.linalg.norm(toward_roi) > 1e-10
        and float(np.dot(normal, toward_roi)) < 0.0
    ):
        normal = -normal
    return normal


def _extract_mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(vertex positions, triangle vertex-index array)`` out of an
    OVITO ``SurfaceMesh`` via ``get_vertices()``/``get_face_vertices()``.

    ``ConstructSurfaceModifier`` always produces a uniformly-triangulated
    mesh, so ``get_face_vertices()`` normally already returns a ``(n, 3)``
    array; a list-of-lists (non-uniform faces) is fan-triangulated as a
    defensive fallback. If OVITO's introspection API doesn't match (an
    ``AttributeError``), the caller in :func:`detect_normal` catches it and
    falls back to :func:`_fallback_normal`.
    """
    positions = np.asarray(mesh.get_vertices())
    face_vertices = mesh.get_face_vertices()
    if isinstance(face_vertices, np.ndarray) and face_vertices.ndim == 2:
        return positions, face_vertices.astype(int)

    tris = []
    for face in face_vertices:
        for k in range(1, len(face) - 1):
            tris.append([face[0], face[k], face[k + 1]])
    return positions, np.array(tris, dtype=int)


def _ovito_normal(
    atoms: Atoms,
    roi_center: np.ndarray,
    remove_species: Optional[list[str]],
    smoothing_level: int,
    radius: Optional[float],
    patch_radius: Optional[float] = None,
) -> np.ndarray:
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import ConstructSurfaceModifier
    from ovito.pipeline import Pipeline, StaticSource

    mesh_atoms = atoms.copy()
    if remove_species:
        symbols = np.array(mesh_atoms.get_chemical_symbols())
        keep = ~np.isin(symbols, list(remove_species))
        if not keep.any():
            raise ValueError(
                "remove_species would delete the entire structure -- "
                "cannot construct a surface mesh"
            )
        mesh_atoms = mesh_atoms[keep]

    modifier_kwargs = {"smoothing_level": smoothing_level}
    if radius is not None:
        modifier_kwargs["radius"] = radius

    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(mesh_atoms)))
    pipeline.modifiers.append(ConstructSurfaceModifier(**modifier_kwargs))
    data = pipeline.compute()
    mesh = data.surfaces["surface"]

    vertex_positions, faces = _extract_mesh_arrays(mesh)
    return _normal_from_mesh(
        vertex_positions,
        faces,
        roi_center,
        patch_radius,
        cell=np.asarray(atoms.get_cell()),
        pbc=atoms.get_pbc(),
    )


def detect_normal(
    atoms: Atoms,
    roi_center: np.ndarray,
    *,
    remove_species: Optional[list[str]] = None,
    phase_species: Optional[list[str]] = None,
    smoothing_level: int = 8,
    radius: Optional[float] = None,
    patch_radius: Optional[float] = None,
) -> np.ndarray:
    """Detect the local interface/surface normal near ``roi_center``.

    ``surface`` topology mode (one condensed phase with a free surface):
    call with ``remove_species=None`` -- the mesh is built directly on
    ``atoms``.

    ``interface`` topology mode (two condensed phases, so no free surface
    exists yet and a direct mesh would be empty): pass ``remove_species``,
    the phase to strip. Deletion happens on an *internal OVITO copy only* --
    ``atoms`` (the real, both-phase structure) is never mutated. Stripping
    that phase exposes the interface as a surface. The closest point on that
    surface anchors an edge-connected patch whose facet normals are averaged
    by area out to ``patch_radius`` geodesic distance (or only the anchor
    facet if ``patch_radius`` is ``None``).

    Falls back to a species center-of-mass heuristic (``phase_species``, or
    ``remove_species`` if that is not given) if OVITO is not installed or
    mesh construction fails (e.g. too few atoms remain after stripping).

    Sign convention: the returned normal points *away* from the condensed
    phase (outward, into the vacuum/solvent region), so rotating it onto
    ``+z`` places the exposed side above the ROI. Both the OVITO mesh path
    and the fallback heuristic are forced to this same convention, keyed off
    the same ``phase_species``/``remove_species`` reference, so they never
    disagree in sign regardless of OVITO's facet winding. (To invert the
    convention globally, flip the ``outward`` definition in
    :func:`_fallback_normal` and the dot-product test below.)

    Args:
        atoms: The (uncarved) structure to detect a normal in.
        roi_center: Position near which the normal is evaluated.
        remove_species: Phase to strip before meshing (interface mode).
        phase_species: Species defining the condensed phase, used only by
            the fallback heuristic (surface mode has no phase to remove).
        smoothing_level: OVITO ``ConstructSurfaceMeshModifier`` smoothing.
        radius: OVITO probe-sphere radius; ``None`` uses OVITO's default.
        patch_radius: Approximate geodesic radius over which edge-connected
            mesh-facet normals are area-averaged (see
            :func:`_normal_from_mesh`). ``None`` uses only the closest facet.

    Returns:
        Unit normal vector.
    """
    reference_species = phase_species or remove_species
    try:
        normal = _ovito_normal(
            atoms,
            roi_center,
            remove_species,
            smoothing_level,
            radius,
            patch_radius,
        )
    except Exception as exc:
        logger.warning(
            "OVITO surface-mesh normal detection failed (%s); falling back "
            "to the species center-of-mass heuristic.",
            exc,
        )
        return _fallback_normal(atoms, roi_center, reference_species)

    # Force the mesh normal to the outward convention (away from the phase),
    # so it agrees with _fallback_normal regardless of OVITO's facet winding.
    phase_dir = _phase_direction(atoms, roi_center, reference_species)
    if phase_dir is not None and float(np.dot(normal, phase_dir)) > 0.0:
        normal = -normal
    return normal
