"""score.py -- the neutral, engine-independent geometry scorer.

The one rule of this benchmark: geometry is judged OUTSIDE any 3D engine, on the
exported artifact, by this single script. Rhino must not be scored with Rhino's
tools and Blender with Blender's -- that would compare measurement code, not
engines. Everything here is plain numpy on exported vertices.

An agent must not be able to pass by anything other than actually solving the
problem, so the scorer enforces four independent constraints, each recorded with
its numbers (not just pass/fail):

  PLANARITY   per quad: max distance of its 4 vertices to their best-fit plane
              (SVD). Reported in mm. Tolerance 5 mm.
  MODULE      panels must be ~1.5 m. Kills the subdivide-to-0.2 m dodge, where
              planarity becomes nearly free but fabrication cost explodes.
  COVERAGE    the panels must actually cover the target surface (area ratio +
              largest gap from a surface sample to the nearest panel). Kills
              panelize-10%-and-stop.
  OFF-SURFACE panels must still approximate the input surface. A perfectly planar
              mesh that has drifted off the building (what a folding optimizer
              produces) is not a solution. Kills the fold / flatten cheats.

Faces are classified by vertex count: quad (4) scored; triangle (3) planarity is
trivially 0 -- any triangle has DODGED the planar-quad problem. Any non-quad
face prevents a pass; ngon (>4) planarity is still measured for diagnostics.

QUALITY (min interior angle, aspect ratio, degenerate-panel count) is measured and
recorded but NOT a hard gate by default: a facade of planar slivers is not
fabricable, yet we have not proven a quality+planar solution exists at 1.5 m on
this surface, so gating on it would be unfair. Egregious collapse (a mesh that is
mostly zero-area slivers) IS flagged, as it indicates a broken or gamed mesh.

TOPOLOGY (vertex irregularity) is graded the same way. Discrete Gauss-Bonnet gives
a provable lower bound, `par`, on how many irregular vertices ANY all-quad mesh of
the domain must carry; the excess over par is irregularity the mesher chose, not
irregularity the domain forced. On the rectangular canopy par = 0, so every
irregular vertex of a remeshed submission is recorded as topological cost.

Units: stated explicitly (--units), or --units auto (guesses from the bounding box
and warns). A scorer that silently compares mm to m is worthless. All distances
reported in mm; the assumed unit and bounding box are always echoed.

Usage:
    python3 score.py mesh.obj --reference reference/canopy_reference.obj --units m
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# Unit -> metres scale factor.
UNIT_SCALE = {"m": 1.0, "mm": 0.001, "cm": 0.01, "in": 0.0254, "ft": 0.3048}


@dataclass
class Thresholds:
    tol_mm: float = 5.0                 # planarity tolerance
    module_target_m: float = 1.5
    module_band: tuple[float, float] = (0.9, 2.4)   # allowed median edge length
    coverage_area_min: float = 0.85     # panel area / reference area
    coverage_area_max: float = 1.15     # symmetric 15% allowance; rejects stacked layers
    coverage_gap_max_m: float = 2.0     # largest surface-sample-to-panel gap
    off_surface_max_mm: float = 250.0   # exported vertex must be this close to surface
    flatten_z_ratio: float = 0.3        # exported z-range < ratio*reference => suspect
    degenerate_area_frac: float = 0.02  # panel area < frac*median => degenerate
    degenerate_max_frac: float = 0.10   # > this fraction degenerate => flag mesh
    # Buildability qualifiers. Quality is deliberately NOT a hard gate — no
    # planar+quality+conforming solution has been shown to exist at this module,
    # so failing an engine for it would be unfair. But a mesh that hits the
    # planarity tolerance by degenerating into slivers must not read as a clean
    # PASS either: that is sliver-gaming, and a facade of slivers is unbuildable.
    # Tripping any of these qualifies the verdict to PASS_UNBUILDABLE.
    buildable_min_angle_deg: float = 5.0    # worst interior angle below this
    buildable_max_aspect: float = 100.0     # worst panel aspect ratio above this


@dataclass
class Mesh:
    vertices: np.ndarray                       # (N, 3) float, native units
    faces: list[list[int]] = field(default_factory=list)
    source: str = ""


# --------------------------------------------------------------------------- #
# Parsers: OBJ, PLY, DXF (control lane), 3DM.
# --------------------------------------------------------------------------- #
def load_mesh(path: Path) -> Mesh:
    ext = path.suffix.lower()
    return {".obj": _load_obj, ".ply": _load_ply, ".dxf": _load_dxf,
            ".3dm": _load_3dm}.get(ext, _unsupported)(path)


def _unsupported(path: Path) -> Mesh:
    raise ValueError(f"unsupported format {path.suffix} (use .obj/.ply/.dxf/.3dm)")


def _load_obj(path: Path) -> Mesh:
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                v = int(tok.split("/")[0])
                idx.append(v - 1 if v > 0 else len(verts) + v)
            faces.append(idx)
    return Mesh(np.asarray(verts, float), faces, str(path))


def _load_ply(path: Path) -> Mesh:
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not an ASCII PLY file")
    n_vert = n_face = header_end = 0
    fmt = ""
    element = None
    vertex_properties = []
    for i, ln in enumerate(lines):
        t = ln.split()
        if t and t[0] == "format":
            fmt = t[1]
        elif len(t) >= 3 and t[0] == "element":
            element = t[1]
            if element == "vertex":
                n_vert = int(t[2])
            elif element == "face":
                n_face = int(t[2])
        elif t and t[0] == "property" and element == "vertex":
            vertex_properties.append((t[-1], t[1] == "list"))
        elif t and t[0] == "end_header":
            header_end = i + 1
            break
    if fmt != "ascii":
        raise ValueError(f"only ASCII PLY supported (got {fmt})")
    body = lines[header_end:]
    if not {"x", "y", "z"} <= {name for name, is_list in vertex_properties if not is_list}:
        raise ValueError("PLY vertices require scalar x, y and z properties")
    verts = []
    for row in body[:n_vert]:
        fields, values, offset = row.split(), {}, 0
        for name, is_list in vertex_properties:
            if is_list:
                offset += 1 + int(fields[offset])
            else:
                values[name] = float(fields[offset])
                offset += 1
        verts.append(tuple(values[name] for name in ("x", "y", "z")))
    faces = []
    for i in range(n_vert, n_vert + n_face):
        t = list(map(int, body[i].split()))
        faces.append(t[1:1 + t[0]])
    return Mesh(np.asarray(verts, float), faces, str(path))


def _load_dxf(path: Path) -> Mesh:
    """Read 3DFACE / MESH / POLYFACE entities (the ezdxf control lane writes
    3DFACE). A triangular 3DFACE repeats its last corner -> collapse to a tri."""
    import ezdxf

    doc = ezdxf.readfile(str(path))
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []

    def add(pts: list[tuple[float, float, float]]) -> None:
        base = len(verts)
        # drop consecutive duplicate corners (triangular 3DFACE)
        clean: list[tuple[float, float, float]] = []
        for p in pts:
            if not clean or np.linalg.norm(np.array(p) - np.array(clean[-1])) > 1e-9:
                clean.append(p)
        if len(clean) >= 2 and np.linalg.norm(
            np.array(clean[0]) - np.array(clean[-1])
        ) < 1e-9:
            clean = clean[:-1]
        if len(clean) < 3:
            return
        verts.extend(clean)
        faces.append(list(range(base, base + len(clean))))

    for e in doc.modelspace():
        dt = e.dxftype()
        if dt == "3DFACE":
            add([tuple(e.dxf.vtx0), tuple(e.dxf.vtx1),
                 tuple(e.dxf.vtx2), tuple(e.dxf.vtx3)])
        elif dt == "MESH":
            base = len(verts)
            vlist = [tuple(v) for v in e.vertices]
            verts.extend(vlist)
            for f in e.faces:
                faces.append([base + i for i in f])
        elif dt == "POLYLINE" and e.is_poly_face_mesh:
            base = len(verts)
            vfaces = list(e.faces())  # (vertices tuple) per face
            # ezdxf gives face vertices directly
            for face in vfaces:
                add([tuple(v.dxf.location) for v in face])
    if not verts:
        raise ValueError("no 3DFACE/MESH geometry found in DXF")
    return Mesh(np.asarray(verts, float), faces, str(path))


def _load_3dm(path: Path) -> Mesh:
    import rhino3dm

    model = rhino3dm.File3dm.Read(str(path))
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for obj in model.Objects:
        geo = obj.Geometry
        if not isinstance(geo, rhino3dm.Mesh):
            continue
        base = len(verts)
        for i in range(len(geo.Vertices)):
            v = geo.Vertices[i]
            verts.append((v.X, v.Y, v.Z))
        for i in range(len(geo.Faces)):
            f = geo.Faces[i]
            idx = [base + f[0], base + f[1], base + f[2]]
            if len(f) == 4 and f[3] != f[2]:
                idx.append(base + f[3])
            faces.append(idx)
    if not verts:
        raise ValueError("no mesh geometry in .3dm (NURBS not meshed?)")
    return Mesh(np.asarray(verts, float), faces, str(path))


# --------------------------------------------------------------------------- #
# Geometry.
# --------------------------------------------------------------------------- #
def planarity_deviation_m(corners: np.ndarray) -> float:
    """Max distance of a face's vertices to their best-fit plane (SVD)."""
    if len(corners) < 4:
        return 0.0
    centered = corners - corners.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return float(np.abs(centered @ vt[-1]).max())


def triangulate(face_idx: list[int], verts: np.ndarray) -> np.ndarray:
    """Fan-triangulate a face -> (T,3,3) triangle vertex array."""
    return np.array([[verts[face_idx[0]], verts[face_idx[k]], verts[face_idx[k + 1]]]
                     for k in range(1, len(face_idx) - 1)])


def face_area(corners: np.ndarray) -> float:
    total = 0.0
    for k in range(1, len(corners) - 1):
        total += 0.5 * np.linalg.norm(
            np.cross(corners[k] - corners[0], corners[k + 1] - corners[0])
        )
    return float(total)


def quad_min_angle_deg(corners: np.ndarray) -> float:
    m = 180.0
    n = len(corners)
    for k in range(n):
        a = corners[(k - 1) % n] - corners[k]
        b = corners[(k + 1) % n] - corners[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        m = min(m, float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))))
    return m


def quad_aspect(corners: np.ndarray) -> float:
    e = [np.linalg.norm(corners[(k + 1) % len(corners)] - corners[k])
         for k in range(len(corners))]
    return float(max(e) / max(min(e), 1e-12))


def edge_lengths(corners: np.ndarray) -> list[float]:
    return [float(np.linalg.norm(corners[(k + 1) % len(corners)] - corners[k]))
            for k in range(len(corners))]


def corner_wants(angle_deg: float, target_deg: float = 90.0,
                 tie_tol: float = 1e-6) -> tuple[int, ...]:
    """Degrees a boundary vertex with this material-side angle is content with:
    round(angle/90)+1, at least 2. At an exact rounding tie (135 deg -> 1.5) both
    treatments are legal, so the vertex carries the set {floor+1, ceil+1} and is
    scored against the nearer one."""
    ratio = angle_deg / target_deg
    if abs(ratio - np.floor(ratio) - 0.5) <= tie_tol:
        low = max(int(np.floor(ratio)) + 1, 2)
        high = max(int(np.ceil(ratio)) + 1, 2)
        if low != high:
            return (low, high)
    return (max(int(np.round(ratio)) + 1, 2),)


def _mesh_graph(V: np.ndarray, faces: list[list[int]], weld_tol_m: float = 1e-6):
    """Welded connectivity: corner count, edge uses, material-side angle sum and
    position per vertex, plus the face count. Summing incident panel corner
    angles gives the material-side angle, so a hole corner reads 360 minus its
    own angle for free."""
    # weld coincident vertices: the DXF/3DM lanes emit one vertex copy per panel
    _, weld = np.unique(np.round(V / weld_tol_m).astype(np.int64), axis=0,
                        return_inverse=True)
    weld = weld.ravel()
    corners: dict[int, int] = defaultdict(int)
    edge_uses: dict[tuple[int, int], int] = defaultdict(int)
    angle: dict[int, float] = defaultdict(float)
    pos: dict[int, np.ndarray] = {}
    n_faces = 0
    for raw in faces:
        # a zero-width sliver welds shut; it is already counted as degenerate
        face = [i for k, i in enumerate(raw) if weld[i] != weld[raw[k - 1]]]
        n = len(face)
        if len({int(weld[i]) for i in face}) < 3:
            continue
        n_faces += 1
        for k in range(n):
            a, b = int(weld[face[k]]), int(weld[face[(k + 1) % n]])
            corners[a] += 1
            edge_uses[(min(a, b), max(a, b))] += 1
            pos[a] = V[face[k]]
            u = V[face[k - 1]] - V[face[k]]
            w = V[face[(k + 1) % n]] - V[face[k]]
            nu, nw = np.linalg.norm(u), np.linalg.norm(w)
            if nu > 1e-12 and nw > 1e-12:
                angle[a] += float(np.degrees(np.arccos(np.clip(u @ w / (nu * nw),
                                                               -1, 1))))
    boundary = {v for e, uses in edge_uses.items() if uses == 1 for v in e}
    return corners, edge_uses, angle, pos, boundary, n_faces


def domain_corners(V: np.ndarray, faces: list[list[int]]) -> list[tuple]:
    """(point, wanted degrees) for every corner of a domain mesh -- its boundary
    vertices that do not want the straight-edge degree 3. Any triangulation of
    the domain gives the same corners, so the reference mesh serves."""
    _, _, angle, pos, boundary, _ = _mesh_graph(V, faces)
    return [(pos[v], corner_wants(angle[v])) for v in boundary
            if corner_wants(angle[v]) != (3,)]


def topology_score(V: np.ndarray, faces: list[list[int]],
                   domain: list[tuple] | None = None, domain_par: int | None = None) -> dict:
    """Vertex irregularity against par, its discrete Gauss-Bonnet lower bound.

    Interior vertices want degree 4, boundary vertices 3, and each domain corner
    corner_wants() of its angle. Par is a property of the DOMAIN, so with a
    reference each domain corner is assigned to the nearest boundary vertex of
    the submission -- a mesh cannot lower its own bound by distorting its
    boundary. Without one, the submission's own boundary angles stand in. (On a
    non-covering mesh two domain corners can land on one vertex; coverage
    already fails such a mesh.)
    Degree counts incident EDGES, +1 on the boundary -- not distinct
    neighbours, which undercounts on non-simple meshes. For any all-quad mesh
    sum(desired - degree) is the invariant corner_excess + 4*chi, so the L1
    vertex score can never go below
    par = min over tie choices t of |corner_excess + 4*chi + t|.
    Ported from arjunnarayanan/par-quad (compute_par, compare_gmsh.score_mesh)."""
    corners, edge_uses, angle, pos, boundary, n_faces = _mesh_graph(V, faces)
    wants = {v: (3,) if v in boundary else (4,) for v in corners}
    if domain is None:
        wants.update({v: corner_wants(angle[v]) for v in boundary})
    elif boundary:
        ring = sorted(boundary)
        ring_pos = np.array([pos[v] for v in ring])
        for point, options in domain:
            wants[ring[int(np.linalg.norm(ring_pos - point, axis=1).argmin())]] = options
    defect = {v: min(abs(corners[v] + (v in boundary) - w) for w in wants[v])
              for v in corners}
    corner_excess = sum(min(wants[v]) - 3 for v in boundary)
    n_ties = sum(len(wants[v]) - 1 for v in boundary)
    chi = len(corners) - len(edge_uses) + n_faces
    par = min(abs(corner_excess + 4 * chi + t) for t in range(n_ties + 1))
    if domain_par is not None:
        par = domain_par
    vertex_score = sum(defect.values())
    return {
        "par": par,
        "par_source": "own_boundary" if domain is None else "reference",
        "vertex_score": vertex_score,
        "excess": vertex_score - par,
        "irregular_vertices": sum(1 for d in defect.values() if d),
        "euler_characteristic": chi,
        "note": "graded axis, not a gate; par is a strict bound only for all-quad meshes",
    }



def connectivity_check(V, faces, RV, ref_faces, boundary_tol_m):
    """Require a single manifold canopy with the reference topology and boundary.

    Coincident per-panel exports are welded on the same 1 micrometre coordinate
    grid used by topology_score. Boundary distances are measured in plan.
    """
    def graph(vertices, panels):
        corners, edges, _, pos, boundary, count = _mesh_graph(vertices, panels)
        adjacent = defaultdict(set)
        boundary_degree = defaultdict(int)
        for (a, b), uses in edges.items():
            adjacent[a].add(b)
            adjacent[b].add(a)
            if uses == 1:
                boundary_degree[a] += 1
                boundary_degree[b] += 1
        unseen = set(corners)
        components = 0
        while unseen:
            components += 1
            pending = [unseen.pop()]
            while pending:
                for other in adjacent[pending.pop()] & unseen:
                    unseen.remove(other)
                    pending.append(other)
        segments = np.array([[pos[a][:2], pos[b][:2]] for (a, b), n in edges.items()
                             if n == 1]).reshape(-1, 2, 2)
        manifold = all(n <= 2 for n in edges.values()) and all(
            n == 2 for n in boundary_degree.values())
        return components, len(corners) - len(edges) + count, manifold, segments

    def distance(source, target):
        if not len(source) or not len(target):
            return float("inf")
        points = np.concatenate([source[:, 0], source.mean(axis=1)])
        a, b = target[:, 0], target[:, 1]
        ab = b - a
        denominator = np.maximum((ab * ab).sum(1), 1e-30)
        distances = []
        for point in points:
            t = np.clip(((point - a) * ab).sum(1) / denominator, 0, 1)
            distances.append(np.linalg.norm(point - (a + t[:, None] * ab), axis=1).min())
        return float(max(distances))

    components, chi, manifold, boundary = graph(V, faces)
    _, ref_chi, _, ref_boundary = graph(RV, ref_faces)
    gap = max(distance(boundary, ref_boundary), distance(ref_boundary, boundary))
    return {"pass": components == 1 and manifold and chi == ref_chi and gap <= boundary_tol_m,
            "components": components, "manifold": manifold,
            "euler_characteristic": chi, "reference_euler_characteristic": ref_chi,
            "boundary_gap_m": gap, "boundary_tolerance_m": boundary_tol_m,
            "weld_grid_m": 1e-6}


def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                          c: np.ndarray) -> np.ndarray:
    """Vectorized closest point of p on each triangle (a,b,c: (T,3)).
    Returns distances (T,). Ericson region test."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = p - c
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = 1.0 / np.where((va + vb + vc) == 0, 1e-30, va + vb + vc)
    v = vb * denom
    w = vc * denom
    closest = a + ab * v[:, None] + ac * w[:, None]           # interior default
    # region masks (priority: verts, then edges, then interior already set)
    m = (d1 <= 0) & (d2 <= 0)
    closest[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    closest[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    closest[m] = c[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where((d1 - d3) == 0, 0.0, d1 / np.where((d1 - d3) == 0, 1e-30, d1 - d3))
    closest[m] = (a + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where((d2 - d6) == 0, 0.0, d2 / np.where((d2 - d6) == 0, 1e-30, d2 - d6))
    closest[m] = (a + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    dd = (d4 - d3) + (d5 - d6)
    t = np.where(dd == 0, 0.0, (d4 - d3) / np.where(dd == 0, 1e-30, dd))
    closest[m] = (b + (c - b) * t[:, None])[m]
    return np.linalg.norm(closest - p, axis=1)


def point_set_to_mesh_dist(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Min distance from each point to a triangle set (tris: (T,3,3)). (P,)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return np.array([_closest_on_triangles(p, a, b, c).min() for p in points])


def off_surface_analytic(V: np.ndarray, surface_z, fold_xy_tol: float = 1e-3,
                         fold_z_tol: float = 0.05) -> dict:
    """Off-surface deviation via the analytic height field z=f(x,y): for each
    vertex, |z_v - f(x_v, y_v)| in mm -- EXACT, no triangulation noise floor.

    Vertical distance is an upper bound on true perpendicular distance (they
    differ by 1/sqrt(1+|grad f|^2)), i.e. conservative, which is the safe
    direction. The fold check is only a vertex-pair heuristic: pairs within
    fold_xy_tol in plan and over fold_z_tol apart in height are flagged. It does
    not detect projected-face overlaps or certify a height field."""
    from scipy.spatial import cKDTree

    xy = V[:, :2]
    surf = np.array([surface_z(float(x), float(y)) for x, y in xy])
    dev = np.abs(V[:, 2] - surf)
    pairs = cKDTree(xy).query_pairs(fold_xy_tol)
    folds = sum(1 for a, b in pairs if abs(V[a, 2] - V[b, 2]) > fold_z_tol)
    return {"max_mm": float(dev.max() * 1000.0), "mean_mm": float(dev.mean() * 1000.0),
            "method": "analytic_vertical", "fold_pairs": int(folds)}


# --------------------------------------------------------------------------- #
# Units.
# --------------------------------------------------------------------------- #
def _resolve_units(vertices: np.ndarray, units: str) -> tuple[str, float, list[str]]:
    warnings: list[str] = []
    max_dim = float((vertices.max(0) - vertices.min(0)).max())
    if units == "auto":
        guess = "mm" if max_dim > 2000 else "m"
        warnings.append(
            f"--units auto GUESSED '{guess}' from max bbox dimension {max_dim:.3f}; "
            f"sound only if the model really is that unit."
        )
        units = guess
    if units not in UNIT_SCALE:
        raise ValueError(f"unknown unit '{units}' (use {list(UNIT_SCALE)})")
    return units, UNIT_SCALE[units], warnings


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def score_mesh(mesh: Mesh, units: str = "m", reference: Mesh | None = None,
               ref_units: str = "m", thr: Thresholds | None = None,
               surface_z=None) -> dict:
    thr = thr or Thresholds()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return {"verdict": "EMPTY", "panel_count": 0,
                "checks": {}, "notes": ["mesh has no faces"]}

    unit_label, scale, warnings = _resolve_units(mesh.vertices, units)
    V = mesh.vertices * scale

    quad_dev, ngon_dev = [], []
    n_tri = n_quad = n_ngon = 0
    all_edges: list[float] = []
    min_angles, aspects, areas = [], [], []
    panel_tris = []                       # triangulation of ALL panels (coverage)
    for face in mesh.faces:
        n = len(face)
        if n < 3:
            continue
        cor = V[face]
        panel_tris.append(triangulate(face, V))
        if n == 3:
            n_tri += 1
        elif n == 4:
            n_quad += 1
            quad_dev.append(planarity_deviation_m(cor))
        else:
            n_ngon += 1
            ngon_dev.append(planarity_deviation_m(cor))
        if n >= 3:  # module + quality + area cover every panel, incl. triangles
            all_edges.extend(edge_lengths(cor))
            min_angles.append(quad_min_angle_deg(cor))
            aspects.append(quad_aspect(cor))
            areas.append(face_area(cor))
    panel_tris = np.concatenate(panel_tris) if panel_tris else np.zeros((0, 3, 3))

    qd = np.array(quad_dev) * 1000.0
    tol_m = thr.tol_mm / 1000.0
    n_fail = int((np.array(quad_dev) > tol_m).sum()) if quad_dev else 0
    edges = np.array(all_edges) if all_edges else np.array([0.0])
    angles = np.array(min_angles) if min_angles else np.array([0.0])
    asp = np.array(aspects) if aspects else np.array([0.0])
    ar = np.array(areas) if areas else np.array([0.0])
    med_area = float(np.median(ar)) if ar.size else 0.0

    bbox_min, bbox_max = V.min(0), V.max(0)

    # --- off-surface (analytic height field if available; exact, no noise) ---
    off_surface = coverage = domain = domain_par = connectivity = None
    if surface_z is not None:
        off_surface = off_surface_analytic(V, surface_z)
    # --- coverage needs the reference mesh; also fallback off-surface ---
    if reference is not None and len(reference.faces):
        rs = UNIT_SCALE.get(ref_units, 1.0)
        RV = reference.vertices * rs
        ref_tris = np.concatenate(
            [triangulate(f, RV) for f in reference.faces if len(f) >= 3]
        )
        domain = domain_corners(RV, reference.faces)
        domain_par = topology_score(RV, reference.faces)["par"]
        connectivity = connectivity_check(V, mesh.faces, RV, reference.faces,
                                           thr.off_surface_max_mm / 1000.0)
        ref_area = float(sum(0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
                             for t in ref_tris))
        if off_surface is None:  # no analytic surface -> mesh distance (noisy)
            d_v = point_set_to_mesh_dist(V, ref_tris) * 1000.0
            off_surface = {"max_mm": float(d_v.max()), "mean_mm": float(d_v.mean()),
                           "method": "point_to_reference_mesh", "fold_pairs": 0}
            warnings.append(
                "off-surface measured against the triangulated reference "
                "(~10 mm noise floor); pass --surface-def for exact analytic distance."
            )
        gap = point_set_to_mesh_dist(RV, panel_tris) if len(panel_tris) else \
            np.full(len(RV), 1e9)
        panel_area = float(ar.sum())
        coverage = {
            "area_ratio": panel_area / ref_area if ref_area else 0.0,
            "max_gap_m": float(gap.max()),
            "reference_area_m2": ref_area,
            "panel_area_m2": panel_area,
        }

    n_degen = int((ar < thr.degenerate_area_frac * med_area).sum()) if med_area else 0

    result = {
        "units_assumed": unit_label,
        "panel_count": n_tri + n_quad + n_ngon,
        "face_breakdown": {"quads": n_quad, "triangles": n_tri, "ngons": n_ngon},
        "planarity_mm": {
            "max": float(qd.max()) if qd.size else None,
            "mean": float(qd.mean()) if qd.size else None,
            "median": float(np.median(qd)) if qd.size else None,
            "quads_failing_tol": n_fail,
            "quads_total": n_quad,
            "pass_rate": (1.0 - n_fail / n_quad) if n_quad else None,
            "tolerance_mm": thr.tol_mm,
        },
        "ngon_planarity_mm_max": float(max(ngon_dev) * 1000) if ngon_dev else None,
        "module_m": {
            "median_edge": float(np.median(edges)),
            "mean_edge": float(edges.mean()),
            "p5_edge": float(np.percentile(edges, 5)),
            "p95_edge": float(np.percentile(edges, 95)),
            "target": thr.module_target_m,
            "band": list(thr.module_band),
        },
        "quality": {
            "min_angle_deg_worst": float(angles.min()),
            "min_angle_deg_median": float(np.median(angles)),
            "aspect_median": float(np.median(asp)),
            "aspect_worst": float(asp.max()),
            "degenerate_panels": n_degen,
            "note": "recorded, graded axis -- not a hard gate (see score.py header)",
        },
        "topology": topology_score(V, mesh.faces, domain, domain_par),
        "connectivity": connectivity,
        "off_surface_mm": off_surface,
        "coverage": coverage,
        "bbox_m": {"min": bbox_min.tolist(), "max": bbox_max.tolist(),
                   "size": (bbox_max - bbox_min).tolist()},
        "warnings": warnings,
    }
    result["checks"], result["verdict"] = _evaluate(result, thr, reference)
    result["trilemma_corner"] = _trilemma_corner(result)
    return result


def _trilemma_corner(r: dict) -> str:
    """Which corner of the PLANAR / QUALITY / CONFORMING trilemma a run landed on.
    Derived from numbers already computed; distinguishes 'found a sophisticated
    technique and hit the known boundary problem' from 'gave up'. Maps every run
    to the three measured reference corners."""
    p = r["planarity_mm"]
    planar = r["checks"]["all_quads"]["pass"] and p["quads_failing_tol"] == 0
    quality = r["quality"]["min_angle_deg_worst"] >= 20.0 and r["face_breakdown"]["quads"] > 0
    cov = r["coverage"]
    if cov is None:
        conforming = None
    else:
        conforming = r["checks"]["coverage"]["pass"] and r["checks"]["connectivity"]["pass"]
    if conforming is None:
        return "UNKNOWN_NO_REFERENCE"
    got = [planar, quality, conforming]
    if all(got):
        return "ALL_THREE"
    if planar and quality and not conforming:
        return "PLANAR+QUALITY (non-conforming)"
    if planar and conforming and not quality:
        return "PLANAR+CONFORMING (slivers)"
    if quality and conforming and not planar:
        return "QUALITY+CONFORMING (not planar)"
    return "NONE"


def _evaluate(r: dict, thr: Thresholds, reference: Mesh | None) -> tuple[dict, str]:
    """Each check records value+threshold+pass. Verdict = first failing check by
    precedence (most fundamental first), else PASS."""
    fb = r["face_breakdown"]
    checks: dict[str, dict] = {}

    if r["off_surface_mm"] is not None:
        checks["height_field"] = {
            "pass": r["off_surface_mm"].get("fold_pairs", 0) == 0,
            "fold_pairs": r["off_surface_mm"].get("fold_pairs", 0),
        }

    checks["module"] = {
        "pass": thr.module_band[0] <= r["module_m"]["median_edge"] <= thr.module_band[1],
        "median_edge_m": r["module_m"]["median_edge"], "band": list(thr.module_band),
    }
    checks["not_triangulated"] = {
        "pass": fb["triangles"] == 0,
        "quads": fb["quads"], "triangles": fb["triangles"],
    }
    checks["all_quads"] = {
        "pass": fb["quads"] > 0 and fb["quads"] == r["panel_count"],
        **fb,
    }
    if r["coverage"] is not None:
        checks["coverage"] = {
            "pass": (thr.coverage_area_min <= r["coverage"]["area_ratio"] <= thr.coverage_area_max
                     and r["coverage"]["max_gap_m"] <= thr.coverage_gap_max_m),
            "area_ratio": r["coverage"]["area_ratio"],
            "area_ratio_band": [thr.coverage_area_min, thr.coverage_area_max],
            "max_gap_m": r["coverage"]["max_gap_m"],
        }
    if r["off_surface_mm"] is not None:
        z = r["bbox_m"]["size"][2]
        flat = False
        if reference is not None:
            flat = False  # flatten handled via off_surface + z-range below
        checks["off_surface"] = {
            "pass": r["off_surface_mm"]["max_mm"] <= thr.off_surface_max_mm,
            "max_mm": r["off_surface_mm"]["max_mm"],
            "threshold_mm": thr.off_surface_max_mm,
        }
    if r["connectivity"] is not None:
        checks["connectivity"] = r["connectivity"]
    checks["degeneracy"] = {
        "pass": r["quality"]["degenerate_panels"] <= thr.degenerate_max_frac * max(
            r["panel_count"], 1),
        "degenerate_panels": r["quality"]["degenerate_panels"],
        "panel_count": r["panel_count"],
    }
    checks["planarity"] = {
        "pass": r["planarity_mm"]["quads_failing_tol"] == 0 and fb["quads"] > 0,
        "quads_failing_tol": r["planarity_mm"]["quads_failing_tol"],
        "max_mm": r["planarity_mm"]["max"],
    }

    # Precedence: fundamental problems first. Detected vertex-pair folds
    # headline; this heuristic does not certify a height field. Then
    # triangulation (the task was planar QUADS), coverage, module, and so on.
    order = [
        ("height_field", "NON_HEIGHTFIELD"),
        ("not_triangulated", "DODGE_TRIANGULATED"),
        ("all_quads", "NON_QUAD_PANELS"),
        ("coverage", "INCOMPLETE_COVERAGE"),
        ("module", "MODULE_VIOLATION"),
        ("off_surface", "OFF_SURFACE"),
        ("connectivity", "INVALID_CONNECTIVITY"),
        ("degeneracy", "DEGENERATE_PANELS"),
        ("planarity", "PLANARITY_FAIL"),
    ]
    for key, verdict in order:
        if key in checks and not checks[key]["pass"]:
            return checks, verdict

    # Everything gating passed. Now qualify the PASS: did it get there honestly,
    # or by slivering? Quality is graded rather than gating (see Thresholds), but
    # a sliver mesh must not be readable as a clean win.
    q = r["quality"]
    unbuildable_reasons = []
    if q["min_angle_deg_worst"] < thr.buildable_min_angle_deg:
        unbuildable_reasons.append(
            f"worst interior angle {q['min_angle_deg_worst']:.1f} deg "
            f"< {thr.buildable_min_angle_deg} deg")
    if q["aspect_worst"] > thr.buildable_max_aspect:
        unbuildable_reasons.append(
            f"worst aspect ratio {q['aspect_worst']:.1f} "
            f"> {thr.buildable_max_aspect}")
    if q["degenerate_panels"] > 0:
        unbuildable_reasons.append(f"{q['degenerate_panels']} degenerate panels")

    checks["buildable"] = {
        "pass": not unbuildable_reasons,
        "reasons": unbuildable_reasons,
    }
    if unbuildable_reasons:
        return checks, "PASS_UNBUILDABLE"
    return checks, "PASS"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def human_summary(r: dict) -> str:
    if r.get("verdict") == "EMPTY":
        return "VERDICT: EMPTY -- mesh has no faces."
    p, fb, q = r["planarity_mm"], r["face_breakdown"], r["quality"]
    out = [
        f"VERDICT: {r['verdict']}   (units: {r['units_assumed']}, "
        f"tol {p['tolerance_mm']} mm)",
        f"  panels {r['panel_count']}  |  quads {fb['quads']} triangles "
        f"{fb['triangles']} ngons {fb['ngons']}",
    ]
    if p["max"] is not None:
        out.append(f"  planarity mm: max {p['max']:.3f} mean {p['mean']:.3f} "
                   f"median {p['median']:.3f}  | failing {p['quads_failing_tol']}"
                   f"/{p['quads_total']}")
    m = r["module_m"]
    out.append(f"  module: median edge {m['median_edge']:.2f} m "
               f"(band {m['band'][0]}-{m['band'][1]}, target {m['target']})")
    if r["coverage"]:
        c = r["coverage"]
        out.append(f"  coverage: area ratio {c['area_ratio']:.2f}  "
                   f"max gap {c['max_gap_m']:.2f} m")
    if r["off_surface_mm"]:
        os_ = r["off_surface_mm"]
        out.append(f"  off-surface: max {os_['max_mm']:.2f} mm  mean "
                   f"{os_['mean_mm']:.2f} mm  ({os_.get('method', '?')})")
    out.append(f"  quality: min-angle worst {q['min_angle_deg_worst']:.1f} deg "
               f"median {q['min_angle_deg_median']:.1f} | aspect median "
               f"{q['aspect_median']:.2f} worst {q['aspect_worst']:.1f} | "
               f"degenerate {q['degenerate_panels']}")
    t = r["topology"]
    out.append(f"  topology: {t['irregular_vertices']} irregular vertices | vertex "
               f"score {t['vertex_score']} vs par {t['par']} -> excess {t['excess']}")
    if "trilemma_corner" in r:
        out.append(f"  trilemma corner: {r['trilemma_corner']}")
    for w in r.get("warnings", []):
        out.append(f"  WARNING: {w}")
    return "\n".join(out)


def load_surface_z(path: Path):
    """Load z_of + DEFAULT from a surface_def.py, return f(x,y)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("surface_def_dyn", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["surface_def_dyn"] = mod   # so @dataclass can resolve __module__
    spec.loader.exec_module(mod)
    prm = mod.DEFAULT
    return lambda x, y: mod.z_of(x, y, prm)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Neutral planar-quad geometry scorer.")
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--units", default="m", help="m|mm|cm|in|ft|auto")
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--ref-units", default="m")
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--surface-def", type=Path, default=None,
                    help="surface_def.py for EXACT analytic off-surface distance")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    thr = Thresholds(tol_mm=args.tol_mm)
    mesh = load_mesh(args.mesh)
    reference = load_mesh(args.reference) if args.reference else None
    surface_z = load_surface_z(args.surface_def) if args.surface_def else None
    result = score_mesh(mesh, args.units, reference, args.ref_units, thr, surface_z)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(human_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
