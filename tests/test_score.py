"""Exact-answer tests for the neutral scorer. The scorer must be provably correct
before it judges anything, so every fixture here has a hand-computable answer and
is asserted against it -- planarity, point-to-triangle distance, unit handling,
and each cheat-verdict."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402


def mesh(verts, faces):
    return score.Mesh(np.asarray(verts, float), [list(f) for f in faces])


# --------------------------------------------------------------------------- #
# Planarity metric -- known-planar reads ~0, known-non-planar reads exact value.
# --------------------------------------------------------------------------- #
def test_planar_quad_reads_zero():
    quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    assert score.planarity_deviation_m(quad) < 1e-12


def test_tilted_planar_quad_reads_zero():
    # 4 points exactly on plane z = 2x + 3y + 1 -> planar, must read 0 (proves the
    # metric is best-fit-plane, not naive z-distance).
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad = np.array([[x, y, 2 * x + 3 * y + 1] for x, y in pts], float)
    assert score.planarity_deviation_m(quad) < 1e-12


def test_saddle_quad_reads_exact_deviation():
    # P alternating +h,-h,+h,-h over the unit square corners. Covariance is
    # diag(4,4,4h^2); best-fit normal is (0,0,1); max vertex-plane distance == h.
    h = 0.003  # 3 mm, in metres
    quad = np.array([[1, 1, h], [-1, 1, -h], [-1, -1, h], [1, -1, -h]], float)
    assert score.planarity_deviation_m(quad) == pytest.approx(h, abs=1e-12)


def test_saddle_scored_in_mm_matches_analytic():
    h = 0.003
    quad = np.array([[1, 1, h], [-1, 1, -h], [-1, -1, h], [1, -1, -h]], float)
    r = score.score_mesh(mesh(quad, [[0, 1, 2, 3]]), units="m")
    assert r["planarity_mm"]["max"] == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Unit-mismatch trap -- same shape in m and mm must give the same mm answer.
# --------------------------------------------------------------------------- #
def test_units_consistency_m_vs_mm():
    h = 0.003
    quad_m = np.array([[1, 1, h], [-1, 1, -h], [-1, -1, h], [1, -1, -h]], float)
    r_m = score.score_mesh(mesh(quad_m, [[0, 1, 2, 3]]), units="m")
    r_mm = score.score_mesh(mesh(quad_m * 1000.0, [[0, 1, 2, 3]]), units="mm")
    assert r_m["planarity_mm"]["max"] == pytest.approx(r_mm["planarity_mm"]["max"],
                                                       rel=1e-9)


# --------------------------------------------------------------------------- #
# Point-to-triangle distance -- hand-computable cases.
# --------------------------------------------------------------------------- #
def test_point_to_triangle_face_edge_vertex():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)  # (1,3,3)
    # above the interior -> perpendicular distance
    assert score.point_set_to_mesh_dist(np.array([[0.2, 0.2, 0.5]]), tri)[0] \
        == pytest.approx(0.5, abs=1e-9)
    # beyond a vertex -> distance to that vertex
    assert score.point_set_to_mesh_dist(np.array([[-1.0, -1.0, 0.0]]), tri)[0] \
        == pytest.approx(np.sqrt(2), abs=1e-9)
    # beyond an edge -> perpendicular distance to the edge line
    assert score.point_set_to_mesh_dist(np.array([[0.5, -1.0, 0.0]]), tri)[0] \
        == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Face classification / triangulation dodge.
# --------------------------------------------------------------------------- #
def test_triangulated_mesh_is_flagged_dodge():
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    tris = [[0, 1, 2], [0, 2, 3]]
    r = score.score_mesh(mesh(v, tris), units="m")
    assert r["face_breakdown"]["triangles"] == 2
    assert r["face_breakdown"]["quads"] == 0
    assert r["verdict"] == "DODGE_TRIANGULATED"


# --------------------------------------------------------------------------- #
# Cheat verdicts against a reference surface.
# --------------------------------------------------------------------------- #
def _flat_reference(n=8, size=4.0):
    """A flat n x n reference patch (z=0), triangulated, spanning [0,size]^2."""
    verts, faces = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((size * i / n, size * j / n, 0.0))
    vid = lambda i, j: j * (n + 1) + i
    for j in range(n):
        for i in range(n):
            faces.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
            faces.append([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)])
    return mesh(verts, faces)


def test_module_violation_on_subdivision():
    # tiny 0.2 m quads -> planarity trivial but module far below the ~1.5 m band
    ref = _flat_reference()
    verts, faces = [], []
    n = 20
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((4.0 * i / n, 4.0 * j / n, 0.0))
    vid = lambda i, j: j * (n + 1) + i
    for j in range(n):
        for i in range(n):
            faces.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    r = score.score_mesh(mesh(verts, faces), units="m", reference=ref)
    assert r["module_m"]["median_edge"] == pytest.approx(0.2, abs=1e-9)
    assert r["verdict"] == "MODULE_VIOLATION"


def test_off_surface_when_panels_drift():
    # a FULL ~1.33 m panelization of the 4x4 patch, but lifted 1 m off the
    # surface -> coverage/module fine, only off-surface fails.
    ref = _flat_reference(size=4.0)
    verts, faces = [], []
    n = 3
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((4.0 * i / n, 4.0 * j / n, 1.0))  # z = 1 m, off surface
    vid = lambda i, j: j * (n + 1) + i
    for j in range(n):
        for i in range(n):
            faces.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    r = score.score_mesh(mesh(verts, faces), units="m", reference=ref)
    assert r["off_surface_mm"]["max_mm"] == pytest.approx(1000.0, abs=1.0)
    assert r["verdict"] == "OFF_SURFACE"


def test_incomplete_coverage_when_partial():
    # one small quad covering a corner of a big reference -> huge coverage gap
    ref = _flat_reference(size=40.0)
    q = np.array([[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [0, 1.5, 0]], float)
    r = score.score_mesh(mesh(q, [[0, 1, 2, 3]]), units="m", reference=ref)
    assert r["coverage"]["area_ratio"] < 0.1
    assert r["verdict"] == "INCOMPLETE_COVERAGE"


def test_clean_pass_on_faithful_panelization():
    # panelize the flat reference at ~1.33 m quads, on-surface -> PASS
    ref = _flat_reference(size=4.0)
    verts, faces = [], []
    n = 3
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((4.0 * i / n, 4.0 * j / n, 0.0))
    vid = lambda i, j: j * (n + 1) + i
    for j in range(n):
        for i in range(n):
            faces.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    r = score.score_mesh(mesh(verts, faces), units="m", reference=ref)
    assert r["planarity_mm"]["max"] < 1e-6
    assert r["verdict"] == "PASS"


def test_sliver_gaming_is_not_a_clean_pass():
    """Planar + covering + correct module, but achieved by slivering.

    This is the sliver-gaming case: an agent warps its grid until every quad is
    planar, at the cost of degenerating some panels to near-zero width. Every
    gating check passes, so a planarity-only scorer calls it a clean win — but a
    facade of slivers cannot be built. It must qualify to PASS_UNBUILDABLE.

    Observed live: a Blender agent hit 3.49 mm max planarity over the whole
    canopy with a worst aspect ratio of 3.6e12 and 4 degenerate panels.
    """
    size, n = 20.0, 12          # 144 quads at a 1.67 m module (inside the band)
    ref = _flat_reference(size=size)
    verts, faces = [], []
    # One interior column is pulled to within a micron of its neighbour. That
    # slivers 12 of 144 panels (8.3%) — under the 10% degeneracy gate, so the
    # mesh passes every gating check and only the buildability qualifier can
    # catch it. Matches the shape of the observed live failure.
    xs = [size * i / n for i in range(n + 1)]
    xs[1] = xs[0] + 1e-6
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((xs[i], size * j / n, 0.0))
    vid = lambda i, j: j * (n + 1) + i
    for j in range(n):
        for i in range(n):
            faces.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    r = score.score_mesh(mesh(verts, faces), units="m", reference=ref)

    assert r["planarity_mm"]["quads_failing_tol"] == 0, "slivers are still planar"
    assert r["verdict"] == "PASS_UNBUILDABLE"
    assert r["checks"]["buildable"]["pass"] is False
    assert r["checks"]["buildable"]["reasons"], "must say why it is unbuildable"


# --------------------------------------------------------------------------- #
# Parser round-trips (OBJ + DXF control lane).
# --------------------------------------------------------------------------- #
def test_analytic_off_surface_is_exact():
    # a quad ON a curved height field must read ~0 off-surface with NO noise floor
    surf = lambda x, y: 0.1 * x * y
    pts = [(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]
    q = np.array([[x, y, surf(x, y)] for x, y in pts], float)
    r = score.score_mesh(mesh(q, [[0, 1, 2, 3]]), units="m", surface_z=surf)
    assert r["off_surface_mm"]["max_mm"] < 1e-9
    assert r["off_surface_mm"]["method"] == "analytic_vertical"


def test_fold_guard_trips_non_heightfield():
    # two vertices at the same (x,y), different z -> not a height field
    v = [[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [0, 1.5, 0],
         [0, 0, 1.0], [1.5, 0, 1.0], [1.5, 1.5, 1.0], [0, 1.5, 1.0]]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    r = score.score_mesh(mesh(v, faces), units="m", surface_z=lambda x, y: 0.0)
    assert r["off_surface_mm"]["fold_pairs"] > 0
    assert r["verdict"] == "NON_HEIGHTFIELD"


def test_trilemma_corner_not_planar():
    # non-planar-but-quality-and-conforming quad on a flat reference -> the
    # "gave up on planarity" corner
    ref = _flat_reference(size=1.6)
    q = np.array([[0, 0, 0], [1.6, 0, 0], [1.6, 1.6, 0.02], [0, 1.6, 0]], float)
    r = score.score_mesh(mesh(q, [[0, 1, 2, 3]]), units="m", reference=ref,
                         surface_z=lambda x, y: 0.0)
    assert r["planarity_mm"]["max"] > 5.0
    assert r["trilemma_corner"] == "QUALITY+CONFORMING (not planar)"


def test_obj_roundtrip(tmp_path):
    p = tmp_path / "q.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    m = score.load_mesh(p)
    assert len(m.vertices) == 4 and m.faces == [[0, 1, 2, 3]]


def test_dxf_3dface_roundtrip(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_3dface([(0, 0, 0), (1, 0, 0), (1, 1, 0.1), (0, 1, 0)])   # a quad
    msp.add_3dface([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)])     # a triangle
    p = tmp_path / "panels.dxf"
    doc.saveas(p)
    m = score.load_mesh(p)
    counts = sorted(len(f) for f in m.faces)
    assert counts == [3, 4]  # triangular 3DFACE collapsed to a tri
