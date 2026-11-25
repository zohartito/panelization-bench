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
import runner  # noqa: E402
from reference import surface_def as sd  # noqa: E402


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


def _write_obj(path, submitted):
    """Measure the exported artifact, not a fabricated score dictionary."""
    lines = ["v " + " ".join(f"{c:.17g}" for c in v) for v in submitted.vertices]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in submitted.faces]
    path.write_text("\n".join(lines) + "\n")
    return path


def _insert_edge_midpoint(verts, face):
    # Same panel boundary and area; only its vertex count changes.
    a, b = np.asarray(verts[face[0]]), np.asarray(verts[face[1]])
    face.insert(1, len(verts))
    verts.append((a + b) / 2)


def _flat_panel_probe(kind):
    verts, faces = _grid(n=3, size=4.0)
    if kind == "triangles":
        a, b, c, d = faces.pop(0)
        faces.extend([[a, b, c], [a, c, d]])
    elif kind == "pentagon":
        _insert_edge_midpoint(verts, faces[0])
    return mesh(verts, faces)


@pytest.mark.parametrize("kind,counts,verdict", [
    ("quads", (9, 0, 0), "PASS"),
    ("triangles", (8, 2, 0), "DODGE_TRIANGULATED"),
    ("pentagon", (8, 0, 1), "DODGE_NGONS"),
])
def test_exported_flat_panels_must_all_be_quads(tmp_path, kind, counts, verdict):
    path = _write_obj(tmp_path / "panels.obj", _flat_panel_probe(kind))
    ref = _write_obj(tmp_path / "reference.obj", _flat_reference(size=4.0))
    rec = runner.evaluate_run(
        "ezdxf-control", "checker-regression",
        [runner.ToolCall(ts=0.0, tool="export_obj", args_summary=kind)], 0.0, path,
        agent_claimed_success=True, agent_reported_planarity=True,
        reference=ref, surface_def=tmp_path / "no_analytic_surface.py")
    r = rec["geometry"]
    assert r["face_breakdown"] == dict(zip(("quads", "triangles", "ngons"), counts))
    assert r["planarity_mm"]["max"] < 1e-6
    assert r["coverage"]["area_ratio"] == pytest.approx(1.0)
    assert r["coverage"]["max_gap_m"] < 1e-9
    assert r["off_surface_mm"]["max_mm"] < 1e-6
    assert r["checks"]["module"]["pass"]
    assert r["verdict"] == verdict
    assert r["checks"]["not_triangulated"]["pass"] is (counts[1] == 0)
    if counts[2]:
        assert r["checks"]["not_ngons"]["pass"] is False
    assert rec["run_verdict"] == ("PASS" if kind == "quads" else "FALSE_SUCCESS")


def _canopy_panel_probe(convert_nonplanar):
    # Same 27x13 construction and analytic surface as runner._ezdxf_weak_agent.
    prm, nx, ny = sd.DEFAULT, 27, 13
    xs = [-prm.a + 2 * prm.a * i / nx for i in range(nx + 1)]
    ys = [-prm.b + 2 * prm.b * j / ny for j in range(ny + 1)]
    verts = [(x, y, sd.z_of(x, y, prm)) for y in ys for x in xs]
    vid = lambda i, j: j * (nx + 1) + i
    faces = [[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
             for j in range(ny) for i in range(nx)]
    if convert_nonplanar:
        for face in faces:
            corners = np.asarray([verts[i] for i in face])
            if score.planarity_deviation_m(corners) * 1000 > score.Thresholds().tol_mm:
                _insert_edge_midpoint(verts, face)
    return mesh(verts, faces)


@pytest.mark.parametrize("convert_nonplanar", [False, True])
def test_exported_canopy_cannot_hide_nonplanarity_in_ngons(tmp_path, convert_nonplanar):
    path = _write_obj(tmp_path / "canopy.obj", _canopy_panel_probe(convert_nonplanar))
    rec = runner.evaluate_run(
        "ezdxf-control", "checker-regression",
        [runner.ToolCall(ts=0.0, tool="export_obj", args_summary="canopy")], 0.0, path,
        agent_claimed_success=True, agent_reported_planarity=True)
    r = rec["geometry"]
    assert r["panel_count"] == 351
    for check in ("height_field", "module", "coverage", "off_surface", "degeneracy"):
        assert r["checks"][check]["pass"], check
    if convert_nonplanar:
        assert 0 < r["face_breakdown"]["ngons"] < r["panel_count"]
        assert r["planarity_mm"]["quads_failing_tol"] == 0
        assert r["ngon_planarity_mm_max"] > r["planarity_mm"]["tolerance_mm"]
        assert r["verdict"] == "DODGE_NGONS"
        assert r["checks"]["not_ngons"]["pass"] is False
    else:
        assert r["face_breakdown"] == {"quads": 351, "triangles": 0, "ngons": 0}
        assert r["planarity_mm"]["quads_failing_tol"] > 0
        assert r["verdict"] == "PLANARITY_FAIL"
    assert rec["run_verdict"] == "FALSE_SUCCESS"


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


def _soundness_probe(kind):
    verts, faces = _grid(n=3, size=4.0)
    if kind == "duplicate":
        faces.append(list(faces[4]))          # centre panel exported twice
    elif kind == "bowtie":
        a, b, c, d = faces[4]
        faces[4] = [a, b, d, c]               # planar, same corners, self-crossing
    return mesh(verts, faces)


@pytest.mark.parametrize("kind,verdict", [
    ("clean", "PASS"),
    ("duplicate", "OVERLAPPING_PANELS"),
    ("bowtie", "OVERLAPPING_PANELS"),
])
def test_overlapping_panels_are_not_a_pass(tmp_path, kind, verdict):
    # Both cheats are planar, on-surface, in module and within the coverage band
    # (a duplicated panel reads area ratio 10/9), so only soundness catches them.
    path = _write_obj(tmp_path / "panels.obj", _soundness_probe(kind))
    ref = _flat_reference(size=4.0)
    r = score.score_mesh(score.load_mesh(path), units="m", reference=ref,
                         surface_z=lambda x, y: 0.0)
    assert r["checks"]["coverage"]["pass"]
    assert r["verdict"] == verdict
    s = r["checks"]["sound"]
    assert (s["duplicate_panels"], s["bowtie_panels"]) == {
        "clean": (0, 0), "duplicate": (1, 0), "bowtie": (0, 1)}[kind]
    assert (s["non_manifold_edges"] > 0) is (kind == "duplicate")


def test_sliver_gaming_is_not_a_clean_pass(tmp_path):
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
    path = _write_obj(tmp_path / "slivers.obj", mesh(verts, faces))
    r = score.score_mesh(score.load_mesh(path), units="m", reference=ref)

    assert r["planarity_mm"]["quads_failing_tol"] == 0, "slivers are still planar"
    assert r["verdict"] == "PASS_UNBUILDABLE"
    assert r["checks"]["buildable"]["pass"] is False
    assert r["checks"]["buildable"]["reasons"], "must say why it is unbuildable"


# --------------------------------------------------------------------------- #
# Topology -- vertex irregularity against its Gauss-Bonnet lower bound (par).
# --------------------------------------------------------------------------- #
def _grid(n=3, size=3.0):
    verts = [(size * i / n, size * j / n, 0.0) for j in range(n + 1)
             for i in range(n + 1)]
    vid = lambda i, j: j * (n + 1) + i
    faces = [[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
             for j in range(n) for i in range(n)]
    return verts, faces


def test_regular_grid_is_at_par_zero():
    r = score.score_mesh(mesh(*_grid()), units="m")
    assert r["topology"]["par"] == 0
    assert r["topology"]["vertex_score"] == 0
    assert r["topology"]["excess"] == 0


def test_remeshed_square_pays_topological_cost():
    # quad-in-quad: same square, 5 planar quads. par is still 0 (rectangle), but
    # the 4 inner vertices have degree 3 and the 4 corners degree 3 (want 2).
    v = [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0],
         [1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]]
    f = [[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    r = score.score_mesh(mesh(v, f), units="m")
    t = r["topology"]
    assert (t["par"], t["vertex_score"], t["excess"]) == (0, 8, 8)
    assert t["irregular_vertices"] == 8
    assert "excess 8" in score.human_summary(r)


def test_forced_irregularity_is_par_not_excess():
    # triangle split into 3 quads: the degree-3 centre is forced by the 60 deg
    # corners, so par = 1 and the mesh is optimal (excess 0).
    s = np.sqrt(3)
    v = [[0, 0, 0], [2, 0, 0], [1, s, 0],             # corners
         [1, 0, 0], [1.5, s / 2, 0], [0.5, s / 2, 0],  # edge midpoints
         [1, s / 3, 0]]                                # centroid
    f = [[0, 3, 6, 5], [3, 1, 4, 6], [6, 4, 2, 5]]
    t = score.score_mesh(mesh(v, f), units="m")["topology"]
    assert (t["par"], t["vertex_score"], t["excess"]) == (1, 1, 0)


def test_topology_welds_per_panel_vertices():
    # DXF 3DFACE export: every panel carries its own vertex copies
    verts, faces = _grid()
    split_v = [verts[i] for f in faces for i in f]
    split_f = [[4 * k + c for c in range(4)] for k in range(len(faces))]
    t = score.score_mesh(mesh(split_v, split_f), units="m")["topology"]
    assert (t["par"], t["excess"], t["euler_characteristic"]) == (0, 0, 1)


def test_par_comes_from_the_domain_not_the_submission():
    # the gold conjugate net's boundary is folded (295/333 deg "corners"), so its
    # own angles would lower its bound to par 2. Against the 40x20 rectangular
    # canopy reference the domain has exactly 4 right-angle corners: par = 0.
    root = Path(__file__).resolve().parents[1] / "reference"
    ref = score.load_mesh(root / "canopy_reference.obj")
    gold = score.load_mesh(root / "canopy_gold_planar.obj")
    assert [w for _, w in score.domain_corners(ref.vertices, ref.faces)] == [(2,)] * 4
    assert score.topology_score(gold.vertices, gold.faces)["par"] == 2
    t = score.score_mesh(gold, units="m", reference=ref)["topology"]
    assert (t["par"], t["par_source"]) == (0, "reference")
    assert t["excess"] == t["vertex_score"] > 0


def test_corner_wants_tie_is_a_set():
    assert score.corner_wants(90.0) == (2,)
    assert score.corner_wants(180.0) == (3,)
    assert score.corner_wants(135.0) == (2, 3)


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
