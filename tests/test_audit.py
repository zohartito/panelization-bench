"""Regression probes for the confirmed public-repository audit findings."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score
from reference import surface_def as sd


def canopy():
    xs, ys = np.linspace(-20, 20, 28), np.linspace(-10, 10, 14)
    vertices = np.array([(x, y, sd.z_of(x, y)) for y in ys for x in xs])
    faces = [[j * 28 + i, j * 28 + i + 1, (j + 1) * 28 + i + 1,
              (j + 1) * 28 + i] for j in range(13) for i in range(27)]
    return score.Mesh(vertices, faces)


def judged(mesh):
    return score.score_mesh(mesh, units="m", reference=canopy(), surface_z=sd.z_of)


def test_pzb01_mixed_ngons_cannot_pass():
    m = canopy()
    keep = min(range(len(m.faces)),
               key=lambda i: score.planarity_deviation_m(m.vertices[m.faces[i]]))
    vertices = m.vertices.tolist()
    for i, face in enumerate(m.faces):
        if i != keep:
            x, y = m.vertices[face[:2], :2].mean(axis=0)
            vertices.append([x, y, sd.z_of(x, y)])
            face.insert(1, len(vertices) - 1)
    m.vertices = np.array(vertices)
    r = judged(m)
    assert r["face_breakdown"] == {"quads": 1, "triangles": 0, "ngons": 350}
    assert r["ngon_planarity_mm_max"] > 100
    assert not r["verdict"].startswith("PASS")
    assert r["trilemma_corner"] != "ALL_THREE"


def test_pzb01_mixed_triangles_cannot_pass():
    v = np.array([[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [0, 1.5, 0], [3, 0, 0]])
    r = score.score_mesh(score.Mesh(v, [[0, 1, 2, 3], [1, 4, 2]]))
    assert r["verdict"] == "DODGE_TRIANGULATED"


def flattened_panels():
    m = canopy()
    vertices = []
    for face in m.faces:
        q = m.vertices[face]
        center = q.mean(0)
        normal = np.linalg.svd(q - center, full_matrices=False)[2][-1]
        vertices.extend(q - ((q - center) @ normal)[:, None] * normal)
    return score.Mesh(np.array(vertices),
                      [list(range(i, i + 4)) for i in range(0, len(vertices), 4)])


def test_pzb02_disconnected_flattening_cannot_pass():
    r = judged(flattened_panels())
    assert r["planarity_mm"]["max"] < 1e-8
    assert 100 < r["off_surface_mm"]["max_mm"] < 250
    assert not r["verdict"].startswith("PASS")
    assert r["trilemma_corner"] != "ALL_THREE"


def test_pzb02_reference_par_ignores_submission_components():
    r = judged(flattened_panels())
    assert r["topology"]["euler_characteristic"] == 351
    assert r["topology"]["par"] == 0
    assert r["topology"]["excess"] == r["topology"]["vertex_score"] > 0


def flat_grid(n=4):
    v = np.array([(x * 1.5, y * 1.5, 0) for y in range(n + 1) for x in range(n + 1)])
    f = [[j*(n+1)+i, j*(n+1)+i+1, (j+1)*(n+1)+i+1, (j+1)*(n+1)+i]
         for j in range(n) for i in range(n)]
    return score.Mesh(v, f)


def test_pzb02_seams_holes_and_boundary():
    ref = flat_grid()
    split = score.Mesh(np.array([ref.vertices[i] for f in ref.faces for i in f]),
                       [list(range(i, i+4)) for i in range(0, 64, 4)])
    assert score.score_mesh(split, reference=ref)["verdict"] == "PASS"
    hole = score.Mesh(ref.vertices, ref.faces[:5] + ref.faces[6:])
    assert score.score_mesh(hole, reference=ref)["verdict"] == "INVALID_CONNECTIVITY"
    shifted = score.Mesh(ref.vertices + [0.3, 0, 0], ref.faces)
    r = score.score_mesh(shifted, reference=ref, surface_z=lambda x, y: 0)
    assert r["verdict"] == "INVALID_CONNECTIVITY"


import json
import runner


def test_pzb03_completion_timing():
    rec = runner.ToolCall(ts=1000, tool="slow", args_summary="", duration_ms=60000)
    r = runner.summarize_transcript([rec], run_start=1000)
    assert r["time_to_first_successful_tool_call_ms"] == 60000
    assert r["wall_time_ms"] == 60000


def test_pzb03_evaluate_preserves_setup_and_end(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    transcript, report = tmp_path / "calls.jsonl", tmp_path / "report.json"
    logger = runner.ToolCallLogger(transcript)
    clock[0] += 0.3
    with logger.call("work", ""):
        clock[0] += 0.5
    clock[0] += 0.2
    logger.finish()
    runner.main(["evaluate", "--lane", "ezdxf-control", "--transcript", str(transcript),
                 "--json", str(report)])
    r = json.loads(report.read_text())
    assert r["catalog"]["time_to_first_successful_tool_call_ms"] == pytest.approx(800)
    assert r["catalog"]["wall_time_ms"] == pytest.approx(1000)
    assert r["run_start"] == 1000
    assert r["run_end"] == pytest.approx(1001)


@pytest.mark.parametrize("lane", ["blender", "rhino-community", "rhino-mcneel"])
def test_pzb05_unknown_components_warn(lane, monkeypatch):
    # Never inspect an installed desktop app or a real user's plugin directory.
    monkeypatch.setattr(runner, "_app_version", lambda path: "1.0")
    monkeypatch.setattr(runner, "_rhino_plugin_version", lambda: "unknown")
    monkeypatch.setattr(runner, "_pkg_version", lambda name: "unknown")
    stack, warnings = runner.probe_stack(lane, "test-model")
    assert "unknown" in stack["mcp_server"].lower()
    assert any("stack.mcp_server" in w for w in warnings)
    assert any("stack.plugin_or_backend" in w for w in warnings)


@pytest.mark.parametrize("claimed", [False, True])
@pytest.mark.parametrize("reported", [False, True])
def test_pzb06_unbuildable_is_qualified_pass(claimed, reported):
    r = runner.assign_run_verdict({"successful_tool_calls": 3},
                                 {"verdict": "PASS_UNBUILDABLE"}, claimed, reported)
    assert r == ("PASS_UNBUILDABLE" if reported else "PASS_UNBUILDABLE_NO_SELF_CHECK")


@pytest.mark.parametrize("shift", [0.0, 0.002])
def test_pzbn1_excess_coverage_fails(shift):
    m = canopy()
    layers = score.Mesh(np.concatenate([m.vertices, m.vertices + [shift, shift, 0.2]]),
                        m.faces + [[i + len(m.vertices) for i in f] for f in m.faces])
    r = judged(layers)
    assert r["coverage"]["area_ratio"] == pytest.approx(2.0)
    assert not r["checks"]["coverage"]["pass"]
    assert r["trilemma_corner"] != "ALL_THREE"
