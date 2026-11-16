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
