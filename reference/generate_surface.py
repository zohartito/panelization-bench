"""Generate the benchmark reference surface deterministically.

Writes:
  - canopy_reference.obj   -- the target surface as a fine TRIANGULATED mesh.
                              The agent imports this (or rebuilds it from the
                              analytic definition in the prompt); score.py uses
                              it as the reference for fidelity + coverage.
  - surface_params.json    -- provenance: exact parameters + hashes.

Everything is a pure function of surface_def.SurfaceParams, so the surface is
byte-identical on every run and every machine. Units: METERS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import surface_def as sd

HERE = Path(__file__).resolve().parent

# Reference surface mesh density. Fine enough that point-to-triangle distance
# from any exported vertex is an accurate surface distance, coarse enough to
# import anywhere. This is NOT the panel grid (that is ~1.5 m / 27x13).
REF_NX = 80
REF_NY = 40


def _grid_vertices(nx: int, ny: int, prm: sd.SurfaceParams):
    verts = []
    for j in range(ny + 1):
        y = -prm.b + 2 * prm.b * j / ny
        for i in range(nx + 1):
            x = -prm.a + 2 * prm.a * i / nx
            verts.append((x, y, sd.z_of(x, y, prm)))
    return verts


def write_reference_obj(path: Path, prm: sd.SurfaceParams, nx: int, ny: int) -> None:
    verts = _grid_vertices(nx, ny, prm)
    lines = [
        "# panelization-bench reference surface (saddle canopy, corrugated hypar)",
        f"# z = {prm.h1}*(x/{prm.a})*(y/{prm.b}) "
        f"+ {prm.h2}*sin({prm.p}*pi*x/{prm.a})*cos({prm.q}*pi*y/{prm.b})",
        "# units: meters. Triangulated. Reference only -- NOT the panelization.",
    ]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for (x, y, z) in verts]

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i + 1  # OBJ is 1-based

    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            lines.append(f"f {a} {b} {c}")
            lines.append(f"f {a} {c} {d}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    prm = sd.DEFAULT
    obj_path = HERE / "canopy_reference.obj"
    write_reference_obj(obj_path, prm, REF_NX, REF_NY)

    digest = hashlib.sha256(obj_path.read_bytes()).hexdigest()
    params = {
        "surface": "corrugated_hypar_saddle_canopy",
        "definition": "z = h1*(x/a)*(y/b) + h2*sin(p*pi*x/a)*cos(q*pi*y/b)",
        "units": "meters",
        "params": asdict(prm),
        "footprint_m": {"x": [-prm.a, prm.a], "y": [-prm.b, prm.b]},
        "target_module_m": 1.5,
        "panel_grid_hint": {"nx": 27, "ny": 13},
        "planarity_tolerance_mm": 5.0,
        "reference_mesh": {
            "file": obj_path.name,
            "nx": REF_NX,
            "ny": REF_NY,
            "sha256": digest,
        },
    }
    (HERE / "surface_params.json").write_text(json.dumps(params, indent=2))
    print(f"wrote {obj_path.name} ({REF_NX}x{REF_NY}) sha256={digest[:12]}...")
    print(f"wrote surface_params.json")


if __name__ == "__main__":
    main()
