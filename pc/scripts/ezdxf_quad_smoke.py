"""Smoke test: can the headless ezdxf engine (the AutoCAD-MCP ezdxf backend's
library) produce planar-quad facade panels and export DXF with NO AutoCAD?
Builds a small doubly-curved grid, panelizes into quads, writes each as a
3DFACE, saves, reopens, and reports planarity + entity count."""
import math
import ezdxf

# small doubly-curved canopy patch (not the full benchmark — just a smoke test)
NX, NY = 8, 5          # -> 7x4 = 28 quad panels
W, D = 12.0, 8.0
def z(u, v):
    return 1.6 * math.sin(math.pi * u) * math.sin(0.5 * math.pi * v)

def P(i, j):
    u, v = i / (NX - 1), j / (NY - 1)
    return (u * W, v * D, z(u, v))

doc = ezdxf.new()                      # headless, in-memory
msp = doc.modelspace()

def planarity_mm(a, b, c, d):
    # max deviation of the 4 corners from the plane of a,b,c (in mm; units=m)
    ax, ay, az = a; bx, by, bz = b; cx, cy, cz = c
    ux, uy, uz = bx-ax, by-ay, bz-az
    vx, vy, vz = cx-ax, cy-ay, cz-az
    nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
    nl = math.sqrt(nx*nx+ny*ny+nz*nz) or 1.0
    nx, ny, nz = nx/nl, ny/nl, nz/nl
    worst = 0.0
    for (px, py, pz) in (a, b, c, d):
        dist = abs((px-ax)*nx + (py-ay)*ny + (pz-az)*nz)
        worst = max(worst, dist)
    return worst * 1000.0

n_quads = 0
worst_mm = 0.0
for i in range(NX - 1):
    for j in range(NY - 1):
        a, b, c, d = P(i, j), P(i+1, j), P(i+1, j+1), P(i, j+1)
        msp.add_3dface([a, b, c, d])     # one planar-quad panel
        worst_mm = max(worst_mm, planarity_mm(a, b, c, d))
        n_quads += 1

out = "/private/tmp/claude-501/-/bdd9d096-1ece-4d70-9e99-da49eb4f9dba/scratchpad/panels_smoke.dxf"
doc.saveas(out)

# reopen and count, proving it's a real persisted DXF
doc2 = ezdxf.readfile(out)
faces = [e for e in doc2.modelspace() if e.dxftype() == "3DFACE"]
import os
print(f"ezdxf version: {ezdxf.__version__}")
print(f"quads authored: {n_quads}")
print(f"3DFACE entities in reopened DXF: {len(faces)}")
print(f"DXF file size: {os.path.getsize(out)} bytes")
print(f"worst as-quad planarity deviation: {worst_mm:.1f} mm (naive grid, pre-optimization)")
print("HEADLESS_EZDXF_OK" if len(faces) == n_quads else "MISMATCH")
