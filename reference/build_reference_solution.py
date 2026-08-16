"""Build the reference solutions that bound the PLANAR / QUALITY / CONFORMING
trilemma, and export them for provenance. NEITHER is a scoring target -- scoring
never rewards similarity to these. They only establish what is achievable.

  canopy_gold_planar.obj          -- discrete conjugate (Q-net) over the full
                                     footprint. Planar to machine epsilon, full
                                     coverage, ~1.5 m module, but UNBUILDABLE
                                     (slivers). The planarity-existence proof.
  canopy_gold_principal_patch.obj -- a principal curvature-line net. Planar AND
                                     buildable quality (orthogonal), but only
                                     covers a small patch (curvature lines do not
                                     conform to the rectangular boundary). The
                                     proof that planar+quality quads EXIST on A.

Prints the measured numbers used in the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import surface_def as sd

HERE = Path(__file__).resolve().parent
PRM = sd.DEFAULT
NX, NY = 27, 13


def qplan(cor: np.ndarray) -> float:
    c = cor - cor.mean(0)
    _, _, vt = np.linalg.svd(c)
    return float(np.abs(c @ vt[-1]).max())


def min_angle_deg(cor: np.ndarray) -> float:
    m = 180.0
    for k in range(4):
        a = cor[(k - 1) % 4] - cor[k]
        b = cor[(k + 1) % 4] - cor[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        m = min(m, float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))))
    return m


def _surface_point_in_plane(pred, c, n):
    x, y = float(pred[0]), float(pred[1])
    for _ in range(80):
        g = n[0] * (x - c[0]) + n[1] * (y - c[1]) + n[2] * (sd.z_of(x, y, PRM) - c[2])
        fx, fy = sd.grad(x, y, PRM)
        dgx, dgy = n[0] + n[2] * fx, n[1] + n[2] * fy
        d = dgx * dgx + dgy * dgy
        if d < 1e-18:
            break
        x -= g * dgx / d
        y -= g * dgy / d
        if abs(g) < 1e-13:
            break
    return np.array([x, y, sd.z_of(x, y, PRM)])


def build_conjugate_net() -> np.ndarray:
    xs = np.linspace(-PRM.a, PRM.a, NX + 1)
    ys = np.linspace(-PRM.b, PRM.b, NY + 1)
    P = np.zeros((NY + 1, NX + 1, 3))
    for i, x in enumerate(xs):
        P[0, i] = (x, ys[0], sd.z_of(x, ys[0], PRM))
    for j, y in enumerate(ys):
        P[j, 0] = (xs[0], y, sd.z_of(xs[0], y, PRM))
    for j in range(1, NY + 1):
        for i in range(1, NX + 1):
            p10, p01, p00 = P[j, i - 1], P[j - 1, i], P[j - 1, i - 1]
            c = (p10 + p01 + p00) / 3.0
            nrm = np.cross(p10 - p00, p01 - p00)
            nl = np.linalg.norm(nrm)
            nrm = nrm / nl if nl > 1e-12 else np.array([0, 0, 1.0])
            P[j, i] = _surface_point_in_plane(p10 + p01 - p00, c, nrm)
    return P


def _principal_dirs(x, y):
    fx, fy = sd.grad(x, y, PRM)
    fxx, fyy, fxy = sd.hessian(x, y, PRM)
    W = np.sqrt(1 + fx * fx + fy * fy)
    I = np.array([[1 + fx * fx, fx * fy], [fx * fy, 1 + fy * fy]])
    II = np.array([[fxx, fxy], [fxy, fyy]]) / W
    w, v = np.linalg.eig(np.linalg.solve(I, II))
    o = np.argsort(w.real)
    return v[:, o[0]].real, v[:, o[1]].real


def _trace(x0, y0, which, ds=0.05):
    pts = [(x0, y0, sd.z_of(x0, y0, PRM))]
    x, y, prev = x0, y0, None
    while True:
        d = (_principal_dirs(x, y)[which]).astype(float)
        fx, fy = sd.grad(x, y, PRM)
        T = np.array([d[0], d[1], fx * d[0] + fy * d[1]])
        nrm = np.linalg.norm(T)
        if nrm < 1e-9:
            break
        s = np.array([d[0], d[1]]) / nrm
        if prev is not None and np.dot(s, prev) < 0:
            s = -s
        prev = s
        x += ds * s[0]
        y += ds * s[1]
        if not (-PRM.a <= x <= PRM.a and -PRM.b <= y <= PRM.b):
            break
        pts.append((x, y, sd.z_of(x, y, PRM)))
    return np.array(pts)


def _resample(pts, n):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.array([np.interp(t, s, pts[:, k]) for k in range(3)]).T


def build_principal_patch(sx=-19.0, sy=-9.0) -> np.ndarray:
    row = _resample(_trace(sx, sy, 1), NX + 1)
    col = _resample(_trace(sx, sy, 0), NY + 1)
    P = np.zeros((NY + 1, NX + 1, 3))
    P[0, :] = row
    P[:, 0] = col
    P[0, 0] = row[0]
    for j in range(1, NY + 1):
        for i in range(1, NX + 1):
            p10, p01, p00 = P[j, i - 1], P[j - 1, i], P[j - 1, i - 1]
            c = (p10 + p01 + p00) / 3.0
            nrm = np.cross(p10 - p00, p01 - p00)
            nl = np.linalg.norm(nrm)
            nrm = nrm / nl if nl > 1e-12 else np.array([0, 0, 1.0])
            P[j, i] = _surface_point_in_plane(p10 + p01 - p00, c, nrm)
    return P


def write_obj(P: np.ndarray, path: Path, header: str) -> None:
    ny, nx = P.shape[0] - 1, P.shape[1] - 1
    lines = [f"# {header}"]
    lines += [f"v {P[j, i, 0]:.6f} {P[j, i, 1]:.6f} {P[j, i, 2]:.6f}"
              for j in range(ny + 1) for i in range(nx + 1)]
    vid = lambda i, j: j * (nx + 1) + i + 1
    for j in range(ny):
        for i in range(nx):
            lines.append(f"f {vid(i, j)} {vid(i+1, j)} {vid(i+1, j+1)} {vid(i, j+1)}")
    path.write_text("\n".join(lines) + "\n")


def stats(P: np.ndarray) -> dict:
    ny, nx = P.shape[0] - 1, P.shape[1] - 1
    dev, ang = [], []
    for j in range(ny):
        for i in range(nx):
            cor = np.array([P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]])
            dev.append(qplan(cor))
            ang.append(min_angle_deg(cor))
    dev, ang = np.array(dev), np.array(ang)
    return {"planarity_mm_max": dev.max() * 1000, "planarity_mm_mean": dev.mean() * 1000,
            "min_angle_worst": ang.min(), "min_angle_median": float(np.median(ang)),
            "slivers_under_20deg": int((ang < 20).sum()), "quads": len(dev)}


def main() -> None:
    net = build_conjugate_net()
    write_obj(net, HERE / "canopy_gold_planar.obj",
              "GOLD conjugate net: PLANAR to machine epsilon + full coverage, but "
              "NOT fabricable (slivers). Planarity-existence proof only -- NOT a target.")
    s1 = stats(net)
    patch = build_principal_patch()
    write_obj(patch, HERE / "canopy_gold_principal_patch.obj",
              "Principal curvature-line net: PLANAR + buildable QUALITY, but covers "
              "only a small patch (curvature lines do not conform to the boundary).")
    s2 = stats(patch)
    xr = patch[..., 0].max() - patch[..., 0].min()
    yr = patch[..., 1].max() - patch[..., 1].min()
    print("canopy_gold_planar.obj (conjugate net, CONFORMING corner):")
    print(f"  planarity max {s1['planarity_mm_max']:.3e} mm (machine epsilon) | "
          f"min-angle median {s1['min_angle_median']:.1f} deg worst {s1['min_angle_worst']:.1f} "
          f"| slivers<20deg {s1['slivers_under_20deg']}/{s1['quads']}")
    print("canopy_gold_principal_patch.obj (principal net, QUALITY corner):")
    print(f"  planarity max {s2['planarity_mm_max']:.3e} mm | min-angle median "
          f"{s2['min_angle_median']:.1f} deg worst {s2['min_angle_worst']:.1f} "
          f"| slivers<20deg {s2['slivers_under_20deg']}/{s2['quads']} | patch footprint "
          f"{xr:.0f} x {yr:.0f} m of 40 x 20")


if __name__ == "__main__":
    main()
