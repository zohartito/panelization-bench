"""Analytic definition of the benchmark reference surface.

Single source of truth. Imported by the surface generator, the scorer's
fidelity check, and the reference planarizer so they can never disagree about
what the target surface is.

The surface is a saddle canopy on the plan-domain
    x in [-A, A]  (40 m span when A = 20)
    y in [-B, B]  (20 m span when B = 10)

    z(x, y) = h1 * (x/A) * (y/B)                      # hypar base
            + h2 * sin(p*pi*x/A) * cos(q*pi*y/B)      # anisotropic corrugation

Why this form (see the design note in the repo):
- The hypar base gives genuine negative Gaussian curvature everywhere and the
  classic doubly-curved saddle-canopy shape.
- A *pure* hypar has a globally-constant twist (d2z/dxdy = const), so its optimal
  planar-quad direction is globally obvious and uniform -- it does not force real
  planarization. The corrugation term injects a spatially-varying twist (it
  changes sign and magnitude across the field), so the local conjugate/asymptotic
  directions rotate across the surface. A single global mesh alignment can no
  longer planarize it; genuine per-panel optimization is required.

All lengths are in METERS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceParams:
    """Parameters of the reference surface. Frozen so a run is reproducible."""

    a: float = 20.0          # half-span in x  (full span 40 m)
    b: float = 10.0          # half-span in y  (full span 20 m)
    h1: float = 2.5          # hypar saddle amplitude (m) -- architectural scale
    h2: float = 0.6          # corrugation amplitude (m)
    p: float = 3.0           # corrugation waves across x
    q: float = 2.0           # corrugation waves across y

    @property
    def span_x(self) -> float:
        return 2.0 * self.a

    @property
    def span_y(self) -> float:
        return 2.0 * self.b


DEFAULT = SurfaceParams()


def z_of(x: float, y: float, prm: SurfaceParams = DEFAULT) -> float:
    """Height of the reference surface at plan point (x, y), in meters."""
    hypar = prm.h1 * (x / prm.a) * (y / prm.b)
    corr = prm.h2 * math.sin(prm.p * math.pi * x / prm.a) * math.cos(
        prm.q * math.pi * y / prm.b
    )
    return hypar + corr


def grad(x: float, y: float, prm: SurfaceParams = DEFAULT) -> tuple[float, float]:
    """Analytic (dz/dx, dz/dy)."""
    dzx = prm.h1 * y / (prm.a * prm.b) + prm.h2 * (
        prm.p * math.pi / prm.a
    ) * math.cos(prm.p * math.pi * x / prm.a) * math.cos(prm.q * math.pi * y / prm.b)
    dzy = prm.h1 * x / (prm.a * prm.b) - prm.h2 * (
        prm.q * math.pi / prm.b
    ) * math.sin(prm.p * math.pi * x / prm.a) * math.sin(prm.q * math.pi * y / prm.b)
    return dzx, dzy


def hessian(
    x: float, y: float, prm: SurfaceParams = DEFAULT
) -> tuple[float, float, float]:
    """Analytic (z_xx, z_yy, z_xy)."""
    px = prm.p * math.pi / prm.a
    qy = prm.q * math.pi / prm.b
    z_xx = -prm.h2 * px * px * math.sin(px * x) * math.cos(qy * y)
    z_yy = -prm.h2 * qy * qy * math.sin(px * x) * math.cos(qy * y)
    z_xy = prm.h1 / (prm.a * prm.b) - prm.h2 * px * qy * math.cos(px * x) * math.sin(
        qy * y
    )
    return z_xx, z_yy, z_xy


def gaussian_curvature(x: float, y: float, prm: SurfaceParams = DEFAULT) -> float:
    """Gaussian curvature K at (x, y). Negative => local saddle (anticlastic)."""
    z_xx, z_yy, z_xy = hessian(x, y, prm)
    dzx, dzy = grad(x, y, prm)
    denom = (1.0 + dzx * dzx + dzy * dzy) ** 2
    return (z_xx * z_yy - z_xy * z_xy) / denom
