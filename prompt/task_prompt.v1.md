# Panelization task — v1

You are given a doubly-curved architectural canopy surface and must panelize it for
fabrication. This is a real facade constraint: flat glass is cheap, curved glass is
not, so every panel must be flat enough to build from flat stock.

## The surface

A saddle canopy spanning **40 m × 20 m** in plan (x ∈ [−20, 20] m, y ∈ [−10, 10] m).
It is provided two equivalent ways — use whichever suits your engine:

- **Import** the mesh `reference/canopy_reference.obj` (units: **meters**).
- **Rebuild** it natively from the exact closed form (units: meters):

  ```
  z(x, y) = 2.5 · (x/20) · (y/10)  +  0.6 · sin(3π·x/20) · cos(2π·y/10)
  ```

## What to produce

1. **Panelize the whole surface into quadrilateral panels at roughly a 1.5 m module.**
   Quads, not triangles. Cover the entire surface.
2. **Every panel must be planar to within 5 mm** — i.e. its four corners must lie
   within 5 mm of a common plane. This is the binding constraint.
3. **Export the panelized geometry** to a neutral format — `.obj`, `.ply`, `.dxf`,
   or `.3dm` — in **meters**. State the file path.
4. **Report**, in your final answer:
   - the total panel count,
   - the **maximum planarity deviation** across all panels, in millimeters,
   - **how many panels exceed the 5 mm tolerance**.

## Notes

- The panels stay flat only if the panel layout is chosen well; a naïve grid on a
  doubly-curved surface will not be planar. How you achieve planarity is up to you
  and your engine.
- Do not subdivide below the ~1.5 m module to make planarity trivial, and do not
  flatten or shrink the surface — the exported panels must still follow the canopy.
- State the units of your export explicitly.

Scoring is performed externally on your exported file; only the geometry you export
is judged.
