# PC Engines — Panelization Bench Eligibility

Wiring the Windows PC (`desktop-zt`, reached from the Mac over Tailscale/SSH) into the
planar-quad facade panelization bake-off. Owner: `pc-wirer`. Last updated 2026-08-15.

Benchmark task recap: a doubly-curved 40×20 m canopy panelized into ~1.5 m quads, every
panel planar within 5 mm, scored externally on **exported geometry**. An engine is
eligible if it can (a) be driven from the Mac and (b) author quad panels and export
geometry. This file records, per engine: install status, transport (+ exact tunnel
command), what is verified vs unverified, what needs Zohar's hands, and whether it can
actually *perform* the task or only *inspect*.

---

## Agent-reachability matrix (the setup-friction dimension)

"How much human action stands between the app being open and an agent driving it" is a
real benchmark dimension, not noise — it maps directly onto what's worth building. It is
**symmetric across the two Rhino machines** (so it can't explain a Mac-vs-PC difference)
but **asymmetric across engines** (so it's genuine signal for "which app is best driven by
an agent").

| Engine | Needs app open? | Zero-touch once app open? | First-load approval gate? | Transport | Can perform task? |
|---|---|---|---|---|---|
| **Rhino 8.30** (rhinomcp) | Yes | **No** — must type `mcpstart` each session | Yes — "load unsigned plug-in?" on first load of the 0.3.2 .rhp | TCP 1999 (loopback) → SSH tunnel | **Yes** — full geometry + export |
| **3ds Max 2026.3** (3dsmax-mcp) | Yes | **Yes** — `mcp_autostart.ms` starts listener on Max launch | Possible (unsigned .gup / script-exec security prompt) — unverified | Named pipe (default, local-only) **or** opt-in TCP 8765 (loopback) → SSH tunnel | **Yes** — create/edit poly, splines |
| **Cinema 4D 2026.2** (cinema4d-mcp) | Yes | **No** — Extensions ▸ Socket Server Plugin ▸ *Start Server* (GUI click) each session | Plugin-load prompt possible — unverified | TCP 5555 (loopback) → SSH tunnel | **Yes** *if it runs on 2026* — see open question |
| **AutoCAD 2026 — COM backend** (drive the *application*) | Yes | Yes (attaches via ProgID) | No | stdio on PC / HTTP :8000 | Yes, but needs AutoCAD running — UNVERIFIED |
| Revit 2027 (official MCP) | Yes | Yes | No | official preview (verified earlier) | **No — read-only.** Validator only |

**Not in the table above (it is not a PC engine):**

| Lane | Machine | What it drives | Transport | Can perform task? |
|---|---|---|---|---|
| **`ezdxf-headless` — CONTROL, library not application** | **machine-independent** (Mac / PC / CI) | a DXF-writing Python library, **not AutoCAD** | none — in-process Python (stdio MCP) | **Yes — VERIFIED headless.** See §4 |

Friction ranking among engines that actually **drive a running application**
(best→worst): **3ds Max (autostart) > Rhino (one `mcpstart` command) > Cinema 4D (GUI dialog
click)**. Blender (Mac lane) is autostart like Max. The `ezdxf-headless` control is
deliberately *outside* this ranking — it drives no application, so "reachability" doesn't
apply to it; that zero-friction is the whole point of a control (see §4).

---

## Transport principle (all socket engines)

Every app-side plugin here opens a **loopback-bound** TCP socket with **no authentication**
(arbitrary code execution over the wire — see `cad-cockpit/docs/research/mcp-security-posture.md`).
The correct way to reach these from the Mac is an **SSH tunnel over Tailscale**, never
rebinding the socket to `0.0.0.0`. This matches `cad-cockpit/docs/product/architecture.md`.
The Python MCP server runs on the Mac and connects to `127.0.0.1:<localport>`; the tunnel
carries that to the PC's loopback socket. Loopback on both ends, no auth exposed to the LAN.

```
Mac MCP server ── connects to 127.0.0.1:<localport> ── SSH tunnel ── PC 127.0.0.1:<appport> ── app plugin
```

---

## 1. Rhino 8.30 — HIGHEST VALUE (same engine, two machines)

**Install status: DONE.** PC plugin upgraded via yak to **rhinomcp 0.3.2**, matching the Mac
(`Yak.exe install rhinomcp 0.3.2`; stale 0.2.1 folder removed). `rhinomcp-mod 0.2.2`
(Mikhael Johanes fork) left untouched — separate fork, not part of the comparison.

**Why the version match matters:** the PC shipped the *legacy* plugin 0.2.1, which uses the
old **unframed** wire protocol. The current PyPI client (`uvx rhinomcp` → 0.3.2) switched to
**4-byte length-prefixed framing** and actively *rejects* a legacy plugin
("plugin predates length-prefixed framing. Update the plugin."). Left unfixed this is a
**silent failure that looks like "PC Rhino just doesn't work."** More important for the
bake-off: if the two machines ran different plugin versions you'd be comparing plugins, not
machines (0.3.2 also exposes far more tools, incl. the Grasshopper set). Both machines are
now 0.3.2 → the machine is the only variable.

**Transport (loopback both ends, no 0.0.0.0, no ALLOW_REMOTE):**
```bash
# tunnel: Mac localhost:2999 -> PC localhost:1999  (2999 avoids the Mac's own Rhino on 1999)
ssh -N -L 2999:127.0.0.1:1999 desktop-zt
# Mac-side MCP server instance, pointed through the tunnel:
RHINO_MCP_HOST=127.0.0.1  RHINO_MCP_PORT=2999  uvx rhinomcp
```
The client honors `RHINO_MCP_HOST`/`RHINO_MCP_PORT` (defaults 127.0.0.1:1999). No code changes.

**VERIFIED:**
- Transport end-to-end **with real bytes**: a loopback stub on PC `127.0.0.1:1999` +
  the tunnel + a Mac client on `localhost:2999` round-tripped real JSON; the PC saw the
  connection as loopback peer `127.0.0.1:56097`; `netstat` showed the PC socket bound to
  `127.0.0.1`, never `0.0.0.0`.
- Rhino launches headless over SSH via the interactive-scheduled-task pattern (see below):
  `Rhino.exe` ran in Console session 1, `Responding=True`.
- Plugin version parity with the Mac (both 0.3.2).

**UNVERIFIED (needs Zohar's hands once):** no real *Rhino* data through the tunnel yet.
rhinomcp does **not auto-load** — upstream requires typing `mcpstart` in Rhino once per
session (`mcpstop` ends it). Launching Rhino with `/runscript="_mcpstart"` did **not** bring
up the socket and the `rhinomcp` module never loaded into the process (`MainWindowTitle`
empty = blocked behind a modal). Most probable cause: the freshly yak-updated 0.3.2 `.rhp`
triggers Rhino's **"load unsigned plug-in?"** approval dialog on first load, which a headless
runscript can't click. Stated as most-likely, not confirmed (dialog not visually seen).

**One-time manual step for Zohar** → then fully agent-eligible:
1. Open Rhino once. If prompted "…not digitally signed. Load it?", click **Yes** (and set
   the plugin to load, so it doesn't re-ask).
2. Type `mcpstart`. Socket comes up on `127.0.0.1:1999`.
3. From then on: tunnel + `RHINO_MCP_PORT=2999 uvx rhinomcp` drives PC Rhino from the Mac,
   no further interaction that session. This `mcpstart` step is **identical on the Mac
   Rhino**, so it's fair — PC Rhino is fully eligible.

**Can it perform the task?** Yes — full geometry authoring + export (`execute_rhinoscript_python_code`,
`execute_rhinocommon_csharp_code`, Grasshopper tools, export). This is the strongest PC lane.

---

## 2. 3ds Max 2026.3 — cl0nazepamm/3dsmax-mcp (151 tools, MaxScript + Python)

**Install status: server installable headlessly; app-side needs Max.** MCP server installs
via `uv sync` then `uv run python install.py`. `install.py` (a) copies the per-year native
bridge `mcp_bridge_2026.gup` + `maxscript/mcp_server.ms` into Max's scripts folder and
`mcp_autostart.ms` into `scripts/startup/`, and (b) **also registers with AI-agent configs**.
Because (b) edits standing config, I did **not** run it unattended — recommend running it
deliberately on the PC, or staging just the file-copies. Supported: **Max 2023–2027** via
per-year `.gup` bridges (2026 bridge present). Not installed yet — this is a "stage it on the
PC" task, headless-safe for the file portion.

**Transport:** two options.
- **Named pipe** `\\.\pipe\3dsmax-mcp` — the DEFAULT; Windows-local IPC, **not tunnelable**
  across machines.
- **TCP fallback**, opt-in, `127.0.0.1:8765` — the `.ms` explicitly binds
  `IPAddress.Parse "127.0.0.1"` (loopback, verified in source). This is the one to use from
  the Mac:
```bash
ssh -N -L 8765:127.0.0.1:8765 desktop-zt   # then point the client at TCP transport, port 8765
```
The pipe-vs-TCP choice is a client transport flag; must select TCP for remote driving.

**Reachability:** once Max is open, `mcp_autostart.ms` starts the listener automatically —
**zero-touch** (like Blender, better than Rhino). A first-load security prompt for the
unsigned `.gup` / script execution is possible but unverified.

**Can it perform the task?** Yes — `create_object`, `draw_spline`, `edit_vertices`
(Editable Poly), MaxScript+Python. Export path exists via MaxScript. Not read-only.

**Unverified:** everything app-side (needs Max open on the PC). Launch Max the same way as
Rhino (interactive scheduled task) when ready to test.

---

## 3. Cinema 4D 2026.2 — ttiimmaacc/cinema4d-mcp — **2026 SUPPORT IS THE OPEN QUESTION**

**Answer: likely-loads, officially untested on 2026, cannot be verified headlessly.**

- **No hard version gate.** The plugin's only version check *warns* if
  `C4D_VERSION_MAJOR < 20`; it never refuses to load. C4D 2026 passes that check, so the
  plugin will not self-reject on 2026.
- **Official support matrix stops at "2025.0+"** and does **not list 2026**:
  R21/S22 (Py2.7) ❌ · 2023 (Py3.9) 🧪 in-progress · 2024 (Py3.11) ✅ verified ·
  **2025.0+ (Py3.11) ✅ primary target** · **2026 = unlisted / untested.**
- **Python-version risk unresolved:** 2024/2025 use Python **3.11**; the plugin targets that.
  I could not read C4D 2026.2's interpreter version off disk (it's bundled inside
  `.xdl64` modules, not a loose `pythonNN.dll`; a stale `Python-3.9.1` license file is present
  but is not authoritative). If 2026 stays on 3.11 the `.pyp` should load; a newer interpreter
  or c4d-API changes in 2026 could break specific tool calls. **Not determinable without C4D
  2026 open.**

**Transport:** TCP socket **`127.0.0.1:5555`**, loopback (plugin defaults `host="127.0.0.1",
port=5555`; server side reads `C4D_HOST`/`C4D_PORT`).
```bash
ssh -N -L 5555:127.0.0.1:5555 desktop-zt   # then C4D_HOST=127.0.0.1 C4D_PORT=5555 on the server
```

**Reachability — worst of the set:** manual plugin copy (`mcp_server_plugin.pyp` → C4D plugin
folder) **and** a per-session GUI action: *Extensions ▸ Socket Server Plugin ▸ Start Server*
(a dialog click, not even a command). Highest setup friction here.

**Can it perform the task?** In principle yes (`add_primitive`, `modify_object`,
`execute_python` for custom panelization), **but conditional on 2026 actually working.**

**To settle it (Zohar's hands):** copy the `.pyp` into C4D 2026.2's plugin folder, launch
C4D, Extensions ▸ Socket Server Plugin ▸ Start Server, then run the repo's CLI socket test
(`tests/`) or a `get_scene_info`. A clean "errors on 2026" is a legitimate finding.

---

## 4. `ezdxf-headless` — the CONTROL condition (NOT a PC engine, NOT "AutoCAD")

**Read this label carefully, because it changes what the result means.** The
`autocad-mcp-pro` package has two backends. Only its **COM** backend drives AutoCAD the
application (see §4b). Its **ezdxf** backend drives a **DXF-writing Python library** — `ezdxf`
— that happens to speak AutoCAD's file format. In the ezdxf configuration **the agent is not
driving AutoCAD at all**; there is no AutoCAD process, no `acad.exe`, no ACIS kernel, no
Autodesk anything. Calling this lane "AutoCAD" in the results would be wrong and would
mislead — it is a geometry library with an MCP wrapper.

**That is precisely why it is valuable — as the benchmark's control, not as a contender.**
It is the no-engine / no-GUI / no-friction baseline that isolates the *agent's geometric
reasoning* from any *engine's affordances*. Every other lane couples two unknowns: can the
model reason about conjugate-net planar-quad panelization, AND can it work the engine's MCP
surface. This lane removes the second unknown. The diagnostic it unlocks:

- **ezdxf scores well, GUI engines score badly** → the bottleneck is the MCP/engine
  interface, not the model's geometry skill. Points Zohar's build effort at transport + tooling.
- **ezdxf scores badly too** → the models can't yet do conjugate-net panelization, and no
  amount of better MCP fixes that. A completely different conclusion — worth knowing *before*
  building anything.

Without this control you cannot tell those two worlds apart. With it, the benchmark answers a
question it otherwise couldn't. **Record it as machine-independent (Mac / PC / CI), not a PC
engine.**

**Install status: DONE + tested on the Mac.** `pip install autocad-mcp-pro` → 1.5.1, pulled
`ezdxf 1.4.4`, `fastmcp`, `pydantic` — no `pywin32`/COM extras for this path. Backend selected
by `AUTOCAD_MCP_BACKEND=ezdxf`. The ezdxf backend advertises "headless DXF file operations";
the COM backend's own error text even points here as the grounded fallback. Supports
`LWPOLYLINE, POLYLINE, CIRCLE, ELLIPSE, SPLINE, HATCH, SOLID, TRACE, 3DFACE`, with
`drawing_new()` / `saveas`.

**VERIFIED with real output** (`pc/scripts/ezdxf_quad_smoke.py`, run under the venv, no
AutoCAD, on the Mac): built a doubly-curved patch, panelized into **28 quad panels** authored
as **3DFACE** entities, `saveas` → 22 KB DXF, reopened → all 28 3DFACE present.
`HEADLESS_EZDXF_OK`. (Naive sampling left ~266 mm as-built planarity — that gap *is* the
benchmark's planarization problem, not a tool limit; 3DFACE is the correct panel primitive and
authoring+export both work.)

**Transport:** none — in-process Python (stdio MCP server). No tunnel, no app, no licence, no
GUI, no per-session setup. Runs anywhere, including the Mac and CI.

**Known limit:** `ezdxf` has no ACIS kernel, so REGION/3DSOLID boolean-area ops are
unavailable — irrelevant here; planar-quad panels + DXF export are fully covered.

**Role in the bake-off:** **control / baseline**, machine-independent. Not a scored "AutoCAD"
result.

## 4b. AutoCAD 2026 — COM backend (the actual "drive the application" lane) — UNVERIFIED

The honest AutoCAD-the-application lane is `AUTOCAD_MCP_BACKEND=com`: live AutoCAD attached via
ProgID (`AutoCAD.Application`, configurable `CAD_PROGID`), Windows-only, **needs AutoCAD
running** on the PC. Transport is the stdio MCP server run on the PC, or HTTP on `:8000`
(COM is local, so it isn't SSH-tunnelled the way the loopback-socket engines are; tunnel the
HTTP port if driving remotely). This path can create geometry in a live drawing and `SaveAs`
DWG/DXF. **Not installed or verified** — needs AutoCAD open on the PC and the `[com]` extra
(`pywin32`) installed there. This is the lane to wire if you want a genuine "agent drives
AutoCAD" competitor distinct from the ezdxf control.

---

## 5. Revit 2027 — official Autodesk MCP — VALIDATOR ONLY

**Read-only; cannot perform the panelization task — do not enter it as a competitor.**
Verified working earlier this session set (discover instances, query model, get element data,
open views, select/zoom, export views). It cannot create walls/panels/geometry.

**Potential role: independent geometry *validator*.** Because it's a different vendor's engine
and read-only, it could import/host geometry produced elsewhere and independently report
element data / counts / bounds — a cross-check that the scoring isn't trusting the producing
engine's own report. Useful as an audit lane, not a scored lane. (Practical caveat: Revit
ingests BIM/IFC-style geometry, not raw quad meshes, so its value as a validator for a
mesh/DXF panelization output is limited; note it, don't over-invest.)

---

## Full stack block (for the per-run record)

| Component | Value |
|---|---|
| PC OS | Windows 11 (10.0.26200) |
| PC Rhino | 8.30.26103.11001 |
| PC Rhino plugin | rhinomcp **0.3.2** (yak) — matches Mac |
| Rhino MCP client | `uvx rhinomcp` → 0.3.2 (framed protocol) |
| Mac host | uv 0.12.5 / managed Python (client host) |
| PC Python | 3.14.4 + 3.13.14 |
| 3ds Max | 2026.3 — 3dsmax-mcp bridge `mcp_bridge_2026.gup`; TCP 8765 (loopback) / named pipe |
| Cinema 4D | 2026.2 — cinema4d-mcp plugin v0.1.9; TCP 5555 (loopback); 2026 support UNVERIFIED |
| `ezdxf-headless` control | autocad-mcp-pro **1.5.1** + ezdxf **1.4.4**, `AUTOCAD_MCP_BACKEND=ezdxf` — VERIFIED, machine-independent (NOT "AutoCAD") |
| AutoCAD (COM lane) | 2026 — `AUTOCAD_MCP_BACKEND=com`, needs AutoCAD running + `[com]`/pywin32 — UNVERIFIED |
| Revit | 2027 — official MCP, read-only (validator only) |

## Reusable: launching a GUI app over SSH on the PC

Plain SSH launch of a GUI app lands in the wrong session / vanishes. Use an **interactive
scheduled task** into the active console session (pattern from
`cad-cockpit/docs/workflows/revit-mcp-over-ssh.md`). Requires a logged-in console session
(there is one: `zohar_4ta16fp`). Always delete the temp task after.
```bat
schtasks /Create /TN BenchLaunchRhino8 /TR "\"C:\Program Files\Rhino 8\System\Rhino.exe\"" /SC ONCE /ST 23:59 /IT /F
schtasks /Run    /TN BenchLaunchRhino8
schtasks /Delete /TN BenchLaunchRhino8 /F
```
For args (e.g. `/runscript`), point `/TR` at a small `.cmd` launcher instead of inlining.

## Helper scripts written this session (`pc/scripts/`)
- `pc_stub_listener.py` — loopback TCP stub proving the tunnel with real bytes (fed to the
  PC via `ssh desktop-zt "python -" < pc_stub_listener.py`).
- `rhino_framed_client.py` — 0.3.2 framed-protocol client; run through the tunnel once
  `mcpstart` is active to pull real Rhino data (`get_document_info`).
- `ezdxf_quad_smoke.py` — headless planar-quad DXF proof (the `ezdxf-headless` control lane,
  not AutoCAD the app); run with the `autocad-mcp-pro` venv's python. Writes `panels_smoke.dxf`.
