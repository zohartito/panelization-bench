"""runner.py -- drive the panelization task against each engine lane, log every
tool call, and assemble a per-run record whose fields are kept deliberately
UN-merged so a reader can separate "which engine authored better geometry" from
"which engine an agent can actually reach".

A run record carries four independent blocks:

  stack         exact version of every moving part (MCP server, host app, plugin,
                driving model). Version skew silently corrupts a comparison, so an
                undeterminable part is a VISIBLE "unknown", never omitted.
  reachability  the setup-friction axis -- human_steps_to_ready,
                requires_host_app_running, time_to_first_successful_tool_call.
                An engine an agent cannot reach without a human is worse at being
                driven, even if its geometry is better; reported ALONGSIDE the
                score, never blended in.
  catalog       the tool-call transcript distilled: per-tool counts, errors,
                timing, dead-ends. This is the product -- the failure catalog that
                becomes the spec for the MCP servers Zohar will build.
  geometry      score.py's neutral verdict on the exported artifact.

Run-level verdict taxonomy (distinct on purpose):
  NEVER_REACHED_ENGINE   harness never got a successful tool call (transport/setup)
  REACHED_TASK_FAILED    engine reached; agent errored / gave up / no valid export
  FALSE_SUCCESS          agent declared success but geometry FAILED scoring
                         (the motivating case: panelize, declare victory, never measure)
  PASS_NO_SELF_CHECK     geometry passes, but the agent never measured planarity
                         (ranks below PASS -- may be luck, not competence)
  PASS                   geometry passes AND the agent measured/reported planarity

The ezdxf-control lane drives a DXF-writing library, NOT an application. It is the
zero-friction control that isolates the agent's geometric reasoning from engine
affordances -- and, because it needs no app open, it lets this whole harness be
exercised end-to-end right now (see `selftest`).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import score

SCHEMA = "panelization-bench.run.v1"
HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "canopy_reference.obj"
SURFACE_DEF = HERE / "reference" / "surface_def.py"


# --------------------------------------------------------------------------- #
# Lane registry: reachability facts (the setup-friction axis).
# --------------------------------------------------------------------------- #
LANES: dict[str, dict] = {
    "ezdxf-control": {
        "kind": "CONTROL -- library, not application",
        "requires_host_app_running": False,
        "human_steps_to_ready": 0,
        "steps": ["none -- in-process Python (ezdxf), no app, no GUI, no licence"],
        "first_load_approval_gate": False,
    },
    "blender": {
        "kind": "application",
        "requires_host_app_running": True,
        "human_steps_to_ready": 0,
        "steps": ["open Blender (the blender-mcp addon auto-starts the listener)"],
        "first_load_approval_gate": False,
    },
    "rhino-community": {
        "kind": "application",
        "requires_host_app_running": True,
        "human_steps_to_ready": 1,
        "steps": ["open Rhino", "type `mcpstart` once per session"],
        "first_load_approval_gate": True,  # unsigned-plugin dialog on first load
    },
    "rhino-mcneel": {
        "kind": "application",
        "requires_host_app_running": True,
        "human_steps_to_ready": 1,
        "steps": ["open Rhino", "type `mcpstart` once per session"],
        "first_load_approval_gate": True,
    },
}


# --------------------------------------------------------------------------- #
# Tool-call transcript.
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    ts: float                      # wall-clock start
    tool: str
    args_summary: str
    result_summary: str = ""
    duration_ms: float = 0.0
    ok: bool = True
    error: str = ""

    def as_dict(self) -> dict:
        return {"ts": self.ts, "tool": self.tool, "args_summary": self.args_summary,
                "result_summary": self.result_summary, "duration_ms": self.duration_ms,
                "ok": self.ok, "error": self.error}


class ToolCallLogger:
    """Append-only JSONL transcript. The harness wraps every MCP call with
    `with logger.call(tool, args) as rec:` so timing/ok/error are captured
    uniformly regardless of which engine served the call."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.records: list[ToolCall] = []
        self.run_start = time.time()
        self.run_end = None
        if path:
            path.write_text("")
            self._save_timing()

    def _save_timing(self):
        if self.path:
            self.path.with_suffix(self.path.suffix + ".timing.json").write_text(
                json.dumps({"run_start": self.run_start, "run_end": self.run_end}))

    def finish(self):
        """Persist the end after setup, calls and any final export work."""
        self.run_end = time.time()
        self._save_timing()

    class _Ctx:
        def __init__(self, logger, rec):
            self.logger, self.rec, self._t = logger, rec, time.time()

        def __enter__(self):
            return self

        def result(self, summary: str = "", ok: bool = True, error: str = ""):
            self.rec.result_summary, self.rec.ok, self.rec.error = summary, ok, error

        def __exit__(self, exc_type, exc, tb):
            self.rec.duration_ms = (time.time() - self._t) * 1000.0
            if exc_type is not None:
                self.rec.ok, self.rec.error = False, f"{exc_type.__name__}: {exc}"
            self.logger._commit(self.rec)
            return False

    def call(self, tool: str, args_summary: str) -> "ToolCallLogger._Ctx":
        return self._Ctx(self, ToolCall(ts=time.time(), tool=tool,
                                        args_summary=args_summary))

    def _commit(self, rec: ToolCall) -> None:
        self.records.append(rec)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec.as_dict()) + "\n")


def load_transcript(path: Path) -> list[ToolCall]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(ToolCall(**d))
    return out


def summarize_transcript(records: list[ToolCall], run_start: float,
                         run_end: float | None = None) -> dict:
    """Distill the transcript into the failure catalog."""
    per_tool: dict[str, dict] = {}
    first_ok_ms = None
    for r in records:
        t = per_tool.setdefault(r.tool, {"calls": 0, "errors": 0, "total_ms": 0.0})
        t["calls"] += 1
        t["total_ms"] += r.duration_ms
        if not r.ok:
            t["errors"] += 1
        if r.ok:
            completed_ms = (r.ts - run_start) * 1000.0 + r.duration_ms
            first_ok_ms = completed_ms if first_ok_ms is None else min(first_ok_ms, completed_ms)
    errors = [{"tool": r.tool, "args": r.args_summary, "error": r.error}
              for r in records if not r.ok]
    return {
        "total_tool_calls": len(records),
        "successful_tool_calls": sum(1 for r in records if r.ok),
        "failed_tool_calls": len(errors),
        "time_to_first_successful_tool_call_ms": first_ok_ms,
        "wall_time_ms": ((run_end - run_start) * 1000.0 if run_end is not None else
                         max(((r.ts - run_start) * 1000.0 + r.duration_ms
                              for r in records), default=0.0)),
        "per_tool": per_tool,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Stack block -- fail loud on anything undeterminable.
# --------------------------------------------------------------------------- #
def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _app_version(app_path: Path) -> str:
    plist = app_path / "Contents" / "Info.plist"
    if not plist.exists():
        return "unknown"
    try:
        import plistlib
        data = plistlib.loads(plist.read_bytes())
        return str(data.get("CFBundleShortVersionString", "unknown"))
    except Exception:
        return "unknown"


def _rhino_plugin_version() -> str:
    base = Path.home() / ("Library/Application Support/McNeel/Rhinoceros/"
                          "packages/8.0")
    if not base.exists():
        return "unknown"
    hits = sorted(p.name for p in base.iterdir()
                  if p.is_dir() and "rhinomcp" in p.name.lower())
    return hits[-1] if hits else "unknown"


def probe_stack(lane: str, model: str) -> tuple[dict, list[str]]:
    """Best-effort local version probe. Anything undeterminable is a VISIBLE
    'unknown' plus a warning -- an unattributable run is an anecdote, not evidence."""
    stack = {"lane": lane, "driving_model": model or "unknown",
             "host": platform.platform()}
    if lane == "ezdxf-control":
        stack.update({
            "mcp_server": f"autocad-mcp-pro {_pkg_version('autocad-mcp-pro')}",
            "host_app": "none (headless library)",
            "plugin_or_backend": f"ezdxf {_pkg_version('ezdxf')}",
        })
    elif lane == "blender":
        stack.update({
            "mcp_server": "blender-mcp (uvx; resolve at runtime)",
            "host_app": f"Blender {_app_version(Path('/Applications/Blender.app'))}",
            "plugin_or_backend": "blender-mcp addon (bl_info at runtime) -- unknown",
        })
    elif lane in ("rhino-community", "rhino-mcneel"):
        stack.update({
            "mcp_server": "rhinomcp / RhinoMCP (resolve at runtime)",
            "host_app": f"Rhino {_app_version(Path('/Applications/Rhino 8.app'))}",
            "plugin_or_backend": f"rhino plugin {_rhino_plugin_version()}",
        })
    else:
        stack.update({"mcp_server": "unknown", "host_app": "unknown",
                      "plugin_or_backend": "unknown"})
    warnings = [f"stack.{k} is UNKNOWN -- run is not fully attributable"
                for k, v in stack.items() if isinstance(v, str) and "unknown" in v.lower()]
    return stack, warnings


# --------------------------------------------------------------------------- #
# Verdict assignment.
# --------------------------------------------------------------------------- #
def assign_run_verdict(catalog: dict, geometry: dict | None,
                       agent_claimed_success: bool,
                       agent_reported_planarity: bool) -> str:
    if catalog["successful_tool_calls"] == 0:
        return "NEVER_REACHED_ENGINE"
    if geometry is None or geometry.get("verdict") == "EMPTY":
        return "REACHED_TASK_FAILED"
    if geometry["verdict"] == "PASS":
        return "PASS" if agent_reported_planarity else "PASS_NO_SELF_CHECK"
    # geometry failed
    return "FALSE_SUCCESS" if agent_claimed_success else "REACHED_TASK_FAILED"


# --------------------------------------------------------------------------- #
# Assemble a full run record.
# --------------------------------------------------------------------------- #
def evaluate_run(lane: str, model: str, records: list[ToolCall], run_start: float,
                 mesh_path: Path | None, agent_claimed_success: bool,
                 agent_reported_planarity: bool, units: str = "m",
                 reference: Path = REFERENCE, surface_def: Path = SURFACE_DEF,
                 run_end: float | None = None) -> dict:
    stack, stack_warnings = probe_stack(lane, model)
    catalog = summarize_transcript(records, run_start, run_end)

    geometry = None
    if mesh_path and Path(mesh_path).exists():
        ref = score.load_mesh(reference) if Path(reference).exists() else None
        surf = score.load_surface_z(surface_def) if Path(surface_def).exists() else None
        geometry = score.score_mesh(score.load_mesh(Path(mesh_path)), units=units,
                                    reference=ref, surface_z=surf)

    reach = dict(LANES.get(lane, {"kind": "unknown"}))
    reach["time_to_first_successful_tool_call_ms"] = \
        catalog["time_to_first_successful_tool_call_ms"]

    verdict = assign_run_verdict(catalog, geometry, agent_claimed_success,
                                 agent_reported_planarity)
    return {
        "schema": SCHEMA,
        "run_start": run_start,
        "run_end": run_end,
        "lane": lane,
        "run_verdict": verdict,
        "geometry_verdict": geometry["verdict"] if geometry else None,
        "trilemma_corner": geometry.get("trilemma_corner") if geometry else None,
        "agent_claimed_success": agent_claimed_success,
        "agent_reported_planarity": agent_reported_planarity,
        "stack": stack,
        "stack_warnings": stack_warnings,
        "reachability": reach,
        "catalog": catalog,
        "geometry": geometry,
    }


# --------------------------------------------------------------------------- #
# ezdxf control lane -- runnable now, no app. Used to de-risk the whole harness.
# --------------------------------------------------------------------------- #
def _ezdxf_weak_agent(logger: ToolCallLogger, out_dxf: Path) -> None:
    """A scripted WEAK agent for the ezdxf control lane: panelizes the reference
    surface on the naive UV grid, writes 3DFACE panels, and never measures
    planarity. Used only to exercise the harness end-to-end (a stand-in for a real
    LLM agent), NOT a benchmark result."""
    import ezdxf
    sys.path.insert(0, str(HERE / "reference"))
    import surface_def as sd

    prm = sd.DEFAULT
    nx, ny = 27, 13
    with logger.call("ezdxf.new", "create empty DXF document") as c:
        doc = ezdxf.new()
        msp = doc.modelspace()
        c.result("doc created")
    xs = [-prm.a + 2 * prm.a * i / nx for i in range(nx + 1)]
    ys = [-prm.b + 2 * prm.b * j / ny for j in range(ny + 1)]
    P = [[(x, y, sd.z_of(x, y, prm)) for x in xs] for y in ys]
    with logger.call("ezdxf.add_3dface", f"author {nx*ny} quad panels") as c:
        for j in range(ny):
            for i in range(nx):
                msp.add_3dface([P[j][i], P[j][i + 1], P[j + 1][i + 1], P[j + 1][i]])
        c.result(f"{nx*ny} 3DFACE panels")
    with logger.call("ezdxf.saveas", f"save {out_dxf.name}") as c:
        doc.saveas(str(out_dxf))
        c.result(f"wrote {out_dxf.name}")


def selftest(workdir: Path, model: str = "harness-selftest") -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    transcript = workdir / "selftest.jsonl"
    out_dxf = workdir / "selftest_panels.dxf"
    logger = ToolCallLogger(transcript)
    run_start = logger.run_start
    _ezdxf_weak_agent(logger, out_dxf)
    logger.finish()
    # the weak agent claims success but never measured planarity
    return evaluate_run("ezdxf-control", model, logger.records, run_start, out_dxf,
                        agent_claimed_success=True, agent_reported_planarity=False,
                        run_end=logger.run_end)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _print_record(rec: dict) -> None:
    print(f"RUN VERDICT: {rec['run_verdict']}   lane={rec['lane']}")
    print(f"  geometry: {rec['geometry_verdict']}  |  trilemma: {rec['trilemma_corner']}")
    r = rec["reachability"]
    print(f"  reachability: {r.get('kind')} | host-app required "
          f"{r.get('requires_host_app_running')} | human steps "
          f"{r.get('human_steps_to_ready')} | first-tool-call "
          f"{r.get('time_to_first_successful_tool_call_ms')} ms")
    s = rec["stack"]
    print(f"  stack: server={s.get('mcp_server')} | app={s.get('host_app')} | "
          f"plugin={s.get('plugin_or_backend')} | model={s.get('driving_model')}")
    for w in rec["stack_warnings"]:
        print(f"  STACK WARNING: {w}")
    for warning in rec.get("timing_warnings", []):
        print(f"  TIMING WARNING: {warning}")
    c = rec["catalog"]
    print(f"  catalog: {c['successful_tool_calls']}/{c['total_tool_calls']} tool "
          f"calls ok, {c['failed_tool_calls']} errors, wall {c['wall_time_ms']:.0f} ms")
    if rec["geometry"]:
        print("  --- geometry ---")
        print("  " + score.human_summary(rec["geometry"]).replace("\n", "\n  "))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Panelization-bench run harness.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-lanes")

    ev = sub.add_parser("evaluate", help="assemble a run record from a transcript+mesh")
    ev.add_argument("--lane", required=True, choices=list(LANES))
    ev.add_argument("--model", default="unknown")
    ev.add_argument("--transcript", type=Path, required=True)
    ev.add_argument("--mesh", type=Path, default=None)
    ev.add_argument("--units", default="m")
    ev.add_argument("--claimed-success", action="store_true")
    ev.add_argument("--reported-planarity", action="store_true")
    ev.add_argument("--json", type=Path, default=None)

    st = sub.add_parser("selftest", help="drive the ezdxf control lane end-to-end")
    st.add_argument("--workdir", type=Path,
                    default=HERE / ".runs" / "selftest")
    st.add_argument("--json", type=Path, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "list-lanes":
        for name, info in LANES.items():
            print(f"{name}: {info['kind']} | host-app "
                  f"{info['requires_host_app_running']} | human steps "
                  f"{info['human_steps_to_ready']} ({'; '.join(info['steps'])})")
        return 0

    if args.cmd == "evaluate":
        records = load_transcript(args.transcript)
        timing_path = args.transcript.with_suffix(args.transcript.suffix + ".timing.json")
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        run_start = timing.get("run_start", min((r.ts for r in records), default=time.time()))
        run_end = timing.get("run_end")
        rec = evaluate_run(args.lane, args.model, records, run_start, args.mesh,
                           args.claimed_success, args.reported_planarity, args.units,
                           run_end=run_end)
        rec["timing_warnings"] = []
        if not timing or run_end is None:
            rec["timing_warnings"].append(
                "Incomplete run timing metadata: elapsed time uses available call bounds; "
                "unrecorded setup/finalization time is unknown.")
    else:  # selftest
        rec = selftest(args.workdir)

    if getattr(args, "json", None):
        args.json.write_text(json.dumps(rec, indent=2))
    _print_record(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
