"""Tests for the run harness: transcript summarization, the run-level verdict
taxonomy (each of the five branches), stack probing, and the ezdxf end-to-end
selftest that must land on FALSE_SUCCESS."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner  # noqa: E402


def _rec(ts, tool, ok=True):
    return runner.ToolCall(ts=ts, tool=tool, args_summary="x", ok=ok)


# --------------------------------------------------------------------------- #
# Transcript summarization.
# --------------------------------------------------------------------------- #
def test_summarize_counts_and_first_success():
    t0 = 1000.0
    recs = [_rec(t0 + 0.0, "a", ok=False), _rec(t0 + 0.5, "a", ok=True),
            _rec(t0 + 1.0, "b", ok=True)]
    for r in recs:
        r.duration_ms = 10.0
    s = runner.summarize_transcript(recs, run_start=t0)
    assert s["total_tool_calls"] == 3
    assert s["successful_tool_calls"] == 2
    assert s["failed_tool_calls"] == 1
    assert s["time_to_first_successful_tool_call_ms"] == 510.0  # first OK completes at t0+0.51
    assert s["per_tool"]["a"]["errors"] == 1


# --------------------------------------------------------------------------- #
# Run-level verdict taxonomy -- all five branches.
# --------------------------------------------------------------------------- #
def _cat(successful):
    return {"successful_tool_calls": successful}


def test_verdict_never_reached():
    assert runner.assign_run_verdict(_cat(0), None, True, True) == "NEVER_REACHED_ENGINE"


def test_verdict_task_failed_no_geometry():
    assert runner.assign_run_verdict(_cat(3), {"verdict": "EMPTY"}, False, False) \
        == "REACHED_TASK_FAILED"


def test_verdict_false_success():
    geo = {"verdict": "PLANARITY_FAIL"}
    assert runner.assign_run_verdict(_cat(3), geo, True, False) == "FALSE_SUCCESS"


def test_verdict_reached_task_failed_when_not_claimed():
    geo = {"verdict": "PLANARITY_FAIL"}
    assert runner.assign_run_verdict(_cat(3), geo, False, False) == "REACHED_TASK_FAILED"


def test_verdict_pass_requires_self_check():
    geo = {"verdict": "PASS"}
    assert runner.assign_run_verdict(_cat(3), geo, True, True) == "PASS"
    assert runner.assign_run_verdict(_cat(3), geo, True, False) == "PASS_NO_SELF_CHECK"


# --------------------------------------------------------------------------- #
# Stack probe -- control lane resolves; unknown lane fails loud.
# --------------------------------------------------------------------------- #
def test_stack_probe_ezdxf_resolves():
    stack, warns = runner.probe_stack("ezdxf-control", "some-model")
    assert "ezdxf" in stack["plugin_or_backend"]
    assert "unknown" not in stack["plugin_or_backend"].lower()
    assert stack["driving_model"] == "some-model"


def test_stack_probe_unknown_lane_is_loud():
    stack, warns = runner.probe_stack("nonexistent-lane", "m")
    assert stack["mcp_server"] == "unknown"
    assert any("UNKNOWN" in w for w in warns)  # visible, not omitted


# --------------------------------------------------------------------------- #
# End-to-end: the ezdxf control lane must expose the motivating failure.
# --------------------------------------------------------------------------- #
def test_selftest_end_to_end_is_false_success(tmp_path):
    rec = runner.selftest(tmp_path / "runs")
    assert rec["run_verdict"] == "FALSE_SUCCESS"          # claimed success, failed geometry
    assert rec["geometry_verdict"] == "PLANARITY_FAIL"
    assert rec["trilemma_corner"] == "QUALITY+CONFORMING (not planar)"
    assert rec["reachability"]["requires_host_app_running"] is False
    assert rec["reachability"]["human_steps_to_ready"] == 0
    assert rec["stack_warnings"] == []                    # control lane fully attributable
    assert rec["catalog"]["successful_tool_calls"] == 3
