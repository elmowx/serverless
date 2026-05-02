from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.baselines import BASELINES, with_max_wait
from core.objective import BlackBoxObjective, policy_from_vector
from core.simulator import run as run_sim
from core.types import RequestArrival


TIMELINE_MAX_CONTAINERS = 20
TIMELINE_MAX_SEGMENTS = 600


def compute_baselines(
    obj: BlackBoxObjective, max_wait_ms: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, raw in BASELINES.items():
        bound = with_max_wait(raw, max_wait_ms)
        sim = obj._simulate(bound)
        x = [bound.keep_alive_s, float(bound.max_containers), bound.prewarm_threshold]
        y = float(obj(np.asarray(x, dtype=float)))
        lat_term, cost_term = obj.terms(sim)
        out.append(
            {
                "name": name,
                "policy": {
                    "keep_alive_s": bound.keep_alive_s,
                    "max_containers": bound.max_containers,
                    "prewarm_threshold": bound.prewarm_threshold,
                },
                "y": y,
                "metrics": {
                    "p99_latency_ms": sim.p99_latency_ms,
                    "avg_latency_ms": sim.avg_latency_ms,
                    "cold_start_rate": sim.cold_start_rate,
                    "p_loss": sim.p_loss,
                    "idle_seconds": sim.idle_seconds,
                    "latency_term": lat_term,
                    "cost_term": cost_term,
                },
            }
        )
    return out


def build_report(
    *,
    job_dir: Path,
    trace: list[RequestArrival],
    w_latency: float,
    w_cost: float,
    seed: int,
    max_wait_ms: float = 0.0,
) -> dict[str, Any]:
    result = json.loads((job_dir / "result.json").read_text())
    best_metrics = result.get("best_metrics") or {}
    best_x = result.get("best_x")

    baselines = result.get("baselines")
    normalization = result.get("normalization") or {}
    if baselines is None:
        obj = BlackBoxObjective(
            trace=trace,
            w_latency=w_latency,
            w_cost=w_cost,
            seed=seed,
            max_wait_ms=max_wait_ms,
        )
        baselines = compute_baselines(obj, max_wait_ms)
        if not normalization:
            normalization = {
                "l_max": obj.l_max,
                "c_max": obj.c_max,
                "w_latency": w_latency,
                "w_cost": w_cost,
            }

    convergence = _parse_convergence(job_dir / "progress.jsonl")
    pareto = _pareto_points(job_dir / "progress.jsonl")
    timeline_payload = _best_policy_timeline(best_x, trace, seed)

    return {
        "best_x": best_x,
        "best_y": result.get("best_y"),
        "best_metrics": best_metrics,
        "n_trials": result.get("n_trials"),
        "elapsed_s": result.get("elapsed_s"),
        "baselines": baselines,
        "convergence": convergence,
        "pareto_points": pareto,
        "container_timeline": timeline_payload,
        "config": {
            "w_latency": w_latency,
            "w_cost": w_cost,
            "seed": seed,
            "max_wait_ms": max_wait_ms,
        },
        "normalization": normalization,
    }


def write_report(
    *,
    job_dir: Path,
    trace: list[RequestArrival],
    w_latency: float,
    w_cost: float,
    seed: int,
    max_wait_ms: float = 0.0,
) -> dict[str, Any]:
    report = build_report(
        job_dir=job_dir,
        trace=trace,
        w_latency=w_latency,
        w_cost=w_cost,
        seed=seed,
        max_wait_ms=max_wait_ms,
    )
    (job_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _parse_convergence(progress_path: Path) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []
    out: list[dict[str, Any]] = []
    best = float("inf")
    for ln in progress_path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        best = min(best, float(row["y"]))
        out.append(
            {
                "trial": row["trial"],
                "y": row["y"],
                "best_so_far": best,
                "n_containers": row.get("n_containers"),
                "cold_start_rate": row.get("cold_start_rate"),
            }
        )
    return out


def _best_policy_timeline(
    best_x: list[float] | None,
    trace: list[RequestArrival],
    seed: int,
) -> dict[str, Any] | None:
    if best_x is None:
        return None
    policy = policy_from_vector(best_x)
    rng = np.random.default_rng(seed)
    res = run_sim(trace, policy, rng=rng, record_timeline=True)
    tracks: list[dict[str, Any]] = []
    for cid, segments in enumerate(res.container_timeline[:TIMELINE_MAX_CONTAINERS]):
        merged = _merge_same_state(segments)
        thinned = _thin_segments(merged, TIMELINE_MAX_SEGMENTS)
        tracks.append(
            {
                "container_id": cid,
                "segments": [
                    {"state": s.state, "t0_ms": s.t0_ms, "t1_ms": s.t1_ms}
                    for s in thinned
                ],
            }
        )
    return {
        "total_ms": res.timeline_end_ms,
        "n_containers": len(res.container_timeline),
        "shown_containers": len(tracks),
        "tracks": tracks,
    }


def _merge_same_state(segments):
    if not segments:
        return segments
    out = [segments[0]]
    for s in segments[1:]:
        last = out[-1]
        if s.state == last.state and abs(s.t0_ms - last.t1_ms) < 1e-6:
            out[-1] = type(s)(state=last.state, t0_ms=last.t0_ms, t1_ms=s.t1_ms)
        else:
            out.append(s)
    return out


def _thin_segments(segments, max_n: int):
    if len(segments) <= max_n:
        return segments
    segs = list(segments)
    while len(segs) > max_n:
        i = min(
            range(len(segs)),
            key=lambda j: segs[j].t1_ms - segs[j].t0_ms,
        )
        if i == 0:
            nb = segs[1]
            segs[1] = type(nb)(state=nb.state, t0_ms=segs[i].t0_ms, t1_ms=nb.t1_ms)
            segs.pop(0)
        else:
            nb = segs[i - 1]
            segs[i - 1] = type(nb)(state=nb.state, t0_ms=nb.t0_ms, t1_ms=segs[i].t1_ms)
            segs.pop(i)
    return segs


def _pareto_points(progress_path: Path) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []
    visited: list[tuple[list[float], float, float]] = []
    for ln in progress_path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        x = row.get("x")
        p99 = row.get("p99_latency_ms")
        idle = row.get("idle_seconds")
        if x is None or p99 is None or idle is None:
            continue
        visited.append((list(x), float(p99), float(idle)))

    visited.sort(key=lambda v: (v[1], v[2]))
    frontier: list[tuple[list[float], float, float]] = []
    best_idle = float("inf")
    for x, p99, idle in visited:
        if idle < best_idle:
            frontier.append((x, p99, idle))
            best_idle = idle
    return [{"x": x, "p99_latency_ms": p99, "idle_seconds": idle} for x, p99, idle in frontier]
