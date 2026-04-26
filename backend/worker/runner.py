from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from core.objective import BOUNDS, BlackBoxObjective


def _load_user_optimize(solution_path: Path):
    spec = importlib.util.spec_from_file_location("user_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {solution_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "optimize", None)
    if fn is None or not callable(fn):
        raise RuntimeError("solution.py must define callable `optimize(objective, budget, bounds)`")
    return fn


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: runner.py <job_dir>\n")
        return 2

    job_dir = Path(argv[1]).resolve()
    result_path = job_dir / "result.json"
    progress_path = job_dir / "progress.jsonl"

    def _write_result(payload: dict[str, Any]) -> None:
        result_path.write_text(json.dumps(payload, indent=2))

    try:
        config = json.loads((job_dir / "config.json").read_text())
        with (job_dir / "trace.pkl").open("rb") as fh:
            trace = pickle.load(fh)
    except Exception as exc:
        _write_result({"ok": False, "error": f"setup failed: {exc}", "traceback": traceback.format_exc(), "n_trials": 0})
        return 1

    cap_raw = config.get("max_containers_cap")
    if cap_raw is None:
        k_hi = BOUNDS[1][1]
    else:
        k_hi = float(max(BOUNDS[1][0], min(float(cap_raw), BOUNDS[1][1])))
    bounds: list[tuple[float, float]] = [
        (BOUNDS[0][0], BOUNDS[0][1]),
        (BOUNDS[1][0], k_hi),
        (BOUNDS[2][0], BOUNDS[2][1]),
    ]

    obj = BlackBoxObjective(
        trace=trace,
        w_latency=float(config.get("w_latency", 0.5)),
        w_cost=float(config.get("w_cost", 0.5)),
        seed=int(config.get("seed", 0)),
        bounds=bounds,
        max_wait_ms=float(config.get("max_wait_ms", 0.0)),
    )
    budget = int(config["budget"])

    progress_fh = progress_path.open("w", buffering=1)
    state = {"trial": 0, "best_x": None, "best_y": float("inf"), "last_emit": time.time()}
    started_at = time.time()

    def wrapped(x: Any) -> float:
        state["trial"] += 1
        if state["trial"] > budget:
            raise RuntimeError(f"optimizer exceeded budget={budget}")
        x_list = [float(v) for v in np.asarray(x, dtype=float).ravel()]
        sim, y = obj.evaluate_with_y(x_list)
        if y < state["best_y"]:
            state["best_x"], state["best_y"] = x_list, y
        now = time.time()
        step_elapsed = now - state["last_emit"]
        state["last_emit"] = now
        latency_term, cost_term = obj.terms(sim)
        progress_fh.write(
            json.dumps(
                {
                    "trial": state["trial"],
                    "x": x_list,
                    "y": y,
                    "best_y": state["best_y"],
                    "elapsed_s": round(now - started_at, 3),
                    "step_s": round(step_elapsed, 3),
                    "n_containers": len(sim.container_summary),
                    "p99_latency_ms": sim.p99_latency_ms,
                    "cold_start_rate": sim.cold_start_rate,
                    "warm_hit_rate": (sim.warm_hits / sim.served) if sim.served else 0.0,
                    "idle_seconds": sim.idle_seconds,
                    "p_loss": sim.p_loss,
                    "latency_term": latency_term,
                    "cost_term": cost_term,
                    "container_summary": [c.to_dict() for c in sim.container_summary],
                }
            )
            + "\n"
        )
        return y

    try:
        optimize = _load_user_optimize(job_dir / "solution.py")
    except Exception as exc:
        progress_fh.close()
        _write_result({"ok": False, "error": str(exc), "traceback": traceback.format_exc(), "n_trials": 0})
        return 0

    try:
        optimize(wrapped, budget, bounds)
    except Exception as exc:
        progress_fh.close()
        _write_result(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "n_trials": state["trial"],
                "best_x": state["best_x"],
                "best_y": state["best_y"] if state["trial"] > 0 else None,
            }
        )
        return 0

    progress_fh.close()
    if state["best_x"] is None:
        _write_result({"ok": False, "error": "optimizer never called objective", "n_trials": 0})
        return 0

    final_metrics, final_y = obj.evaluate_with_y(state["best_x"])
    final_lat, final_cost = obj.terms(final_metrics)
    _write_result(
        {
            "ok": True,
            "best_x": state["best_x"],
            "best_y": state["best_y"],
            "final_y_resimulated": final_y,
            "n_trials": state["trial"],
            "elapsed_s": round(time.time() - started_at, 3),
            "best_metrics": {
                "p99_latency_ms": final_metrics.p99_latency_ms,
                "avg_latency_ms": final_metrics.avg_latency_ms,
                "cold_start_rate": final_metrics.cold_start_rate,
                "p_loss": final_metrics.p_loss,
                "idle_seconds": final_metrics.idle_seconds,
                "warm_hits": final_metrics.warm_hits,
                "cold_starts": final_metrics.cold_starts,
                "latency_term": final_lat,
                "cost_term": final_cost,
                "container_summary": [c.to_dict() for c in final_metrics.container_summary],
            },
            "normalization": {
                "l_max": obj.l_max,
                "c_max": obj.c_max,
                "w_latency": obj.w_latency,
                "w_cost": obj.w_cost,
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
