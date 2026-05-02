from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np

from api.reporting import build_report, _pareto_points
from core.types import RequestArrival
from datagen import PoissonGenerator


def _heavy_trace() -> list[RequestArrival]:
    return PoissonGenerator().generate(
        intensity=0.95, duration_minutes=10, n_functions=4, seed=0
    )


def _seed_job_dir(tmp_path: Path, trace: list[RequestArrival]) -> Path:
    job_dir = tmp_path / "run"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "ok": True,
                "best_x": [60.0, 3.0, 1.0],
                "best_y": 0.5,
                "n_trials": 1,
                "elapsed_s": 0.1,
                "best_metrics": {
                    "p99_latency_ms": 0.0,
                    "avg_latency_ms": 0.0,
                    "cold_start_rate": 0.0,
                    "p_loss": 0.0,
                    "idle_seconds": 0.0,
                    "latency_term": 0.0,
                    "cost_term": 0.0,
                },
            }
        )
    )
    (job_dir / "progress.jsonl").write_text(
        json.dumps(
            {
                "trial": 1,
                "x": [60.0, 3.0, 1.0],
                "y": 0.5,
                "best_y": 0.5,
                "p99_latency_ms": 0.0,
                "idle_seconds": 0.0,
            }
        )
        + "\n"
    )
    with (job_dir / "trace.pkl").open("wb") as fh:
        pickle.dump(trace, fh)
    return job_dir


def test_baselines_respect_max_wait_ms(tmp_path: Path) -> None:
    trace = _heavy_trace()
    job_a = _seed_job_dir(tmp_path / "a", trace)
    job_b = _seed_job_dir(tmp_path / "b", trace)

    rep_loss = build_report(
        job_dir=job_a,
        trace=trace,
        w_latency=0.5,
        w_cost=0.5,
        seed=0,
        max_wait_ms=0.0,
    )
    rep_queue = build_report(
        job_dir=job_b,
        trace=trace,
        w_latency=0.5,
        w_cost=0.5,
        seed=0,
        max_wait_ms=30_000.0,
    )

    minimal_loss = next(b for b in rep_loss["baselines"] if b["name"] == "minimal")
    minimal_queue = next(b for b in rep_queue["baselines"] if b["name"] == "minimal")

    assert minimal_loss["metrics"]["p_loss"] != minimal_queue["metrics"]["p_loss"], (
        "max_wait_ms must affect baseline simulations; "
        f"loss-mode p_loss={minimal_loss['metrics']['p_loss']} "
        f"queue-mode p_loss={minimal_queue['metrics']['p_loss']}"
    )
    assert minimal_loss["metrics"]["p_loss"] >= minimal_queue["metrics"]["p_loss"], (
        "loss queue (max_wait_ms=0) should reject at least as many as a "
        "30s-tolerant queue on a saturated trace"
    )

    assert rep_loss["config"]["max_wait_ms"] == 0.0
    assert rep_queue["config"]["max_wait_ms"] == 30_000.0


def test_pareto_built_from_progress_without_resim(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"trial": 1, "x": [600.0, 10.0, 1.0], "y": 0.4, "p99_latency_ms": 100.0, "idle_seconds": 2000.0},
                {"trial": 2, "x": [30.0, 3.0, 1.0], "y": 0.6, "p99_latency_ms": 150.0, "idle_seconds": 2500.0},
                {"trial": 3, "x": [60.0, 5.0, 1.0], "y": 0.5, "p99_latency_ms": 200.0, "idle_seconds": 1000.0},
            ]
        )
        + "\n"
    )

    with patch("api.reporting.BlackBoxObjective") as patched_obj:
        result = _pareto_points(progress)
    assert patched_obj.call_count == 0, (
        "_pareto_points must not instantiate BlackBoxObjective; got "
        f"{patched_obj.call_count} calls"
    )

    assert [(round(p["p99_latency_ms"]), round(p["idle_seconds"])) for p in result] == [
        (100, 2000),
        (200, 1000),
    ]


def test_build_report_uses_stored_baselines_without_simulator(tmp_path: Path) -> None:
    job_dir = tmp_path / "run"
    job_dir.mkdir(parents=True)
    stored_baselines = [
        {
            "name": "minimal",
            "policy": {"keep_alive_s": 60.0, "max_containers": 3, "prewarm_threshold": 1.0},
            "y": 0.42,
            "metrics": {
                "p99_latency_ms": 999.0,
                "avg_latency_ms": 100.0,
                "cold_start_rate": 0.5,
                "p_loss": 0.01,
                "idle_seconds": 5.0,
                "latency_term": 0.9,
                "cost_term": 0.1,
            },
        }
    ]
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "ok": True,
                "best_x": [60.0, 3.0, 1.0],
                "best_y": 0.5,
                "n_trials": 1,
                "elapsed_s": 0.1,
                "best_metrics": {"latency_term": 0.0, "cost_term": 0.0, "p99_latency_ms": 0.0, "cold_start_rate": 0.0, "p_loss": 0.0, "idle_seconds": 0.0},
                "baselines": stored_baselines,
                "normalization": {"l_max": 1234.0, "c_max": 5678.0, "w_latency": 0.5, "w_cost": 0.5},
            }
        )
    )
    (job_dir / "progress.jsonl").write_text(
        json.dumps({"trial": 1, "x": [60.0, 3.0, 1.0], "y": 0.5, "p99_latency_ms": 0.0, "idle_seconds": 0.0}) + "\n"
    )

    with patch("api.reporting.BlackBoxObjective") as patched_obj, patch(
        "api.reporting.run_sim"
    ) as patched_run:
        report = build_report(
            job_dir=job_dir,
            trace=[RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=10.0)],
            w_latency=0.5,
            w_cost=0.5,
            seed=0,
            max_wait_ms=0.0,
        )

    assert patched_obj.call_count == 0, "build_report must read baselines from result.json"
    assert patched_run.call_count == 1, "_best_policy_timeline must call simulator exactly once"
    assert report["baselines"] == stored_baselines
    assert report["normalization"] == {"l_max": 1234.0, "c_max": 5678.0, "w_latency": 0.5, "w_cost": 0.5}
