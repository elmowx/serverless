from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from api.jobs import JobManager
from api.sandbox import SandboxResult
from api.store import RunStore
from core.types import RequestArrival


@pytest.mark.asyncio
async def test_completed_tasks_removed_from_jobmanager(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runs.sqlite")
    jobs = JobManager(store)

    def fake_sandbox(job_dir, *, timeout_s, python_executable):
        return SandboxResult(
            status="done",
            exit_code=0,
            stdout_tail="",
            stderr_tail="",
            result={"ok": True, "best_x": [1.0, 1.0, 1.0], "best_y": 0.0, "n_trials": 1, "elapsed_s": 0.0},
            job_dir=job_dir,
        )

    def fake_write_report(*, job_dir, trace, w_latency, w_cost, seed, max_wait_ms=0.0):
        return {}

    monkeypatch.setattr("api.jobs.run_sandbox", fake_sandbox)
    monkeypatch.setattr("api.jobs.write_report", fake_write_report)

    arrivals = [RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=10.0)]

    for i in range(3):
        run_id = f"run{i}"
        job_dir = tmp_path / run_id
        store.create(run_id=run_id, config={"budget": 1}, job_dir=job_dir)
        await jobs.submit(
            run_id=run_id,
            job_dir=job_dir,
            solution_source="def optimize(o,b,bnds):\n  o([1,1,1])\n  return [1,1,1]\n",
            trace=arrivals,
            budget=1,
            w_latency=0.5,
            w_cost=0.5,
            seed=0,
        )

    deadline = asyncio.get_event_loop().time() + 5.0
    while jobs._tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)

    assert jobs._tasks == {}, (
        f"completed tasks must be removed from JobManager._tasks, got {list(jobs._tasks)}"
    )
