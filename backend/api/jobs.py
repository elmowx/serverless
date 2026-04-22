"""
Async job orchestration.

A `JobManager` owns a bounded pool (semaphore) of concurrent sandbox runs. The
actual sandbox executes in a thread via `asyncio.to_thread` — the subprocess
itself is blocking, which is fine since we cap concurrency and each run has a
hard timeout.

On completion, the manager augments the raw `result.json` written by
`worker.runner` with baseline comparisons and convergence data, producing
`report.json` consumed by `GET /runs/{id}/report`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from api.reporting import write_report
from api.sandbox import SandboxResult, prepare_job_dir, run_sandbox
from api.store import RunStore
from core.types import RequestArrival


MAX_CONCURRENT_RUNS = 2
DEFAULT_TIMEOUT_S = 120


class JobManager:
    def __init__(self, store: RunStore, *, concurrency: int = MAX_CONCURRENT_RUNS) -> None:
        self._store = store
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(
        self,
        *,
        run_id: str,
        job_dir: Path,
        solution_source: str,
        trace: list[RequestArrival],
        budget: int,
        w_latency: float,
        w_cost: float,
        seed: int,
        max_containers_cap: int | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        prepare_job_dir(
            job_dir,
            solution_source=solution_source,
            trace=trace,
            budget=budget,
            w_latency=w_latency,
            w_cost=w_cost,
            seed=seed,
            max_containers_cap=max_containers_cap,
        )
        task = asyncio.create_task(
            self._run(
                run_id=run_id,
                job_dir=job_dir,
                trace=trace,
                w_latency=w_latency,
                w_cost=w_cost,
                seed=seed,
                timeout_s=timeout_s,
            )
        )
        self._tasks[run_id] = task

    async def _run(
        self,
        *,
        run_id: str,
        job_dir: Path,
        trace: list[RequestArrival],
        w_latency: float,
        w_cost: float,
        seed: int,
        timeout_s: int,
    ) -> None:
        async with self._sem:
            self._store.set_status(run_id, "running")
            result: SandboxResult = await asyncio.to_thread(
                run_sandbox,
                job_dir,
                timeout_s=timeout_s,
                python_executable=sys.executable,
            )
            error = None
            if result.status == "timeout":
                error = f"run exceeded {timeout_s}s wall timeout"
            elif result.status == "crashed":
                error = (result.stderr_tail or result.stdout_tail or "process crashed")[-512:]
            elif result.status == "user_error" and result.result:
                error = result.result.get("error")

            if result.status == "done" and result.result and result.result.get("ok"):
                await asyncio.to_thread(
                    write_report,
                    job_dir=job_dir,
                    trace=trace,
                    w_latency=w_latency,
                    w_cost=w_cost,
                    seed=seed,
                )
            self._store.mark_finished(
                run_id,
                status=result.status,
                exit_code=result.exit_code,
                error=error,
            )

    async def wait(self, run_id: str, timeout: float | None = None) -> None:
        task = self._tasks.get(run_id)
        if task is None:
            return
        await asyncio.wait_for(task, timeout=timeout)
