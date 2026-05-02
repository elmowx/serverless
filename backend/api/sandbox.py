from __future__ import annotations

import json
import logging
import os
import pickle
import resource
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import RequestArrival


MEM_LIMIT_BYTES = 512 * 1024 * 1024
CPU_LIMIT_SECONDS = 150
FILE_LIMIT = 64
PROC_LIMIT = 32
DEFAULT_WALL_TIMEOUT_S = 90

BACKEND_DIR = Path(__file__).resolve().parents[1]

_log = logging.getLogger(__name__)
if sys.platform == "darwin":
    _log.warning(
        "RLIMIT_AS is a no-op on Darwin; sandbox memory cap %.0f MB is not enforced "
        "outside Linux/Docker. CPU and wall-clock limits still apply.",
        MEM_LIMIT_BYTES / (1024 * 1024),
    )


def _preexec() -> None:
    mem_limits = [
        getattr(resource, "RLIMIT_AS", None),
        getattr(resource, "RLIMIT_DATA", None),
    ]
    for lim in mem_limits:
        if lim is None:
            continue
        try:
            resource.setrlimit(lim, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS + 5))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (FILE_LIMIT, FILE_LIMIT))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (PROC_LIMIT, PROC_LIMIT))
    except (ValueError, OSError, AttributeError):
        pass
    os.setpgrp()


@dataclass
class SandboxResult:
    status: str
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    result: dict[str, Any] | None = None
    job_dir: Path | None = None
    progress_path: Path | None = None


def prepare_job_dir(
    job_dir: Path,
    solution_source: str,
    trace: list[RequestArrival],
    *,
    budget: int,
    w_latency: float = 0.5,
    w_cost: float = 0.5,
    seed: int = 0,
    max_containers_cap: int | None = None,
    max_wait_ms: float = 0.0,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "solution.py").write_text(solution_source)
    with (job_dir / "trace.pkl").open("wb") as fh:
        pickle.dump(trace, fh)
    cfg: dict[str, Any] = {
        "budget": int(budget),
        "w_latency": float(w_latency),
        "w_cost": float(w_cost),
        "seed": int(seed),
        "max_wait_ms": float(max_wait_ms),
    }
    if max_containers_cap is not None:
        cfg["max_containers_cap"] = int(max_containers_cap)
    (job_dir / "config.json").write_text(json.dumps(cfg))


def start_sandbox(
    job_dir: Path,
    *,
    python_executable: str | None = None,
) -> subprocess.Popen:
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(job_dir),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(BACKEND_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    cmd = [
        python_executable or sys.executable,
        "-m",
        "worker.runner",
        str(job_dir.resolve()),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(job_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=_preexec,
    )


def wait_sandbox(
    proc: subprocess.Popen,
    job_dir: Path,
    *,
    timeout_s: int = DEFAULT_WALL_TIMEOUT_S,
) -> SandboxResult:
    status = "done"
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = b"", b""
        status = "timeout"

    result_path = job_dir / "result.json"
    progress_path = job_dir / "progress.jsonl"
    parsed: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            parsed = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            parsed = None

    if status != "timeout":
        if proc.returncode != 0 and parsed is None:
            status = "crashed"
        elif parsed is not None and not parsed.get("ok", False):
            status = "user_error"
        else:
            status = "done"

    return SandboxResult(
        status=status,
        exit_code=proc.returncode,
        stdout_tail=stdout.decode("utf-8", errors="replace")[-4096:],
        stderr_tail=stderr.decode("utf-8", errors="replace")[-4096:],
        result=parsed,
        job_dir=job_dir,
        progress_path=progress_path if progress_path.exists() else None,
    )


def run_sandbox(
    job_dir: Path,
    *,
    timeout_s: int = DEFAULT_WALL_TIMEOUT_S,
    python_executable: str | None = None,
) -> SandboxResult:
    proc = start_sandbox(job_dir, python_executable=python_executable)
    return wait_sandbox(proc, job_dir, timeout_s=timeout_s)


def cleanup_job_dir(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)
