from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from api.sandbox import cleanup_job_dir, prepare_job_dir, run_sandbox
from datagen import PoissonGenerator

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="sandbox relies on POSIX fork + rlimits")


@pytest.fixture(scope="module")
def small_trace():
    return PoissonGenerator().generate(intensity=0.2, duration_minutes=5, n_functions=4, seed=0)


def _write_and_run(tmp_path: Path, small_trace, source: str, *, budget: int = 5, timeout_s: int = 30):
    job_dir = tmp_path / "job"
    prepare_job_dir(job_dir, solution_source=dedent(source), trace=small_trace, budget=budget, seed=0)
    try:
        return run_sandbox(job_dir, timeout_s=timeout_s, python_executable=sys.executable)
    finally:
        pass


def test_valid_optimizer_round_trips(tmp_path: Path, small_trace):
    src = """
        import random
        def optimize(objective, budget, bounds):
            rng = random.Random(0)
            best_x, best_y = None, float("inf")
            for _ in range(budget):
                x = [rng.uniform(lo, hi) for lo, hi in bounds]
                y = objective(x)
                if y < best_y:
                    best_x, best_y = x, y
            return best_x
    """
    res = _write_and_run(tmp_path, small_trace, src, budget=4)
    assert res.status == "done", res
    assert res.result is not None and res.result["ok"] is True
    assert res.result["n_trials"] == 4
    assert len(res.result["best_x"]) == 3
    assert res.progress_path is not None
    lines = [json.loads(ln) for ln in res.progress_path.read_text().splitlines()]
    assert len(lines) == 4
    assert all("trial" in ln and "x" in ln and "y" in ln for ln in lines)
    cleanup_job_dir(res.job_dir)


def test_missing_optimize_function_reports_error(tmp_path: Path, small_trace):
    src = "x = 42  # no optimize() here"
    res = _write_and_run(tmp_path, small_trace, src, budget=3)
    assert res.status == "user_error"
    assert res.result is not None and res.result["ok"] is False
    assert "optimize" in res.result["error"].lower()
    cleanup_job_dir(res.job_dir)


def test_user_exception_surfaces_traceback(tmp_path: Path, small_trace):
    src = """
        def optimize(objective, budget, bounds):
            objective([600, 5, 0.5])
            raise RuntimeError("boom")
    """
    res = _write_and_run(tmp_path, small_trace, src, budget=3)
    assert res.status == "user_error"
    assert res.result["ok"] is False
    assert "boom" in res.result["error"]
    assert res.result["n_trials"] == 1
    cleanup_job_dir(res.job_dir)


def test_infinite_loop_hit_wall_timeout(tmp_path: Path, small_trace):
    src = """
        def optimize(objective, budget, bounds):
            while True:
                pass
    """
    res = _write_and_run(tmp_path, small_trace, src, budget=5, timeout_s=3)
    assert res.status == "timeout"
    assert res.exit_code != 0
    cleanup_job_dir(res.job_dir)


def test_memory_bomb_terminated(tmp_path: Path, small_trace):
    src = """
        def optimize(objective, budget, bounds):
            buf = []
            chunk = b'x' * (64 * 1024 * 1024)
            for _ in range(100):
                buf.append(chunk * 1)
            return [0, 0, 0]
    """
    res = _write_and_run(tmp_path, small_trace, src, budget=3, timeout_s=20)
    assert res.status in {"user_error", "crashed", "timeout"}
    assert res.exit_code != 0 or (res.result is not None and not res.result["ok"])
    cleanup_job_dir(res.job_dir)


def test_budget_exceeded_caught(tmp_path: Path, small_trace):
    src = """
        def optimize(objective, budget, bounds):
            for _ in range(budget + 3):
                objective([600, 5, 0.5])
    """
    res = _write_and_run(tmp_path, small_trace, src, budget=2)
    assert res.status == "user_error"
    assert "budget" in res.result["error"].lower()
    cleanup_job_dir(res.job_dir)


def test_network_syscall_is_not_seccomp_blocked(tmp_path: Path, small_trace):
    """Documented limitation: the bare-OS sandbox applies rlimits but has no
    seccomp/netns/iptables filter, so ``socket.connect`` is NOT blocked —
    it only fails when the OS can't route the packet (e.g. connect to
    127.0.0.1 on a closed port yields ConnectionRefusedError).

    This test pins that behavior on purpose. When we move isolation into
    Docker (or add a netns wrapper), this test should flip: the user code
    will fail much earlier with PermissionError/EACCES before the connect
    round-trip. Failing this test is a signal that isolation semantics
    changed and PROGRESS.md's "Known Limitations" section needs an edit.
    """
    src = """
        import socket
        def optimize(objective, budget, bounds):
            objective([600, 5, 0.5])
            # port 1 is privileged; unbound locally -> ConnectionRefusedError.
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(("127.0.0.1", 1))
    """
    res = _write_and_run(tmp_path, small_trace, src, budget=3, timeout_s=10)
    assert res.status == "user_error"
    assert res.result is not None and res.result["ok"] is False
    err = (res.result.get("error") or "").lower()
    # Exact errno wording varies by OS/libc; on Linux+macOS both hit one
    # of these. The point is: the failure is a normal Python exception
    # from the socket, not a sandbox-level kill.
    assert any(
        tok in err
        for tok in ("refused", "timed out", "timeout", "unreachable", "connection")
    ), f"unexpected socket error surface: {err!r}"
    cleanup_job_dir(res.job_dir)
