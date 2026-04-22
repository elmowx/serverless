"""
CLI smoke tests. Runs the same example optimizer as the API tests, with a
tiny budget / trace so they complete in < 5 s.

We invoke the CLI via ``cli.main(argv)`` so pytest picks up coverage and
avoids the overhead / flakiness of spawning a full ``python -m cli`` in
subprocess. The subprocess path inside the CLI (``run_sandbox``) is still
exercised by the happy-path test.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from cli import main as cli_main


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_SOLUTION = EXAMPLE_DIR / "user_optimizer_random.py"


def _common_args(solution: Path, output_dir: Path, *, extra: list[str] | None = None) -> list[str]:
    return [
        "run",
        "--solution",
        str(solution),
        "--source",
        "poisson",
        "--intensity",
        "0.2",
        "--duration-minutes",
        "2",
        "--n-functions",
        "3",
        "--budget",
        "3",
        "--seed",
        "0",
        "--output-dir",
        str(output_dir),
        *(extra or []),
    ]


def test_cli_happy_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = cli_main(
            _common_args(EXAMPLE_SOLUTION, out_dir, extra=["--no-sandbox"])
        )
    assert exit_code == 0
    text = buf.getvalue()
    assert "RUN DONE" in text
    assert "best_x" in text
    assert "Baseline comparison" in text
    report_path = out_dir / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["best_x"] is not None
    assert report["best_y"] is not None
    assert len(report["baselines"]) == 5


def test_cli_json_mode_stdout_is_valid_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = cli_main(
            _common_args(EXAMPLE_SOLUTION, out_dir, extra=["--no-sandbox", "--json"])
        )
    assert exit_code == 0
    payload = json.loads(buf.getvalue())
    assert payload["best_x"] is not None
    assert isinstance(payload["best_y"], float)
    assert len(payload["baselines"]) == 5
    assert "convergence" in payload


def test_cli_user_error_returns_nonzero(tmp_path: Path) -> None:
    bad_solution = tmp_path / "bad.py"
    bad_solution.write_text("def not_optimize(a, b, c):\n    pass\n")
    out_dir = tmp_path / "run"
    buf_err = io.StringIO()
    # The CLI reports failures via ``SystemExit("...")``. --no-sandbox path
    # raises from the inline runner; subprocess path raises from the
    # wait-and-check. Use --no-sandbox so the assertion works cross-platform.
    with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf_err):
        cli_main(
            _common_args(bad_solution, out_dir, extra=["--no-sandbox"])
        )
    msg = str(excinfo.value)
    assert "optimize" in msg.lower() or "ok" in msg.lower()


def test_cli_subprocess_sandbox_path(tmp_path: Path) -> None:
    """Exercises the real subprocess/Popen path end-to-end (no --no-sandbox).

    This is the one CLI test that actually forks ``worker.runner``, mirroring
    what ``python -m cli run ...`` does on the command line.
    """
    out_dir = tmp_path / "run"
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = cli_main(_common_args(EXAMPLE_SOLUTION, out_dir))
    assert exit_code == 0
    report = json.loads((out_dir / "report.json").read_text())
    assert report["best_x"] is not None
    # The sandbox subprocess should have written progress.jsonl with >= 1 line.
    progress = (out_dir / "progress.jsonl").read_text().strip().splitlines()
    assert len(progress) >= 1
    last = json.loads(progress[-1])
    # Full payload from the updated runner is present.
    assert "container_summary" in last
    assert "p99_latency_ms" in last


# Sanity: we import from the backend package directly, the example file must
# exist at the documented location.
def test_example_solution_present() -> None:
    assert EXAMPLE_SOLUTION.is_file(), f"missing example at {EXAMPLE_SOLUTION}"
    assert sys.version_info >= (3, 11)
