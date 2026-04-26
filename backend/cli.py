from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from api.reporting import write_report
from api.sandbox import prepare_job_dir, start_sandbox, wait_sandbox
from core.types import RequestArrival
from datagen import PoissonGenerator, parse_user_csv


DEFAULT_TIMEOUT_S = 600


def _build_trace(args: argparse.Namespace) -> list[RequestArrival]:
    if args.source == "poisson":
        return PoissonGenerator().generate(
            intensity=args.intensity,
            duration_minutes=args.duration_minutes,
            n_functions=args.n_functions,
            seed=args.seed,
        )
    if args.source == "flow":
        from datagen.flow import FlowGenerator

        if args.dataset_id:
            root = Path(args.datasets_root).expanduser().resolve()
            w = root / args.dataset_id / "flow.pt"
            m = root / args.dataset_id / "flow_meta.json"
            if not w.is_file() or not m.is_file():
                raise SystemExit(
                    f"dataset {args.dataset_id} not found under {root}"
                )
            gen = FlowGenerator(weights_path=w, meta_path=m)
        else:
            if not FlowGenerator.is_available():
                raise SystemExit(
                    "flow source requested but shipped weights missing; "
                    "use --dataset-id or train flow weights first"
                )
            gen = FlowGenerator()
        return gen.generate(
            intensity=args.intensity,
            duration_minutes=args.duration_minutes,
            n_functions=args.n_functions,
            seed=args.seed,
        )
    if args.source == "upload":
        if not args.trace_csv:
            raise SystemExit("--source upload requires --trace-csv <path>")
        path = Path(args.trace_csv).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"trace csv not found: {path}")
        return parse_user_csv(path)
    raise SystemExit(f"unknown source: {args.source}")


def _read_new_lines(path: Path, sent: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], sent
    lines = path.read_text().splitlines()
    return lines[sent:], len(lines)


def _format_trial_line(row: dict[str, Any], budget: int) -> str:
    trial = row.get("trial", 0)
    y = row.get("y", 0.0)
    best_y = row.get("best_y", y)
    p99 = row.get("p99_latency_ms", 0.0) or 0.0
    cold = row.get("cold_start_rate", 0.0) or 0.0
    idle = row.get("idle_seconds", 0.0) or 0.0
    k = row.get("n_containers", 0) or 0
    step = row.get("step_s", 0.0) or 0.0
    return (
        f"[t={trial:>3}/{budget:<3}]  "
        f"f={y:7.4f}  best={best_y:7.4f}  "
        f"p99={p99:5.0f}ms  cold={cold * 100:4.1f}%  "
        f"idle={idle:6.0f}s  k={k:2d}  Δ={step:5.2f}s"
    )


def _print_final_block(report: dict[str, Any]) -> None:
    best_x = report.get("best_x") or []
    best_y = report.get("best_y")
    m = report.get("best_metrics") or {}
    config = report.get("config") or {}
    norm = report.get("normalization") or {}
    baselines = report.get("baselines") or []
    elapsed = report.get("elapsed_s")

    best_x_fmt = "[" + ", ".join(f"{float(v):.3f}" for v in best_x) + "]"
    print()
    print(
        f"=== RUN DONE · trials={report.get('n_trials')} · total={elapsed}s ==="
    )
    print(f"best_x   = {best_x_fmt}")
    print(f"best_y   = {best_y:.4f}" if isinstance(best_y, (int, float)) else f"best_y   = {best_y}")
    p99 = float(m.get("p99_latency_ms", 0.0))
    cold = float(m.get("cold_start_rate", 0.0))
    ploss = float(m.get("p_loss", 0.0))
    idle = float(m.get("idle_seconds", 0.0))
    print(
        f"metrics  : p99={p99:.0f}ms  cold={cold * 100:.1f}%  "
        f"p_loss={ploss * 100:.2f}%  idle={idle:.0f}s"
    )
    lat_term = m.get("latency_term")
    cost_term = m.get("cost_term")
    w_lat = float(config.get("w_latency", norm.get("w_latency", 0.5)))
    w_cost = float(config.get("w_cost", norm.get("w_cost", 0.5)))
    if lat_term is not None and cost_term is not None:
        print(
            f"breakdown: latency_term={float(lat_term):.4f}  "
            f"cost_term={float(cost_term):.4f}  "
            f"(w_lat={w_lat}, w_cost={w_cost})"
        )

    if baselines:
        print()
        print(
            "Baseline comparison (y = w_lat·p99/L_max + w_cost·idle/C_max):"
        )
        header = f"  {'policy':<14}{'y':>10}{'Δ vs best':>12}{'p99':>9}{'cold':>8}{'idle':>9}"
        print(header)
        best_val = float(best_y) if isinstance(best_y, (int, float)) else float("nan")
        row0 = (
            f"  {'your best':<14}{best_val:>10.4f}{'—':>12}"
            f"{p99:>7.0f}ms{cold * 100:>6.1f}%{idle:>7.0f}s"
        )
        print(row0)
        for b in baselines:
            by = float(b.get("y", 0.0))
            delta = (by - best_val) / best_val * 100.0 if best_val > 0 else 0.0
            bm = b.get("metrics") or {}
            bp99 = float(bm.get("p99_latency_ms", 0.0))
            bcold = float(bm.get("cold_start_rate", 0.0))
            bidle = float(bm.get("idle_seconds", 0.0))
            print(
                f"  {str(b.get('name', '?')):<14}{by:>10.4f}"
                f"{delta:>+11.1f}%"
                f"{bp99:>7.0f}ms{bcold * 100:>6.1f}%{bidle:>7.0f}s"
            )


def _stream_and_wait_subprocess(
    *, job_dir: Path, progress_path: Path, budget: int, json_mode: bool
) -> None:
    proc = start_sandbox(job_dir)
    sent = 0
    try:
        while True:
            if proc.poll() is not None:
                break
            new_lines, sent = _read_new_lines(progress_path, sent)
            if not json_mode:
                for ln in new_lines:
                    if not ln.strip():
                        continue
                    try:
                        row = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    print(_format_trial_line(row, budget), flush=True)
            time.sleep(0.1)
        new_lines, sent = _read_new_lines(progress_path, sent)
        if not json_mode:
            for ln in new_lines:
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                print(_format_trial_line(row, budget), flush=True)
    finally:
        sandbox_result = wait_sandbox(proc, job_dir, timeout_s=DEFAULT_TIMEOUT_S)
    if sandbox_result.status != "done":
        err = sandbox_result.result.get("error") if sandbox_result.result else None
        tail = sandbox_result.stderr_tail or sandbox_result.stdout_tail
        raise SystemExit(
            f"run ended with status={sandbox_result.status}\n"
            f"error: {err}\n"
            f"stderr tail:\n{tail}"
        )


def _run_inline(*, job_dir: Path, progress_path: Path, budget: int, json_mode: bool) -> None:
    import threading

    from worker.runner import main as runner_main

    thread = threading.Thread(
        target=runner_main,
        args=([sys.argv[0], str(job_dir.resolve())],),
        daemon=True,
    )
    thread.start()
    sent = 0
    while thread.is_alive():
        new_lines, sent = _read_new_lines(progress_path, sent)
        if not json_mode:
            for ln in new_lines:
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                print(_format_trial_line(row, budget), flush=True)
        time.sleep(0.05)
    thread.join()
    new_lines, sent = _read_new_lines(progress_path, sent)
    if not json_mode:
        for ln in new_lines:
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            print(_format_trial_line(row, budget), flush=True)

    result_path = job_dir / "result.json"
    if not result_path.exists():
        raise SystemExit("inline runner failed to write result.json")
    result = json.loads(result_path.read_text())
    if not result.get("ok"):
        raise SystemExit(
            f"inline runner reported failure: {result.get('error')}\n"
            f"{result.get('traceback', '')}"
        )


def _cmd_run(args: argparse.Namespace) -> int:
    solution_path = Path(args.solution).expanduser().resolve()
    if not solution_path.is_file():
        raise SystemExit(f"solution file not found: {solution_path}")
    solution_source = solution_path.read_text()

    trace = _build_trace(args)
    if not trace:
        raise SystemExit("generated trace is empty — nothing to optimize on")

    if args.output_dir:
        job_dir = Path(args.output_dir).expanduser().resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_root = Path(tempfile.gettempdir()) / "serverless_blackbox_cli"
        tmp_root.mkdir(parents=True, exist_ok=True)
        job_dir = tmp_root / uuid.uuid4().hex[:12]
        job_dir.mkdir(parents=True, exist_ok=False)

    prepare_job_dir(
        job_dir,
        solution_source=solution_source,
        trace=trace,
        budget=args.budget,
        w_latency=args.w_latency,
        w_cost=args.w_cost,
        seed=args.seed,
        max_containers_cap=args.max_containers_cap,
        max_wait_ms=args.max_wait_ms,
    )
    progress_path = job_dir / "progress.jsonl"
    progress_path.touch()

    if not args.json:
        print(
            f"# run_id={job_dir.name}  source={args.source}  arrivals={len(trace)}  "
            f"budget={args.budget}  w_lat={args.w_latency}  w_cost={args.w_cost}  seed={args.seed}",
            flush=True,
        )

    if args.no_sandbox:
        _run_inline(
            job_dir=job_dir,
            progress_path=progress_path,
            budget=args.budget,
            json_mode=args.json,
        )
    else:
        _stream_and_wait_subprocess(
            job_dir=job_dir,
            progress_path=progress_path,
            budget=args.budget,
            json_mode=args.json,
        )

    report = write_report(
        job_dir=job_dir,
        trace=trace,
        w_latency=args.w_latency,
        w_cost=args.w_cost,
        seed=args.seed,
        max_wait_ms=args.max_wait_ms,
    )

    if args.json:
        json.dump(report, sys.stdout)
        sys.stdout.write("\n")
    else:
        _print_final_block(report)
        print(f"\nArtifacts: {job_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cli",
        description="Run the black-box serverless optimizer without the frontend.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run one optimizer end-to-end.")
    r.add_argument("--solution", required=True, help="Path to user solution.py defining optimize(...)")
    r.add_argument(
        "--source",
        choices=["poisson", "flow", "upload"],
        default="poisson",
    )
    r.add_argument("--intensity", type=float, default=0.5)
    r.add_argument("--duration-minutes", type=int, default=60, dest="duration_minutes")
    r.add_argument("--n-functions", type=int, default=10, dest="n_functions")
    r.add_argument("--budget", type=int, default=30)
    r.add_argument("--w-latency", type=float, default=0.5, dest="w_latency")
    r.add_argument("--w-cost", type=float, default=0.5, dest="w_cost")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument(
        "--max-containers-cap",
        type=int,
        default=None,
        dest="max_containers_cap",
        help="upper bound the optimizer is allowed to pick for Policy.max_containers (1..30)",
    )
    r.add_argument(
        "--max-wait-ms",
        type=float,
        default=30000.0,
        dest="max_wait_ms",
        help="hyperparameter: per-request waiting-queue timeout in ms (default 30000)",
    )
    r.add_argument("--trace-csv", default=None, dest="trace_csv", help="per-arrival CSV for --source upload")
    r.add_argument("--dataset-id", default=None, dest="dataset_id", help="user-trained flow weights id")
    r.add_argument(
        "--datasets-root",
        default=str(Path(tempfile.gettempdir()) / "serverless_blackbox_runs" / "datasets"),
        dest="datasets_root",
    )
    r.add_argument("--output-dir", default=None, dest="output_dir")
    r.add_argument("--no-sandbox", action="store_true", help="inline run (no subprocess, no rlimits)")
    r.add_argument("--json", action="store_true", help="suppress live log; print report.json to stdout")
    r.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
