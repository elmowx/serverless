from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from datagen.upload import parse_user_csv


FLOW_N_EPOCHS = 8
FLOW_N_LAYERS = 4
FLOW_HIDDEN = (24, 24)
FLOW_SAMPLE_SIZE = 500
TRAIN_TIMEOUT_S = 300


class FitFlowError(RuntimeError):
    pass


def aggregate_to_minutes(csv_path: Path, out_path: Path) -> int:
    arrivals = parse_user_csv(csv_path)
    buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
    for a in arrivals:
        minute = a.timestamp_ms // 60_000
        buckets[(minute, a.function_id)].append(a.execution_time_ms)
    if not buckets:
        raise FitFlowError("no arrivals in CSV")
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["minute", "function_id", "count", "avg_exec_time_ms"])
        for (minute, fid), xs in sorted(buckets.items()):
            w.writerow([minute, fid, len(xs), sum(xs) / len(xs)])
    return len(buckets)


def parse_result_trailer(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("[train_flow] RESULT "):
            return json.loads(line[len("[train_flow] RESULT "):])
    raise FitFlowError("train_flow did not emit RESULT trailer")


async def fit_flow(
    csv_bytes: bytes,
    datasets_root: Path,
    *,
    n_epochs: int = FLOW_N_EPOCHS,
    n_layers: int = FLOW_N_LAYERS,
    hidden: tuple[int, ...] = FLOW_HIDDEN,
) -> dict[str, Any]:
    if not csv_bytes:
        raise FitFlowError("empty CSV")
    dataset_id = hashlib.sha256(csv_bytes).hexdigest()[:16]
    dataset_dir = datasets_root / dataset_id
    weights_path = dataset_dir / "flow.pt"
    meta_path = dataset_dir / "flow_meta.json"

    if weights_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return {
            "dataset_id": dataset_id,
            "cached": True,
            "passed": True,
            "ks_p_count": float(meta.get("ks_p_count", 0.0)),
            "ks_p_exec": float(meta.get("ks_p_exec", 0.0)),
            "n_aggregated_rows": int(meta.get("n_training_rows", 0)),
        }

    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dataset_dir / "trace.csv"
    agg_path = dataset_dir / "aggregated.csv"
    raw_path.write_bytes(csv_bytes)
    n_rows = aggregate_to_minutes(raw_path, agg_path)

    cmd = [
        sys.executable, "-m", "datagen.train_flow",
        "--data", str(agg_path),
        "--n-epochs", str(n_epochs),
        "--n-layers", str(n_layers),
        "--hidden", *[str(h) for h in hidden],
        "--sample-size", str(FLOW_SAMPLE_SIZE),
        "--output-dir", str(dataset_dir),
        "--weights-name", "flow.pt",
        "--meta-name", "flow_meta.json",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=TRAIN_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise FitFlowError(f"training timed out after {TRAIN_TIMEOUT_S}s") from exc

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    (dataset_dir / "train.log").write_text(stdout)

    try:
        trailer = parse_result_trailer(stdout)
    except FitFlowError as exc:
        raise FitFlowError(f"{exc}; last 20 log lines:\n" + "\n".join(stdout.splitlines()[-20:])) from None

    return {
        "dataset_id": dataset_id,
        "cached": False,
        "passed": bool(trailer["passed"]),
        "ks_p_count": float(trailer["ks_p_count"]),
        "ks_p_exec": float(trailer["ks_p_exec"]),
        "n_aggregated_rows": n_rows,
    }
