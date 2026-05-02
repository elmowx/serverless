"""
Tests for the user-upload flow training endpoint pipeline
(`api.flow_training.fit_flow`).

The actual training path shells out to `python -m datagen.train_flow`, needs
torch + probaforms, and takes ~5-20 s even on the smallest config we can ship.
To keep the default `pytest` run fast, the end-to-end cases are marked
`@pytest.mark.slow` and skipped by default; run them explicitly with::

    pytest -m slow tests/test_fit_flow.py

The fast, always-on tests exercise the aggregation helper and input
validation without requiring torch.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from api.flow_training import (
    FitFlowError,
    aggregate_to_minutes,
    fit_flow,
)


def _write_trace_csv(path: Path, rows: list[tuple[int, str, int]]) -> bytes:
    lines = ["timestamp_ms,function_id,execution_time_ms"]
    for ts, fid, exec_ms in rows:
        lines.append(f"{ts},{fid},{exec_ms}")
    content = "\n".join(lines) + "\n"
    path.write_text(content)
    return content.encode("utf-8")


def test_aggregate_to_minutes_groups_by_minute_and_function(tmp_path: Path):
    src = tmp_path / "trace.csv"
    _write_trace_csv(
        src,
        rows=[
            (1_000, "f0", 10),
            (30_000, "f0", 20),
            (70_000, "f0", 30),
            (90_000, "f1", 50),
        ],
    )
    out = tmp_path / "agg.csv"
    n = aggregate_to_minutes(src, out)
    assert n == 3
    text = out.read_text().strip().splitlines()
    assert text[0] == "minute,function_id,count,avg_exec_time_ms"
    body = [row.split(",") for row in text[1:]]
    rows_by_key = {(r[0], r[1]): r for r in body}
    assert rows_by_key[("0", "f0")][2] == "2"
    assert float(rows_by_key[("0", "f0")][3]) == pytest.approx(15.0)
    assert rows_by_key[("1", "f0")][2] == "1"
    assert rows_by_key[("1", "f1")][2] == "1"


def test_aggregate_empty_csv_raises(tmp_path: Path):
    """An upload that parses but contains zero rows is caught upstream in
    parse_user_csv (CsvSchemaError). If parse_user_csv ever relaxes that,
    aggregate_to_minutes keeps a defensive guard that raises FitFlowError."""
    from datagen.upload import CsvSchemaError

    src = tmp_path / "empty.csv"
    src.write_text("timestamp_ms,function_id,execution_time_ms\n")
    with pytest.raises((FitFlowError, CsvSchemaError)):
        aggregate_to_minutes(src, tmp_path / "agg.csv")


def test_fit_flow_rejects_empty_bytes(tmp_path: Path):
    with pytest.raises(FitFlowError, match="empty CSV"):
        asyncio.run(fit_flow(b"", tmp_path))


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("probaforms") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _has_torch(), reason="torch+probaforms not installed")
def test_fit_flow_end_to_end_and_cached(tmp_path: Path):
    """Train on a synthetic 2-function trace (~4 min), then re-fit the same
    bytes and verify we short-circuit to the cached result without reshelling
    out to the training subprocess."""
    csv_path = tmp_path / "in.csv"
    rows: list[tuple[int, str, int]] = []
    for minute in range(6):
        base = minute * 60_000
        for i in range(20):
            rows.append((base + i * 100, "f0", 20 + (i % 5)))
        for i in range(10):
            rows.append((base + 30_000 + i * 100, "f1", 50 + (i % 3)))
    csv_bytes = _write_trace_csv(csv_path, rows)

    datasets_root = tmp_path / "datasets"
    first = asyncio.run(
        fit_flow(
            csv_bytes,
            datasets_root,
            n_epochs=2,
            n_layers=2,
            hidden=(8, 8),
        )
    )
    assert first["cached"] is False
    assert len(first["dataset_id"]) == 16
    assert first["n_aggregated_rows"] >= 1
    dataset_dir = datasets_root / first["dataset_id"]
    assert (dataset_dir / "aggregated.csv").is_file()
    if first["passed"]:
        assert (dataset_dir / "flow.pt").is_file()
        assert (dataset_dir / "flow_meta.json").is_file()

    if first["passed"]:
        second = asyncio.run(fit_flow(csv_bytes, datasets_root))
        assert second["cached"] is True
        assert second["dataset_id"] == first["dataset_id"]
        assert second["passed"] is True
