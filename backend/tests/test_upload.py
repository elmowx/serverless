from __future__ import annotations

from pathlib import Path

import pytest

from datagen.upload import CsvSchemaError, parse_user_csv


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_valid_csv_parses_and_sorts(tmp_path: Path):
    csv = (
        "timestamp_ms,function_id,execution_time_ms\n"
        "2000,f1,50\n"
        "1000,f2,100\n"
        "3000,f1,25\n"
    )
    trace = parse_user_csv(_write(tmp_path, "ok.csv", csv))
    assert len(trace) == 3
    assert [r.timestamp_ms for r in trace] == [1000, 2000, 3000]


def test_column_order_insensitive(tmp_path: Path):
    csv = (
        "function_id,execution_time_ms,timestamp_ms\n"
        "f1,50,1000\n"
    )
    trace = parse_user_csv(_write(tmp_path, "order.csv", csv))
    assert trace[0].function_id == "f1"
    assert trace[0].execution_time_ms == 50.0


def test_missing_column_raises(tmp_path: Path):
    csv = "timestamp_ms,function_id\n1000,f1\n"
    with pytest.raises(CsvSchemaError, match="missing"):
        parse_user_csv(_write(tmp_path, "bad.csv", csv))


def test_empty_function_id_raises(tmp_path: Path):
    csv = (
        "timestamp_ms,function_id,execution_time_ms\n"
        "1000,,50\n"
    )
    with pytest.raises(CsvSchemaError, match="function_id empty"):
        parse_user_csv(_write(tmp_path, "empty.csv", csv))


def test_negative_timestamp_raises(tmp_path: Path):
    csv = (
        "timestamp_ms,function_id,execution_time_ms\n"
        "-1,f1,50\n"
    )
    with pytest.raises(CsvSchemaError, match="non-negative"):
        parse_user_csv(_write(tmp_path, "neg.csv", csv))


def test_non_numeric_execution_time_raises(tmp_path: Path):
    csv = (
        "timestamp_ms,function_id,execution_time_ms\n"
        "1000,f1,not-a-number\n"
    )
    with pytest.raises(CsvSchemaError, match="numeric parse error"):
        parse_user_csv(_write(tmp_path, "bad_num.csv", csv))


def test_empty_csv_raises(tmp_path: Path):
    csv = "timestamp_ms,function_id,execution_time_ms\n"
    with pytest.raises(CsvSchemaError, match="no data"):
        parse_user_csv(_write(tmp_path, "empty.csv", csv))


def test_file_not_found(tmp_path: Path):
    with pytest.raises(CsvSchemaError, match="not found"):
        parse_user_csv(tmp_path / "nope.csv")
