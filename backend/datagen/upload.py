from __future__ import annotations

import csv
from pathlib import Path

from core.types import RequestArrival

REQUIRED_COLUMNS = {"timestamp_ms", "function_id", "execution_time_ms"}
MAX_ROWS = 2_000_000


class CsvSchemaError(ValueError):
    pass


def parse_user_csv(path: str | Path) -> list[RequestArrival]:
    p = Path(path)
    if not p.is_file():
        raise CsvSchemaError(f"file not found: {p}")

    with p.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise CsvSchemaError("CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise CsvSchemaError(
                f"CSV missing required columns: {sorted(missing)} "
                f"(got: {reader.fieldnames})"
            )

        out: list[RequestArrival] = []
        for i, row in enumerate(reader, start=2):
            if i - 1 > MAX_ROWS:
                raise CsvSchemaError(f"CSV exceeds {MAX_ROWS} rows")
            try:
                ts = int(float(row["timestamp_ms"]))
                ex = float(row["execution_time_ms"])
            except (TypeError, ValueError) as exc:
                raise CsvSchemaError(f"row {i}: numeric parse error: {exc}") from None
            fid = (row["function_id"] or "").strip()
            if not fid:
                raise CsvSchemaError(f"row {i}: function_id empty")
            if ts < 0:
                raise CsvSchemaError(f"row {i}: timestamp_ms must be non-negative")
            if ex < 0:
                raise CsvSchemaError(f"row {i}: execution_time_ms must be non-negative")
            out.append(RequestArrival(timestamp_ms=ts, function_id=fid, execution_time_ms=ex))

    if not out:
        raise CsvSchemaError("CSV contained no data rows")
    out.sort(key=lambda r: r.timestamp_ms)
    return out
