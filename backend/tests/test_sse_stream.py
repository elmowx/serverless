from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from api.main import _tail_progress


async def _drain(gen, max_events: int = 200, timeout_s: float = 2.0) -> list[dict]:
    out: list[dict] = []
    try:
        async with asyncio.timeout(timeout_s):
            async for ev in gen:
                out.append(ev)
                if len(out) >= max_events:
                    break
    except asyncio.TimeoutError:
        pass
    return out


@pytest.mark.asyncio
async def test_tail_progress_yields_each_trial_once_then_done(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    n = 50
    progress.write_text(
        "\n".join(json.dumps({"trial": i, "y": float(i)}) for i in range(n)) + "\n"
    )

    state = {"phase": 0}

    def poll() -> tuple[str | None, str | None, int | None]:
        state["phase"] += 1
        if state["phase"] >= 3:
            return "done", None, 0
        return "running", None, None

    events = await _drain(
        _tail_progress(progress, poll, sleep_s=0.0), max_events=n + 1
    )
    trial_events = [e for e in events if e["event"] == "trial"]
    done_events = [e for e in events if e["event"] == "done"]
    assert len(trial_events) == n, (
        f"expected {n} trial events, got {len(trial_events)}"
    )
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_tail_progress_does_not_use_read_text(tmp_path, monkeypatch) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({"trial": 1, "y": 0.5}) + "\n")

    calls = {"n": 0}
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self == progress:
            calls["n"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    def poll() -> tuple[str | None, str | None, int | None]:
        return "done", None, 0

    await _drain(_tail_progress(progress, poll, sleep_s=0.0), max_events=5)
    assert calls["n"] == 0, (
        f"_tail_progress must stream incrementally, not read_text(); "
        f"got {calls['n']} read_text calls"
    )


@pytest.mark.asyncio
async def test_tail_progress_picks_up_appended_lines(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({"trial": 1}) + "\n")

    state = {"polls": 0}

    def poll() -> tuple[str | None, str | None, int | None]:
        state["polls"] += 1
        if state["polls"] == 1:
            with progress.open("a") as fh:
                fh.write(json.dumps({"trial": 2}) + "\n")
            return "running", None, None
        return "done", None, 0

    events = await _drain(_tail_progress(progress, poll, sleep_s=0.0), max_events=5)
    trial_events = [e for e in events if e["event"] == "trial"]
    assert [json.loads(e["data"])["trial"] for e in trial_events] == [1, 2]


@pytest.mark.asyncio
async def test_tail_progress_handles_partial_line(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_text('{"trial": 1, "y"')

    state = {"polls": 0}

    def poll() -> tuple[str | None, str | None, int | None]:
        state["polls"] += 1
        if state["polls"] == 1:
            with progress.open("a") as fh:
                fh.write(": 0.5}\n")
            return "running", None, None
        return "done", None, 0

    events = await _drain(_tail_progress(progress, poll, sleep_s=0.0), max_events=5)
    trial_events = [e for e in events if e["event"] == "trial"]
    assert len(trial_events) == 1
    assert json.loads(trial_events[0]["data"]) == {"trial": 1, "y": 0.5}
