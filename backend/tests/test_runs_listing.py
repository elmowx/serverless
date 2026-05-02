from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from api.jobs import JobManager
from api.main import app
from api.store import RunStore


@pytest.fixture
async def client(tmp_path: Path):
    store = RunStore(tmp_path / "runs.sqlite")
    app.state.store = store
    app.state.jobs = JobManager(store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, store


@pytest.mark.asyncio
async def test_list_runs_returns_recent_first(client) -> None:
    c, store = client
    job_root = Path("/tmp/serverless_blackbox_runs")
    job_root.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        store.create(
            run_id=f"run{i}",
            config={"budget": 1, "source": "poisson"},
            job_dir=job_root / f"run{i}",
        )
        time.sleep(0.01)

    r = await c.get("/runs", params={"limit": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["id"] == "run2"
    assert body[1]["id"] == "run1"
    for item in body:
        assert set(item.keys()) >= {"id", "status", "created_at", "finished_at"}


@pytest.mark.asyncio
async def test_list_runs_default_limit(client) -> None:
    c, store = client
    r = await c.get("/runs")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_list_runs_rejects_oversize_limit(client) -> None:
    c, _ = client
    r = await c.get("/runs", params={"limit": 999})
    assert r.status_code == 422
