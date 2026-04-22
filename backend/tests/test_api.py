from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from api.main import app


EXAMPLE_OPTIMIZER = (Path(__file__).resolve().parents[2] / "examples" / "user_optimizer_random.py").read_text()

TIMEOUT_POLL_S = 45


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        # lifespan events don't fire with ASGITransport by default
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as _:
            pass
        # manually trigger lifespan-equivalent setup
        from api.jobs import JobManager
        from api.main import DB_PATH, RUNS_ROOT
        from api.store import RunStore

        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        store = RunStore(DB_PATH)
        app.state.store = store
        app.state.jobs = JobManager(store)
        yield c


async def _poll_until_terminal(client: httpx.AsyncClient, run_id: str, timeout_s: int = TIMEOUT_POLL_S) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in {"done", "user_error", "crashed", "timeout"}:
            return body
        await asyncio.sleep(0.25)
    raise AssertionError(f"run {run_id} never finished within {timeout_s}s")


@pytest.mark.asyncio
async def test_datasets_available(client):
    r = await client.get("/datasets/available")
    assert r.status_code == 200
    body = r.json()
    assert body["poisson"] is True
    assert "flow" in body
    assert body["upload"] is True


@pytest.mark.asyncio
async def test_datasets_preview_poisson(client):
    r = await client.get("/datasets/preview", params={"source": "poisson", "intensity": 0.3, "duration_minutes": 5, "n_functions": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] > 0
    assert len(body["preview"]) <= 200
    first = body["preview"][0]
    assert set(first.keys()) == {"timestamp_ms", "function_id", "execution_time_ms"}


@pytest.mark.asyncio
async def test_unknown_run_is_404(client):
    r = await client.get("/runs/nope")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_happy_path_run_through(client):
    cfg = {
        "source": "poisson",
        "intensity": 0.2,
        "duration_minutes": 5,
        "n_functions": 4,
        "seed": 0,
        "budget": 6,
        "w_latency": 0.5,
        "w_cost": 0.5,
    }
    files = {"solution": ("solution.py", EXAMPLE_OPTIMIZER, "text/x-python")}
    r = await client.post("/runs", files=files, data={"config": json.dumps(cfg)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    run_id = body["run_id"]
    assert body["n_arrivals"] > 0

    terminal = await _poll_until_terminal(client, run_id)
    assert terminal["status"] == "done", terminal

    rep = await client.get(f"/runs/{run_id}/report")
    assert rep.status_code == 200
    report = rep.json()
    assert report["best_x"] and len(report["best_x"]) == 3
    assert report["n_trials"] == cfg["budget"]
    assert len(report["baselines"]) == 5
    assert len(report["convergence"]) == cfg["budget"]
    assert report["convergence"][-1]["best_so_far"] <= report["convergence"][0]["best_so_far"]


@pytest.mark.asyncio
async def test_user_error_surfaces_in_status(client):
    bad_solution = "def not_optimize():\n    pass\n"
    cfg = {
        "source": "poisson",
        "intensity": 0.2,
        "duration_minutes": 5,
        "n_functions": 4,
        "seed": 0,
        "budget": 3,
    }
    files = {"solution": ("solution.py", bad_solution, "text/x-python")}
    r = await client.post("/runs", files=files, data={"config": json.dumps(cfg)})
    run_id = r.json()["run_id"]
    terminal = await _poll_until_terminal(client, run_id)
    assert terminal["status"] == "user_error"
    assert terminal["error"] and "optimize" in terminal["error"].lower()


@pytest.mark.asyncio
async def test_invalid_config_rejected(client):
    files = {"solution": ("solution.py", EXAMPLE_OPTIMIZER, "text/x-python")}
    r = await client.post("/runs", files=files, data={"config": json.dumps({"source": "bogus"})})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_budget_over_cap_rejected(client):
    """Pydantic should reject budget > 200 (new strict cap)."""
    cfg = {
        "source": "poisson",
        "intensity": 0.2,
        "duration_minutes": 5,
        "n_functions": 4,
        "seed": 0,
        "budget": 300,
    }
    files = {"solution": ("solution.py", EXAMPLE_OPTIMIZER, "text/x-python")}
    r = await client.post("/runs", files=files, data={"config": json.dumps(cfg)})
    assert r.status_code == 422, r.text


@pytest.mark.asyncio
async def test_oversized_trace_rejected(client, monkeypatch):
    """Shrink MAX_ARRIVALS to cheap-to-trigger value, then verify the API
    returns 400 with a helpful message instead of queuing a monster run."""
    from api import main as api_main

    monkeypatch.setattr(api_main, "MAX_ARRIVALS", 100)
    cfg = {
        "source": "poisson",
        "intensity": 0.8,
        "duration_minutes": 60,
        "n_functions": 20,
        "seed": 0,
        "budget": 3,
    }
    files = {"solution": ("solution.py", EXAMPLE_OPTIMIZER, "text/x-python")}
    r = await client.post("/runs", files=files, data={"config": json.dumps(cfg)})
    assert r.status_code == 400, r.text
    assert "cap" in r.json()["detail"].lower() or "too large" in r.json()["detail"].lower() or "over the" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_report_404_before_done(client):
    cfg = {"source": "poisson", "intensity": 0.2, "duration_minutes": 5, "n_functions": 4, "seed": 0, "budget": 3}
    files = {"solution": ("solution.py", EXAMPLE_OPTIMIZER, "text/x-python")}
    r = await client.post("/runs", files=files, data={"config": json.dumps(cfg)})
    run_id = r.json()["run_id"]
    # Immediately poll /report — should 409 because status is queued or running
    rep = await client.get(f"/runs/{run_id}/report")
    assert rep.status_code in {404, 409}
    await _poll_until_terminal(client, run_id)
