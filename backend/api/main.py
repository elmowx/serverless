"""
FastAPI app: `uvicorn api.main:app --reload`.

Endpoints (see OpenAPI at /docs):
    POST  /runs                       enqueue optimizer run
    GET   /runs/{run_id}              status + minimal info
    GET   /runs/{run_id}/events       SSE stream of progress lines + final event
    GET   /runs/{run_id}/report       final report JSON (only after status=done)
    GET   /datasets/available         {poisson: true, flow: bool}
    GET   /datasets/preview           first N arrivals of a synthetic sample
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from api.flow_training import FitFlowError, fit_flow
from api.jobs import JobManager
from api.pdf import render_report_pdf
from api.reporting import write_report
from api.store import Run, RunStore
from core.types import RequestArrival
from datagen import PoissonGenerator, parse_user_csv
from datagen.flow import FlowGenerator


RUNS_ROOT = Path(tempfile.gettempdir()) / "serverless_blackbox_runs"
DB_PATH = RUNS_ROOT / "runs.sqlite"
DATASETS_ROOT = RUNS_ROOT / "datasets"

# Hard cap on trace size fed into a single run. The simulator is O(n log n)
# in trace length (heap operations); at ~2 µs per arrival this is ~1 s of
# compute per objective call. Combined with budget=200 that is up to 200 s
# of work, comfortably under the sandbox's wall-clock timeout.
MAX_ARRIVALS = 500_000

SourceLiteral = Literal["poisson", "flow", "upload"]


class RunConfig(BaseModel):
    source: SourceLiteral
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    duration_minutes: int = Field(60, ge=1, le=1440)
    n_functions: int = Field(10, ge=1, le=200)
    seed: int = 0
    budget: int = Field(30, ge=1, le=200)
    w_latency: float = Field(0.5, ge=0.0, le=1.0)
    w_cost: float = Field(0.5, ge=0.0, le=1.0)
    # Upper bound the optimizer is allowed to set for Policy.max_containers.
    # Exposed so users running on smaller fleets can narrow the search space
    # without editing backend BOUNDS. The hard ceiling (30) comes from the
    # simulator's calibration range and is enforced by Field(le=...).
    max_containers_cap: int = Field(30, ge=1, le=30)
    dataset_id: str | None = None  # if set with source=flow, use user-trained weights

    @field_validator("source")
    @classmethod
    def flow_available(cls, v: str, info: Any) -> str:  # type: ignore[override]
        if v == "flow":
            # Only require shipped weights when the run does NOT reference a
            # user-trained dataset. Availability of per-dataset weights is
            # re-checked inside _build_trace.
            if not FlowGenerator.is_available() and not (info.data or {}).get("dataset_id"):
                raise ValueError("flow source requested but weights not shipped")
        return v


def _dataset_weights(dataset_id: str) -> tuple[Path, Path]:
    d = DATASETS_ROOT / dataset_id
    return d / "flow.pt", d / "flow_meta.json"


def _build_trace(cfg: RunConfig, uploaded_csv: Path | None) -> list[RequestArrival]:
    if cfg.source == "poisson":
        trace = PoissonGenerator().generate(
            intensity=cfg.intensity,
            duration_minutes=cfg.duration_minutes,
            n_functions=cfg.n_functions,
            seed=cfg.seed,
        )
    elif cfg.source == "flow":
        if cfg.dataset_id:
            w, m = _dataset_weights(cfg.dataset_id)
            if not w.is_file() or not m.is_file():
                raise HTTPException(404, f"dataset {cfg.dataset_id} has no trained weights")
            gen = FlowGenerator(weights_path=w, meta_path=m)
        else:
            gen = FlowGenerator()
        trace = gen.generate(
            intensity=cfg.intensity,
            duration_minutes=cfg.duration_minutes,
            n_functions=cfg.n_functions,
            seed=cfg.seed,
        )
    elif cfg.source == "upload":
        if uploaded_csv is None:
            raise HTTPException(400, "source=upload requires trace_csv file part")
        trace = parse_user_csv(uploaded_csv)
    else:
        raise HTTPException(400, f"unknown source: {cfg.source}")
    if len(trace) > MAX_ARRIVALS:
        raise HTTPException(
            400,
            f"generated trace has {len(trace):,} arrivals, over the "
            f"{MAX_ARRIVALS:,} cap — lower intensity, duration_minutes, "
            f"or n_functions",
        )
    return trace


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    store = RunStore(DB_PATH)
    app.state.store = store
    app.state.jobs = JobManager(store)
    yield


app = FastAPI(title="Serverless Black-Box API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/datasets/available")
def datasets_available() -> dict[str, Any]:
    return {
        "poisson": True,
        "flow": FlowGenerator.is_available(),
        "upload": True,
        "fit_flow": True,
    }


@app.post("/datasets/fit-flow")
async def fit_flow_endpoint(
    trace_csv: UploadFile = File(..., description="per-arrival CSV to train a conditional flow on"),
) -> dict[str, Any]:
    contents = await trace_csv.read()
    if len(contents) > 64 * 1024 * 1024:
        raise HTTPException(400, "trace_csv must be under 64 MB")
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        result = await fit_flow(contents, DATASETS_ROOT)
    except FitFlowError as exc:
        raise HTTPException(400, f"flow training failed: {exc}") from exc
    return result


@app.get("/datasets/preview")
def datasets_preview(
    source: SourceLiteral = "poisson",
    intensity: float = 0.5,
    duration_minutes: int = 60,
    n_functions: int = 10,
    seed: int = 0,
    limit: int = 200,
) -> dict[str, Any]:
    if source == "flow" and not FlowGenerator.is_available():
        raise HTTPException(404, "flow weights not available")
    if source == "upload":
        raise HTTPException(400, "preview does not support upload source")
    cfg = RunConfig(
        source=source,
        intensity=intensity,
        duration_minutes=duration_minutes,
        n_functions=n_functions,
        seed=seed,
        budget=1,
    )
    trace = _build_trace(cfg, None)
    return {
        "n_total": len(trace),
        "preview": [
            {"timestamp_ms": r.timestamp_ms, "function_id": r.function_id, "execution_time_ms": r.execution_time_ms}
            for r in trace[:limit]
        ],
    }


@app.post("/runs")
async def create_run(
    solution: UploadFile = File(..., description="python file defining optimize(...)"),
    config: str = Form(..., description="JSON-encoded RunConfig"),
    trace_csv: UploadFile | None = File(None),
) -> dict[str, Any]:
    try:
        cfg = RunConfig.model_validate_json(config)
    except Exception as exc:
        raise HTTPException(422, f"invalid config: {exc}") from exc

    solution_bytes = await solution.read()
    if not solution_bytes:
        raise HTTPException(400, "solution.py is empty")
    if len(solution_bytes) > 256 * 1024:
        raise HTTPException(400, "solution.py must be under 256 KB")
    solution_source = solution_bytes.decode("utf-8", errors="replace")

    run_id = uuid.uuid4().hex[:12]
    job_dir = RUNS_ROOT / run_id
    job_dir.mkdir(parents=True, exist_ok=False)

    uploaded_csv_path: Path | None = None
    if trace_csv is not None:
        contents = await trace_csv.read()
        if contents:
            uploaded_csv_path = job_dir / "user_trace.csv"
            uploaded_csv_path.write_bytes(contents)

    try:
        trace = _build_trace(cfg, uploaded_csv_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"failed to build trace: {exc}") from exc

    if not trace:
        raise HTTPException(400, "generated trace is empty")

    app.state.store.create(run_id=run_id, config=cfg.model_dump(), job_dir=job_dir)
    await app.state.jobs.submit(
        run_id=run_id,
        job_dir=job_dir,
        solution_source=solution_source,
        trace=trace,
        budget=cfg.budget,
        w_latency=cfg.w_latency,
        w_cost=cfg.w_cost,
        seed=cfg.seed,
        max_containers_cap=cfg.max_containers_cap,
    )
    return {"run_id": run_id, "status": "queued", "n_arrivals": len(trace)}


def _run_or_404(store: RunStore, run_id: str) -> Run:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = _run_or_404(app.state.store, run_id)
    return {
        "id": run.id,
        "status": run.status,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "exit_code": run.exit_code,
        "error": run.error,
        "config": run.config,
    }


def _load_or_refresh_report(run: Run) -> dict[str, Any]:
    """Return the run's report, regenerating it on-disk if the cached copy
    was produced by an older runner (missing the per-baseline latency/cost
    decomposition that the current UI depends on). Uses the pickled trace
    and config stored next to result.json — cheap because baseline sims are
    already part of ``build_report``."""
    import pickle

    job_dir = Path(run.job_dir)
    report_path = job_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(500, "report.json missing — report assembly failed")

    report = json.loads(report_path.read_text())
    baselines = report.get("baselines") or []
    stale = (
        not baselines
        or any(
            "latency_term" not in (b.get("metrics") or {})
            or "cost_term" not in (b.get("metrics") or {})
            for b in baselines
        )
    )
    if not stale:
        return report

    trace_path = job_dir / "trace.pkl"
    config_path = job_dir / "config.json"
    if not trace_path.exists() or not config_path.exists():
        return report
    with trace_path.open("rb") as fh:
        trace = pickle.load(fh)
    cfg = json.loads(config_path.read_text())
    return write_report(
        job_dir=job_dir,
        trace=trace,
        w_latency=float(cfg.get("w_latency", 0.5)),
        w_cost=float(cfg.get("w_cost", 0.5)),
        seed=int(cfg.get("seed", 0)),
    )


@app.get("/runs/{run_id}/report")
def get_report(run_id: str) -> JSONResponse:
    run = _run_or_404(app.state.store, run_id)
    if run.status != "done":
        raise HTTPException(409, f"report not available; run status={run.status}")
    return JSONResponse(_load_or_refresh_report(run))


@app.get("/runs/{run_id}/report.pdf")
def get_report_pdf(run_id: str) -> Response:
    run = _run_or_404(app.state.store, run_id)
    if run.status != "done":
        raise HTTPException(409, f"report not available; run status={run.status}")
    report = _load_or_refresh_report(run)
    pdf_bytes = render_report_pdf(report, run_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="report-{run_id}.pdf"'},
    )


@app.get("/runs/{run_id}/events")
async def get_events(run_id: str) -> EventSourceResponse:
    run = _run_or_404(app.state.store, run_id)
    progress_path = Path(run.job_dir) / "progress.jsonl"

    async def stream() -> AsyncIterator[dict[str, Any]]:
        sent = 0
        while True:
            if progress_path.exists():
                lines = progress_path.read_text().splitlines()
                for raw in lines[sent:]:
                    if raw.strip():
                        yield {"event": "trial", "data": raw}
                sent = len(lines)
            current = app.state.store.get(run_id)
            if current is None:
                return
            if current.status in {"done", "user_error", "crashed", "timeout"}:
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "status": current.status,
                            "error": current.error,
                            "exit_code": current.exit_code,
                        }
                    ),
                }
                return
            await asyncio.sleep(0.2)

    return EventSourceResponse(stream())
