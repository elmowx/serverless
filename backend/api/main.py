from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from api.flow_training import FitFlowError, fit_flow
from api.jobs import JobManager
from api.pdf import render_report_pdf
from api.store import Run, RunStore
from core.types import RequestArrival
from datagen import PoissonGenerator, parse_user_csv
from datagen.flow import FlowGenerator


RUNS_ROOT = Path(tempfile.gettempdir()) / "serverless_blackbox_runs"
DB_PATH = RUNS_ROOT / "runs.sqlite"
DATASETS_ROOT = RUNS_ROOT / "datasets"
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
    max_containers_cap: int = Field(30, ge=1, le=30)
    max_wait_ms: float = Field(30_000.0, ge=0.0, le=600_000.0)
    dataset_id: str | None = None

    @field_validator("source")
    @classmethod
    def flow_available(cls, v: str, info: Any) -> str:  # type: ignore[override]
        if v == "flow":
            if not FlowGenerator.is_available() and not (info.data or {}).get("dataset_id"):
                raise ValueError("flow source requested but weights not shipped")
        return v


class DatasetsAvailable(BaseModel):
    poisson: bool
    flow: bool
    upload: bool
    fit_flow: bool


class FitFlowResult(BaseModel):
    dataset_id: str
    cached: bool
    passed: bool
    ks_p_count: float
    ks_p_exec: float
    n_aggregated_rows: int


class PreviewArrival(BaseModel):
    timestamp_ms: int
    function_id: str
    execution_time_ms: float


class PreviewResponse(BaseModel):
    n_total: int
    preview: list[PreviewArrival]


class RunCreated(BaseModel):
    run_id: str
    status: str
    n_arrivals: int


class RunSummary(BaseModel):
    id: str
    status: str
    created_at: str
    finished_at: str | None = None


class RunDetail(RunSummary):
    exit_code: int | None = None
    error: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class _Permissive(BaseModel):
    model_config = ConfigDict(extra="allow")


class ContainerSummaryEntry(_Permissive):
    container_id: int
    busy_frac: float
    idle_frac: float
    free_frac: float
    warming_frac: float
    cold_starts: int
    warm_hits: int


class BestMetrics(_Permissive):
    p99_latency_ms: float
    avg_latency_ms: float | None = None
    cold_start_rate: float
    p_loss: float
    idle_seconds: float
    warm_hits: int | None = None
    cold_starts: int | None = None
    latency_term: float | None = None
    cost_term: float | None = None
    container_summary: list[ContainerSummaryEntry] | None = None


class BaselinePolicy(BaseModel):
    keep_alive_s: float
    max_containers: int
    prewarm_threshold: float


class BaselineResult(_Permissive):
    name: str
    policy: BaselinePolicy
    y: float
    metrics: BestMetrics


class ConvergencePoint(_Permissive):
    trial: int
    y: float
    best_so_far: float
    n_containers: int | None = None
    cold_start_rate: float | None = None


class ParetoPoint(BaseModel):
    x: list[float]
    p99_latency_ms: float
    idle_seconds: float


class TimelineSegmentDTO(BaseModel):
    state: str
    t0_ms: float
    t1_ms: float


class ContainerTrack(BaseModel):
    container_id: int
    segments: list[TimelineSegmentDTO]


class ContainerTimelineDTO(BaseModel):
    total_ms: float
    n_containers: int
    shown_containers: int
    tracks: list[ContainerTrack]


class ReportConfig(BaseModel):
    w_latency: float
    w_cost: float
    seed: int
    max_wait_ms: float | None = None


class Normalization(BaseModel):
    l_max: float
    c_max: float
    w_latency: float
    w_cost: float


class ReportResponse(_Permissive):
    best_x: list[float]
    best_y: float
    best_metrics: BestMetrics
    n_trials: int
    elapsed_s: float
    baselines: list[BaselineResult]
    convergence: list[ConvergencePoint]
    pareto_points: list[ParetoPoint]
    container_timeline: ContainerTimelineDTO | None = None
    config: ReportConfig
    normalization: Normalization | None = None


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


@app.get("/datasets/available", response_model=DatasetsAvailable)
def datasets_available() -> dict[str, Any]:
    return {
        "poisson": True,
        "flow": FlowGenerator.is_available(),
        "upload": True,
        "fit_flow": True,
    }


@app.post("/datasets/fit-flow", response_model=FitFlowResult)
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


@app.get("/datasets/preview", response_model=PreviewResponse)
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


@app.post("/runs", response_model=RunCreated)
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
        max_wait_ms=cfg.max_wait_ms,
    )
    return {"run_id": run_id, "status": "queued", "n_arrivals": len(trace)}


def _run_or_404(store: RunStore, run_id: str) -> Run:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    runs = app.state.store.list_recent(limit=limit)
    return [
        {
            "id": r.id,
            "status": r.status,
            "created_at": r.created_at,
            "finished_at": r.finished_at,
        }
        for r in runs
    ]


@app.get("/runs/{run_id}", response_model=RunDetail)
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


def _load_report(run: Run) -> dict[str, Any]:
    report_path = Path(run.job_dir) / "report.json"
    if not report_path.exists():
        raise HTTPException(500, "report.json missing — report assembly failed")
    return json.loads(report_path.read_text())


@app.get("/runs/{run_id}/report", response_model=ReportResponse)
def get_report(run_id: str) -> dict[str, Any]:
    run = _run_or_404(app.state.store, run_id)
    if run.status != "done":
        raise HTTPException(409, f"report not available; run status={run.status}")
    return _load_report(run)


@app.get("/runs/{run_id}/report.pdf")
def get_report_pdf(run_id: str) -> Response:
    run = _run_or_404(app.state.store, run_id)
    if run.status != "done":
        raise HTTPException(409, f"report not available; run status={run.status}")
    report = _load_report(run)
    pdf_bytes = render_report_pdf(report, run_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="report-{run_id}.pdf"'},
    )


async def _tail_progress(
    progress_path: Path,
    poll_status: "callable[[], tuple[str | None, str | None, int | None]]",
    sleep_s: float = 0.2,
) -> AsyncIterator[dict[str, Any]]:
    fh = None
    carry = ""
    try:
        while True:
            if fh is None and progress_path.exists():
                fh = progress_path.open("r")
            if fh is not None:
                chunk = fh.read()
                if chunk:
                    carry += chunk
                    while "\n" in carry:
                        line, carry = carry.split("\n", 1)
                        if line.strip():
                            yield {"event": "trial", "data": line}
            status, error, exit_code = poll_status()
            if status is None:
                return
            if status in {"done", "user_error", "crashed", "timeout"}:
                if fh is not None:
                    chunk = fh.read()
                    if chunk:
                        carry += chunk
                    for line in carry.splitlines():
                        if line.strip():
                            yield {"event": "trial", "data": line}
                    carry = ""
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {"status": status, "error": error, "exit_code": exit_code}
                    ),
                }
                return
            await asyncio.sleep(sleep_s)
    finally:
        if fh is not None:
            fh.close()


@app.get("/runs/{run_id}/events")
async def get_events(run_id: str) -> EventSourceResponse:
    run = _run_or_404(app.state.store, run_id)
    progress_path = Path(run.job_dir) / "progress.jsonl"

    def _poll() -> tuple[str | None, str | None, int | None]:
        current = app.state.store.get(run_id)
        if current is None:
            return None, None, None
        return current.status, current.error, current.exit_code

    return EventSourceResponse(_tail_progress(progress_path, _poll))
