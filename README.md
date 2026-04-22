# Serverless Black-Box

A benchmarking harness for serverless function scheduling. It ships the **cold-start vs. keep-alive trade-off** as a scalar black-box objective `f(x) → metrics`, allowing you to point your own derivative-free optimizer at it and see how it compares to five hand-crafted baselines under highly realistic workloads.

The research context is the HSE coursework **"Benchmarking Black-Box for Serverless Function Scheduling and Cold Start Mitigation"** — see [`report/final_report.tex`](report/final_report.tex) for the full write-up.

---

## The Core Concepts

### The Trade-off
In serverless computing (like AWS Lambda or Azure Functions), platforms face a constant trade-off:
- **Cold Starts:** Initializing a new container takes time and causes latency.
- **Keep-Alive:** Keeping containers warm (idle) avoids cold starts but costs money.

### The Simulator & FSM
The core of the benchmark is a **G/G/k/k loss queue simulator** with a four-state container Finite State Machine (FSM):
1. **`FREE`**: An unallocated concurrency slot.
2. **`WARMING_UP`**: The container is initializing. Cold start latency is strictly the sum of four phases (`env_init`, `code_loading`, `runtime_start`, `function_init`), calibrated against 805,745 real production events from Huawei's SIR-Lab dataset.
3. **`BUSY`**: The container is actively executing a request.
4. **`IDLE`**: The container is warm, kept-alive, and waiting for a hit.

*Note: If a target function already has a container in the `WARMING_UP` state, new requests join a **pending queue** rather than being rejected, mirroring AWS Lambda's provisioned concurrency behavior.*

### The Black-Box Objective
The simulator collapses the entire trace execution into a single normalized scalar `f(x) ∈ [0, 1]`:

```text
f(x) = w_lat · min(CVaR_0.99(x) / L_max, 1.0) + w_cost · min(idle_seconds(x) / C_max, 1.0)
```
- **`CVaR_0.99`**: The mean latency of the top 1% worst requests. It is smoother and has lower variance than a raw p99, making the objective landscape significantly more stable for optimizers.
- **`idle_seconds`**: The total container-seconds spent in the IDLE state (the keep-alive cost).
- **`L_max` & `C_max`**: Per-trace normalization constants derived from evaluating two extreme reference policies (`minimal` and `generous`) before your optimizer runs.

---

## What You Get

- **Data sources** — four interchangeable workloads:
  1. **Synthetic Poisson (NHPP)** — Mathematical generator with a 24h diurnal multiplier and Zipf(1.2) function popularity. Always available.
  2. **Historical Flow (RealNVP)** — A normalizing flow trained on Azure 2019 day 1 data. Uses uniform dequantization and passes a strict KS-gate to preserve joint distributions.
  3. **Your data (raw)** — Upload a per-arrival CSV to replay it 1:1.
  4. **Train flow on my logs** — Upload logs, and the backend trains a lightweight conditional flow in ~30s, KS-gates the fit, and uses the resulting weights.
- **Sandbox** (`backend/worker/`) — Runs your `optimize()` function in a secure subprocess with scrubbed env, strict rlimits (120s wall-clock, 60s CPU, 512MB RAM), and no network access.
- **FastAPI + SSE backend** (`backend/api/`) with a `/docs` OpenAPI page.
- **React + Vite frontend** — Upload page, live running page with animated pixel-cat workers, and a report page with convergence charts, Pareto scatter plots, and PDF export.
- **Reproducible report figures** — `report/scripts/gen_figures.py` regenerates all LaTeX figures from the shipped code and weights.

---

## Quickstart

### 1. Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate

# probaforms pins old pandas/numpy/torch — install its deps separately first
pip install torch scipy scikit-learn numpy optuna
pip install --no-deps probaforms
pip install -e ".[api,dev]"

uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Swagger at <http://127.0.0.1:8000/docs>.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev    # → http://127.0.0.1:5173
```
The Vite dev server proxies `/api/*` to port 8000.

### 3. Run a benchmark via UI

1. Open `http://127.0.0.1:5173/new`.
2. Upload a `.py` file that defines your optimizer (see the **Optimizer Contract** below). An example lives at [`examples/user_optimizer_random.py`](examples/user_optimizer_random.py).
3. Pick a data source, configure your budget and weights, and click **Start**.
4. Watch the cats animate the FSM states in real-time. When finished, you land on the report page.

### 4. Run via Docker Compose (Recommended for strict Sandbox limits)

A single command brings up both services on a clean box (no Python/Node required on the host). **Note:** Running via Docker is required to enforce memory limits (`RLIMIT_AS`) on macOS.

```bash
docker compose up --build
# backend:   http://127.0.0.1:8000/docs
# frontend:  http://127.0.0.1:8080
```

---

## Your Optimizer Contract

You must upload a single `.py` file exposing this exact top-level function:

```python
def optimize(
    objective,                           # callable: list[float] -> float
    budget: int,                         # hard cap on objective() calls
    bounds: list[tuple[float, float]],   # per-dim search bounds
) -> list[float]:
    """
    bounds = [
        (1.0, 1800.0),            # keep_alive_s
        (1.0, max_containers),    # max_containers (default cap: 30)
        (0.1, 1.0)                # prewarm_threshold
    ]
    Returns: The best [keep_alive_s, max_containers, prewarm_threshold] found.
    """
    ...
```

---

## CLI Usage (No Frontend)

Same pipeline, terminal-only — useful for CI, scripted sweeps, or Docker-only setups:

```bash
cd backend
source .venv/bin/activate
python -m cli run \
    --solution ../examples/user_optimizer_random.py \
    --source poisson \
    --intensity 0.3 --duration-minutes 5 --n-functions 4 \
    --budget 10 --seed 0
```

**Useful flags:**
- `--source {poisson,flow,upload}` (for `upload` pass `--trace-csv path/to/user.csv`).
- `--dataset-id <sha16>` to reuse weights produced by `POST /datasets/fit-flow`.
- `--output-dir ./cli_runs/my_exp` to keep the artifacts (`progress.jsonl`, `report.json`).
- `--no-sandbox` runs inline (no subprocess, no rlimits) — faster for iterative debugging.

---

## Repository Layout

- **`backend/`** — Python backend (FastAPI + Simulator)
  - **`core/`** — Pure library: types, simulator, objective, baselines, cold-start calibration.
  - **`datagen/`** — Workload generators: Poisson, Flow (RealNVP), CSV parser, and training script.
  - **`api/`** — FastAPI app, SSE streaming, job orchestrator, sandbox manager, and PDF renderer.
  - **`worker/`** — Subprocess entry point for running user optimizers safely.
  - **`weights/`** — Canonical flow weights and metadata (Azure 2019 day 1, KS-passed).
  - **`tests/`** — Pytest suite (48 tests, no network required).
- **`frontend/`** — React 19 + Vite + TS + Tailwind + framer-motion + recharts.
- **`examples/`** — Example optimizers (e.g., `user_optimizer_random.py`) demonstrating the contract.
- **`report/`** — LaTeX source for the final report (`final_report.tex`), generated figures, and archived checkpoints.
- **`PROGRESS.md`** — Milestone log and development history.
- **`CLAUDE.md`** — Working agreement and instructions for AI contributors.

---

## Design Choices Worth Knowing

- **Two independent data pipelines.** `PoissonGenerator` has zero ML deps and always works; `FlowGenerator` is best-effort. If the canonical weights ever fail to load or the KS-gate regresses, `FlowGenerator.is_available() → False`, the UI hides that tab, and nothing else breaks.
- **Sandbox, not jail.** rlimits (CPU / files / processes) + scrubbed env + hard wall-timeout. On macOS `RLIMIT_AS` silently no-ops (kernel quirk); we document this rather than pretend otherwise. For a public deployment, use the Docker-based isolation.
- **Determinism.** Both generators are seeded; the objective is deterministic for a given `(trace, w, seed)`.
- **No bundled raw datasets.** The Azure day we trained on is absorbed into `weights/flow_v1.pt` (88 KB). Huawei cold-start calibration is a frozen JSON in `core/calibration.py` (four tuples). Repo size stays in single-digit MB.

---

## Attribution

- Pixel cat sprites from [last-tick's Animated Pixel Kittens](https://last-tick.itch.io/animated-pixel-kittens-cats-32x32). Free-tier palette (3 colours) used under the pack's personal-and-commercial licence.
- Azure Functions traces: Shahrad et al., 2020.
- Huawei SIR-Lab 2025 cold-start dataset: Lin et al., 2025.

See `report/final_report.tex` for the full reference list.
