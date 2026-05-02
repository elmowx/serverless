# Serverless Black-Box

A benchmarking harness for serverless function scheduling. The simulator collapses a workload trace into a scalar `f(x)` so any derivative-free optimizer can be plugged in and compared against five fixed baselines.

HSE coursework "Benchmarking Black-Box for Serverless Function Scheduling and Cold Start Mitigation"

---

## Quickstart

### Docker Compose (recommended)

```bash
docker compose up --build
# backend:  http://127.0.0.1:8000/docs
# frontend: http://127.0.0.1:8080
```

Required on macOS to enforce memory limits (`RLIMIT_AS` no-ops on Darwin).

### Local dev (Linux/macOS)

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate

pip install --no-deps "probaforms>=0.1"
pip install -e ".[api,datagen,dev]"

uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`probaforms` ships transitive pins (pandas, numpy, torch) that conflict with our
stack, so it must be installed first with `--no-deps`. The Dockerfile follows
the same pattern; see `Dockerfile.backend` and DEPLOYMENT.md.

Swagger at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Frontend

```bash
cd frontend
npm install
npm run dev    # http://127.0.0.1:5173
```

The Vite dev server proxies `/api/*` to port 8000.

---

## Optimizer contract

Upload a single `.py` file with a top-level `optimize`:

```python
def optimize(
    objective,                           # callable: list[float] -> float
    budget: int,                         # hard cap on objective() calls
    bounds: list[tuple[float, float]],   # per-dim search bounds
) -> list[float]:
    # bounds = [
    #     (1.0, 1800.0),            # keep_alive_s
    #     (1.0, max_containers),    # max_containers (default cap: 30)
    #     (0.1, 1.0),               # prewarm_threshold
    # ]
    ...
```

`max_wait_ms` is a fixed hyperparameter of the run (default 30000 ms): requests that find no free container join a global FIFO queue and are rejected if they wait longer than `max_wait_ms`. Wait time is part of the per-request latency in `CVaR_0.99`. Setting `max_wait_ms = 0` reverts to a strict G/G/k/k loss queue.

Reference optimizers in `examples/`:

- `user_optimizer_random.py` — uniform random search.
- `user_optimizer_de.py` — `scipy.optimize.differential_evolution`.
- `user_optimizer_super.py` — Optuna TPE.

---

## CLI

```bash
cd backend
source .venv/bin/activate
python -m cli run \
    --solution ../examples/user_optimizer_de.py \
    --source poisson \
    --intensity 0.3 --duration-minutes 5 --n-functions 4 \
    --budget 30 --seed 0 --max-wait-ms 30000
```

Useful flags:

- `--source {poisson,flow,upload}` — for `upload` pass `--trace-csv path/to/user.csv`.
- `--dataset-id <sha16>` — reuse weights from `POST /datasets/fit-flow`.
- `--max-wait-ms <ms>` — waiting-queue SLA (default 30000).
- `--output-dir ./runs/my_exp` — keep `progress.jsonl` and `report.json`.
- `--no-sandbox` — inline run, no subprocess, no rlimits.

---

## Repository layout

- `backend/core/` — types, simulator, objective, baselines, cold-start calibration.
- `backend/datagen/` — workload generators (Poisson NHPP, RealNVP flow), CSV parser, training script.
- `backend/api/` — FastAPI app, SSE streaming, job orchestrator, sandbox manager, PDF renderer.
- `backend/worker/` — subprocess entry point for user optimizers.
- `backend/weights/` — canonical flow weights (Azure 2019 day 1).
- `backend/tests/` — pytest suite.
- `frontend/` — React 19 + Vite + TS + Tailwind.
- `examples/` — reference optimizers.
- `report/` — LaTeX source, scripts, figures, sweep results.

---

## Attribution

- Pixel cat sprites: [last-tick — Animated Pixel Kittens](https://last-tick.itch.io/animated-pixel-kittens-cats-32x32). Free-tier palette under the pack's licence.
- Azure Functions traces: Shahrad et al., 2020.
- Huawei SIR-Lab 2025 cold-start dataset: Lin et al., 2025.

