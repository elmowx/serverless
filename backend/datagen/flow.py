from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from core.types import RequestArrival


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "flow_v1.pt"
META_PATH = WEIGHTS_DIR / "flow_v1_meta.json"

LAMBDA_MIN = 1.0
LAMBDA_MAX = 50.0

_BOOTSTRAP_JITTER_MS = 5_000.0


class FlowGenerator:
    name = "flow"

    def __init__(self, weights_path: Path = WEIGHTS_PATH, meta_path: Path = META_PATH):
        if not weights_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"flow weights not found at {weights_path}")
        import torch

        from probaforms.models import RealNVP

        self._meta = json.loads(meta_path.read_text())
        self._x_mean = np.asarray(self._meta["x_mean"], dtype=np.float32)
        self._x_std = np.asarray(self._meta["x_std"], dtype=np.float32)

        self._model = RealNVP(
            n_layers=int(self._meta["n_layers"]),
            hidden=tuple(self._meta["hidden"]),
        )
        self._prime_model()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        self._torch = torch

    @staticmethod
    def is_available() -> bool:
        return WEIGHTS_PATH.is_file() and META_PATH.is_file()

    def _prime_model(self) -> None:
        dummy_X = np.zeros((2, int(self._meta["x_dim"])), dtype=np.float32)
        dummy_C = np.zeros((2, int(self._meta["cond_dim"])), dtype=np.float32)
        self._model.n_epochs = 0
        self._model.fit(dummy_X, dummy_C)

    def generate(
        self,
        *,
        intensity: float,
        duration_minutes: int,
        n_functions: int,
        seed: int,
    ) -> list[RequestArrival]:
        if duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        if n_functions <= 0:
            raise ValueError("n_functions must be positive")

        intensity = float(np.clip(intensity, 0.0, 1.0))
        target_rps = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * intensity

        torch = self._torch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        rows = duration_minutes * n_functions
        C = np.empty((rows, 3), dtype=np.float32)
        n_denom = max(1, n_functions - 1)
        for fi in range(n_functions):
            rank_norm = fi / n_denom
            for minute in range(duration_minutes):
                hour = minute / 60.0
                idx = minute * n_functions + fi
                C[idx, 0] = rank_norm
                C[idx, 1] = math.sin(2.0 * math.pi * hour / 24.0)
                C[idx, 2] = math.cos(2.0 * math.pi * hour / 24.0)

        with torch.no_grad():
            sampled = self._model.sample(C)
        if hasattr(sampled, "detach"):
            sampled = sampled.detach().cpu().numpy()
        sampled = np.asarray(sampled, dtype=np.float32) * self._x_std + self._x_mean
        counts_raw = np.clip(np.expm1(sampled[:, 0]), 0.0, None)
        execs = np.clip(np.expm1(sampled[:, 1]), 0.0, None)

        counts_native = rng.poisson(counts_raw)

        func_ids = [f"f{i:03d}" for i in range(n_functions)]
        native: list[RequestArrival] = []
        for minute in range(duration_minutes):
            minute_start = minute * 60_000
            for fi in range(n_functions):
                idx = minute * n_functions + fi
                c_int = int(counts_native[idx])
                if c_int <= 0:
                    continue
                mean_exec = max(1.0, float(execs[idx]))
                exec_times = rng.exponential(scale=mean_exec, size=c_int)
                offs = rng.uniform(0.0, 60_000.0, size=c_int)
                fid = func_ids[fi]
                for j, off in enumerate(offs):
                    native.append(
                        RequestArrival(
                            timestamp_ms=int(minute_start + off),
                            function_id=fid,
                            execution_time_ms=float(exec_times[j]),
                        )
                    )

        if not native:
            return []

        native_rps = len(native) / (duration_minutes * 60.0)
        scale = target_rps / native_rps if native_rps > 0 else 0.0
        if scale <= 0.0:
            return []

        if scale <= 1.0:
            mask = rng.random(len(native)) < scale
            out = [a for a, keep in zip(native, mask) if keep]
        else:
            n_extra = int(round(len(native) * (scale - 1.0)))
            out = list(native)
            if n_extra > 0:
                picks = rng.integers(0, len(native), size=n_extra)
                jitters = rng.uniform(-_BOOTSTRAP_JITTER_MS, _BOOTSTRAP_JITTER_MS, size=n_extra)
                for i, jt in zip(picks, jitters):
                    src = native[int(i)]
                    out.append(
                        RequestArrival(
                            timestamp_ms=max(0, src.timestamp_ms + int(jt)),
                            function_id=src.function_id,
                            execution_time_ms=src.execution_time_ms,
                        )
                    )

        out.sort(key=lambda r: r.timestamp_ms)
        return out
