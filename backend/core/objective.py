from __future__ import annotations

import logging
import numpy as np

from .baselines import BASELINES, with_max_wait
from .calibration import HUAWEI_P5_P95, ColdStartPhases
from .simulator import run
from .types import Policy, RequestArrival, SimResult

logger = logging.getLogger(__name__)

BOUNDS: list[tuple[float, float]] = [
    (1.0, 1800.0),
    (1.0, 30.0),
    (0.1, 1.0),
]


def policy_from_vector(
    x: np.ndarray | list[float],
    bounds: list[tuple[float, float]] | None = None,
    max_wait_ms: float = 0.0,
) -> Policy:
    b = bounds if bounds is not None else BOUNDS
    arr = np.asarray(x, dtype=float)
    keep_alive = float(np.clip(arr[0], b[0][0], b[0][1]))
    k_raw = int(round(float(np.clip(arr[1], b[1][0], b[1][1]))))
    prewarm = float(np.clip(arr[2], b[2][0], b[2][1]))
    return Policy(
        keep_alive_s=keep_alive,
        max_containers=max(1, k_raw),
        prewarm_threshold=prewarm,
        max_wait_ms=float(max_wait_ms),
    )


class BlackBoxObjective:
    def __init__(
        self,
        trace: list[RequestArrival],
        *,
        w_latency: float = 0.5,
        w_cost: float = 0.5,
        phases: ColdStartPhases | None = None,
        seed: int = 42,
        bounds: list[tuple[float, float]] | None = None,
        max_wait_ms: float = 0.0,
    ) -> None:
        if not trace:
            raise ValueError("trace must be non-empty")
        self._trace = trace
        self._w_lat = w_latency
        self._w_cost = w_cost
        self._phases = phases if phases is not None else HUAWEI_P5_P95
        self._seed = seed
        self._bounds: list[tuple[float, float]] = (
            list(bounds) if bounds is not None else list(BOUNDS)
        )
        self._max_wait_ms = float(max_wait_ms)
        self._l_max, self._c_max = self._calibrate_norm()

    def _simulate(self, policy: Policy) -> SimResult:
        rng = np.random.default_rng(self._seed)
        return run(self._trace, policy, rng=rng, phases=self._phases)

    def _calibrate_norm(self) -> tuple[float, float]:
        minimal = self._simulate(with_max_wait(BASELINES["minimal"], self._max_wait_ms))
        generous = self._simulate(with_max_wait(BASELINES["generous"], self._max_wait_ms))
        l_max = max(1.0, minimal.cvar99_latency_ms)
        c_max = max(1.0, generous.idle_seconds)
        return l_max, c_max

    def terms(self, res: SimResult) -> tuple[float, float]:
        lat_raw = res.cvar99_latency_ms / self._l_max
        cost_raw = res.idle_seconds / self._c_max
        
        lat_clipped = min(lat_raw, 1.0)
        cost_clipped = min(cost_raw, 1.0)
        
        if lat_raw > 1.0 or cost_raw > 1.0:
            logger.warning(
                f"Objective clipped: latency {lat_raw:.2f} -> {lat_clipped:.2f}, "
                f"cost {cost_raw:.2f} -> {cost_clipped:.2f}"
            )
            
        return lat_clipped, cost_clipped

    def _y(self, res: SimResult) -> float:
        lat, cost = self.terms(res)
        return self._w_lat * lat + self._w_cost * cost

    def __call__(self, x: np.ndarray | list[float]) -> float:
        return self._y(self._simulate(policy_from_vector(x, self._bounds, self._max_wait_ms)))

    def evaluate(self, x: np.ndarray | list[float]) -> SimResult:
        return self._simulate(policy_from_vector(x, self._bounds, self._max_wait_ms))

    def evaluate_with_y(self, x: np.ndarray | list[float]) -> tuple[SimResult, float]:
        res = self._simulate(policy_from_vector(x, self._bounds, self._max_wait_ms))
        return res, self._y(res)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return list(self._bounds)

    @property
    def l_max(self) -> float:
        return self._l_max

    @property
    def c_max(self) -> float:
        return self._c_max

    @property
    def w_latency(self) -> float:
        return self._w_lat

    @property
    def w_cost(self) -> float:
        return self._w_cost

    @property
    def max_wait_ms(self) -> float:
        return self._max_wait_ms
