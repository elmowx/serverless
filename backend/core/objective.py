"""
Black-box objective wrapping the simulator as a scalar-valued function
    f: R^3 -> R
suitable for derivative-free optimizers.

x = (keep_alive_s, max_containers, prewarm_threshold)

Normalized as
    f(x) = w_latency * min(CVaR_0.99 / L_max, 1.0)  +  w_cost * min(idle_seconds / C_max, 1.0)
where L_max and C_max are fixed per-trace references derived from two
extreme policies (minimal resources -> high latency upper bound,
generous resources -> high idle-cost upper bound).

Calibration cost: constructing :class:`BlackBoxObjective` triggers *two*
extra simulations of the trace (one with BASELINES["minimal"], one with
BASELINES["generous"]) before the optimizer issues its first query. On the
default shipping sizes (~60 min of ~10 RPS, ~36k arrivals) this adds ~120 ms
to wall-clock on a laptop and ~25 MB of peak RSS; on a 500k-arrival trace
(our API cap) it is ~1.5 s. We call it once per ``BlackBoxObjective``
instance and cache the results for the lifetime of that instance, so the
overhead is amortized across the entire budget — budget=30 brings it down
to <1% of total run time. If you need to call the objective from a hot
loop and already have normalized references from a sibling instance, pass
them via the public ``l_max``/``c_max`` properties on the sibling rather
than re-instantiating.
"""

from __future__ import annotations

import logging
import numpy as np

from .baselines import BASELINES
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
) -> Policy:
    """Map a 3-dim search vector to a ``Policy``, clipping each dim to
    its search bound. ``bounds`` defaults to the module-level ``BOUNDS``;
    callers that want to narrow the search space (e.g. cap
    ``max_containers`` below 30 for a particular run) pass a custom list.
    """
    b = bounds if bounds is not None else BOUNDS
    arr = np.asarray(x, dtype=float)
    keep_alive = float(np.clip(arr[0], b[0][0], b[0][1]))
    k_raw = int(round(float(np.clip(arr[1], b[1][0], b[1][1]))))
    prewarm = float(np.clip(arr[2], b[2][0], b[2][1]))
    return Policy(
        keep_alive_s=keep_alive,
        max_containers=max(1, k_raw),
        prewarm_threshold=prewarm,
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
    ) -> None:
        if not trace:
            raise ValueError("trace must be non-empty")
        self._trace = trace
        self._w_lat = w_latency
        self._w_cost = w_cost
        self._phases = phases if phases is not None else HUAWEI_P5_P95
        self._seed = seed
        # Per-instance search bounds. Defaults to the module constant;
        # runners that want to restrict e.g. max_containers pass an
        # override. Calibration (L_max, C_max) intentionally still uses
        # the full-range BASELINES so the normalisation stays comparable
        # across runs with different caps.
        self._bounds: list[tuple[float, float]] = (
            list(bounds) if bounds is not None else list(BOUNDS)
        )
        self._l_max, self._c_max = self._calibrate_norm()

    def _simulate(self, policy: Policy) -> SimResult:
        rng = np.random.default_rng(self._seed)
        return run(self._trace, policy, rng=rng, phases=self._phases)

    def _calibrate_norm(self) -> tuple[float, float]:
        minimal = self._simulate(BASELINES["minimal"])
        generous = self._simulate(BASELINES["generous"])
        l_max = max(1.0, minimal.cvar99_latency_ms)
        c_max = max(1.0, generous.idle_seconds)
        return l_max, c_max

    def terms(self, res: SimResult) -> tuple[float, float]:
        """Decomposed objective terms ``(latency_term, cost_term)`` for a given
        simulation result. ``y = w_latency * latency_term + w_cost * cost_term``.
        Handy for reporting; the weights are left to the caller."""
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
        return self._y(self._simulate(policy_from_vector(x, self._bounds)))

    def evaluate(self, x: np.ndarray | list[float]) -> SimResult:
        return self._simulate(policy_from_vector(x, self._bounds))

    def evaluate_with_y(self, x: np.ndarray | list[float]) -> tuple[SimResult, float]:
        """Single simulation that returns both the full result and the
        normalized scalar — avoids simulating twice when the caller wants both."""
        res = self._simulate(policy_from_vector(x, self._bounds))
        return res, self._y(res)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Search-space bounds for this objective, potentially narrower
        than the module-level ``BOUNDS`` if the caller capped one of the
        dimensions at construction time."""
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
