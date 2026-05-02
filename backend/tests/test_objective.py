from __future__ import annotations

import numpy as np
import pytest

from core import BlackBoxObjective, Policy, RequestArrival
from core.objective import BOUNDS, policy_from_vector


def _trace(n: int = 300, seed: int = 1) -> list[RequestArrival]:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(100.0, size=n)
    times = np.cumsum(gaps).astype(int)
    execs = rng.exponential(50.0, size=n)
    funcs = [f"f{i % 4}" for i in range(n)]
    return [RequestArrival(int(t), fid, float(ex)) for t, fid, ex in zip(times, funcs, execs)]


def test_empty_trace_raises():
    with pytest.raises(ValueError):
        BlackBoxObjective([])


def test_returns_float_in_reasonable_range():
    obj = BlackBoxObjective(_trace(), seed=0)
    x = np.array([600.0, 10.0, 1.0])
    y = obj(x)
    assert isinstance(y, float)
    assert y >= 0.0
    assert np.isfinite(y)


def test_deterministic_for_same_x():
    obj = BlackBoxObjective(_trace(), seed=0)
    x = np.array([300.0, 8.0, 1.0])
    assert obj(x) == obj(x)


def test_generous_policy_has_higher_cost_term_than_aggressive():
    obj = BlackBoxObjective(
        _trace(400, seed=3), w_latency=0.0, w_cost=1.0, seed=0
    )
    aggressive = np.array([30.0, 20.0, 1.0])
    generous = np.array([1800.0, 10.0, 1.0])
    y_aggr = obj(aggressive)
    y_gen = obj(generous)
    assert y_gen >= y_aggr


def test_policy_from_vector_clamps_to_bounds():
    p = policy_from_vector([-10.0, 100.0, 5.0])
    assert p.keep_alive_s == BOUNDS[0][0]
    assert p.max_containers == int(BOUNDS[1][1])
    assert p.prewarm_threshold == BOUNDS[2][1]


def test_policy_from_vector_rounds_containers():
    p = policy_from_vector([600.0, 10.7, 1.0])
    assert isinstance(p.max_containers, int)
    assert p.max_containers == 11


def test_evaluate_returns_full_result():
    obj = BlackBoxObjective(_trace(200, seed=5), seed=0)
    res = obj.evaluate([600.0, 5.0, 1.0])
    assert res.total_requests == 200


def test_custom_bounds_clip_max_containers():
    """Policy built from the search vector must respect the caller-provided
    cap, not the module-level BOUNDS. Ensures per-run max_containers_cap
    actually narrows the search space."""
    narrow = [(1.0, 1800.0), (1.0, 5.0), (0.1, 1.0)]
    p = policy_from_vector([600.0, 30.0, 1.0], bounds=narrow)
    assert p.max_containers == 5


def test_objective_respects_custom_bounds():
    """When BlackBoxObjective is constructed with a tighter cap, calling
    it on an out-of-cap x must simulate under the clipped policy."""
    trace = _trace(200, seed=7)
    narrow = [(1.0, 1800.0), (1.0, 3.0), (0.1, 1.0)]
    obj = BlackBoxObjective(trace, seed=0, bounds=narrow)
    res = obj.evaluate([600.0, 30.0, 1.0])
    assert len(res.container_summary) <= 3
    assert obj.bounds[1] == (1.0, 3.0)
