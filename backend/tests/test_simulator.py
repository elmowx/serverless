from __future__ import annotations

import numpy as np
import pytest

from core import ContainerState, Policy, RequestArrival, run
from core.calibration import HUAWEI_P5_P95
from core.simulator import _Container, _find_reusable
from core.types import SimResult


def _synthetic_trace(
    n: int, rate_per_s: float, n_functions: int = 3, seed: int = 0
) -> list[RequestArrival]:
    rng = np.random.default_rng(seed)
    gaps_ms = rng.exponential(scale=1000.0 / rate_per_s, size=n)
    times = np.cumsum(gaps_ms).astype(int)
    execs = rng.exponential(scale=50.0, size=n)
    funcs = [f"f{i % n_functions}" for i in range(n)]
    return [
        RequestArrival(int(t), fid, float(ex))
        for t, fid, ex in zip(times, funcs, execs)
    ]


def test_empty_trace_gives_zero_metrics():
    res = run([], Policy(), rng=np.random.default_rng(1))
    assert res.total_requests == 0
    assert res.avg_latency_ms == 0.0
    assert res.p99_latency_ms == 0.0
    assert res.cold_start_rate == 0.0
    assert res.p_loss == 0.0


def test_deterministic_with_fixed_seed():
    trace = _synthetic_trace(200, rate_per_s=10.0, seed=42)
    p = Policy(keep_alive_s=30.0, max_containers=5, prewarm_threshold=1.0)
    a = run(trace, p, rng=np.random.default_rng(123))
    b = run(trace, p, rng=np.random.default_rng(123))
    assert a.warm_hits == b.warm_hits
    assert a.cold_starts == b.cold_starts
    assert a.rejected == b.rejected
    assert a.latencies_ms == b.latencies_ms
    assert a.idle_ms == b.idle_ms


def test_warm_hit_on_repeated_function_id():
    # Same function, spaced so first call finishes before the second arrives,
    # but well before keep-alive expires.
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=10.0),
        RequestArrival(timestamp_ms=10_000, function_id="f", execution_time_ms=10.0),
    ]
    policy = Policy(keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0)
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.warm_hits == 1
    assert res.cold_starts == 1
    assert res.rejected == 0


def test_single_container_capacity_causes_rejections_for_different_functions():
    # Tight bursts with k=1 for DIFFERENT functions: second request arrives before first completes.
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=100.0),
        RequestArrival(timestamp_ms=1, function_id="g", execution_time_ms=100.0),
        RequestArrival(timestamp_ms=2, function_id="h", execution_time_ms=100.0),
    ]
    policy = Policy(keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0)
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.rejected >= 1
    assert res.p_loss > 0.0


def test_cold_start_latency_bounded_by_phase_sum():
    trace = [RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=1.0)]
    res = run(trace, Policy(max_containers=1), rng=np.random.default_rng(0))
    lo = sum(lo for lo, _ in HUAWEI_P5_P95.values())
    hi = sum(hi for _, hi in HUAWEI_P5_P95.values())
    assert len(res.latencies_ms) == 1
    total = res.latencies_ms[0]
    # latency = cold + exec; exec=1 ms
    assert lo + 1.0 <= total <= hi + 1.0


def test_keep_alive_expiry_returns_container_to_free_pool():
    # Two calls to different functions, spaced > keep_alive_s apart: second
    # should still succeed (the first container expired, re-used for cold).
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=5.0),
        RequestArrival(timestamp_ms=120_000, function_id="g", execution_time_ms=5.0),
    ]
    policy = Policy(keep_alive_s=30.0, max_containers=1, prewarm_threshold=1.0)
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.cold_starts == 2
    assert res.rejected == 0


def test_warming_up_queues_same_function_pending():
    """Per-function pending queue semantics: when a second arrival for the same
    function comes in while the container for that function is still WARMING_UP,
    the simulator queues the second arrival behind the warmup.

    Trace: two arrivals of "f", 10 ms apart, k=2. Cold start takes ≥
    Huawei-p5 (~250 ms total of all phases), so the second arrival comes
    in while the first container is still warming. We expect cold_starts
    == 1 and warm_hits == 1 (second request waits for the first to finish
    warming, then executes).
    """
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=5.0),
        RequestArrival(timestamp_ms=10, function_id="f", execution_time_ms=5.0),
    ]
    policy = Policy(keep_alive_s=60.0, max_containers=2, prewarm_threshold=1.0)
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.cold_starts == 1, res
    assert res.warm_hits == 1, res
    assert res.rejected == 0, res


def test_warming_up_queues_even_when_no_other_containers_free():
    """Companion to the pending queue test above. With k=1, the second
    concurrent arrival attaches to the warming container and waits,
    so it is not rejected."""
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=5.0),
        RequestArrival(timestamp_ms=10, function_id="f", execution_time_ms=5.0),
    ]
    policy = Policy(keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0)
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.cold_starts == 1, res
    assert res.rejected == 0, res
    assert res.warm_hits == 1, res


def _result(latencies: list[float]) -> SimResult:
    return SimResult(
        warm_hits=0,
        cold_starts=0,
        rejected=0,
        latencies_ms=latencies,
        idle_ms=0.0,
    )


def test_cvar99_averages_exactly_top_one_percent():
    # n=100 -> ceil(100*0.01)=1 -> tail is just the max.
    res = _result([float(i) for i in range(100)])
    assert res.cvar99_latency_ms == 99.0

    # n=1000 -> ceil(1000*0.01)=10 -> mean of top 10 = mean(990..999)=994.5.
    res = _result([float(i) for i in range(1000)])
    assert res.cvar99_latency_ms == 994.5

    # n=50 -> ceil(50*0.01)=1 -> single max.
    res = _result([float(i) for i in range(50)])
    assert res.cvar99_latency_ms == 49.0


def test_cvar99_empty_is_zero():
    res = _result([])
    assert res.cvar99_latency_ms == 0.0


def test_prewarm_increases_warm_hit_rate():
    trace = _synthetic_trace(500, rate_per_s=20.0, n_functions=3, seed=7)
    no_prewarm = Policy(keep_alive_s=120.0, max_containers=8, prewarm_threshold=1.0)
    with_prewarm = Policy(keep_alive_s=120.0, max_containers=8, prewarm_threshold=0.5)
    a = run(trace, no_prewarm, rng=np.random.default_rng(0))
    b = run(trace, with_prewarm, rng=np.random.default_rng(0))
    # Prewarming should never reduce warm hits given the same trace.
    assert b.warm_hits >= a.warm_hits


def test_find_reusable_prefers_prewarmed_idle_over_free():
    """A prewarmed IDLE container (function_id="") must win over a FREE slot,
    otherwise the prewarm machinery is wasted (we would still pay a full
    cold start even though a ready-to-assign container is sitting idle).
    """
    free = _Container(id=0, state=ContainerState.FREE)
    prewarmed = _Container(id=1, state=ContainerState.IDLE, function_id="")
    stale_warm = _Container(id=2, state=ContainerState.IDLE, function_id="x")
    picked = _find_reusable([free, prewarmed, stale_warm])
    assert picked is prewarmed


def test_waiting_queue_absorbs_into_latency():
    """With max_wait_ms > 0 and a single container, an arrival for a different
    function while the first is still busy waits in the global FIFO queue
    instead of being rejected. Wait time is part of the second request's
    end-to-end latency.
    """
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=100.0),
        RequestArrival(timestamp_ms=1, function_id="g", execution_time_ms=50.0),
    ]
    policy = Policy(
        keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0, max_wait_ms=10_000.0
    )
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.rejected == 0, res
    assert res.cold_starts == 2, res
    assert len(res.latencies_ms) == 2
    # Second arrival waited at least until the first cold-start + exec finished.
    assert res.latencies_ms[1] > 100.0, res.latencies_ms


def test_waiting_queue_rejects_on_timeout():
    """With a small max_wait_ms, a queued request whose wait would exceed
    the cap is rejected once the simulator notices (next container free-up)."""
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=100.0),
        RequestArrival(timestamp_ms=1, function_id="g", execution_time_ms=50.0),
    ]
    policy = Policy(
        keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0, max_wait_ms=5.0
    )
    res = run(trace, policy, rng=np.random.default_rng(0))
    assert res.rejected == 1, res
    assert res.cold_starts == 1, res
    assert len(res.latencies_ms) == 1


def test_max_wait_zero_matches_loss_queue_baseline():
    """Default Policy() has max_wait_ms=0.0 → loss-queue behaviour. Explicit
    max_wait_ms=0.0 must produce byte-identical metrics."""
    trace = [
        RequestArrival(timestamp_ms=0, function_id="f", execution_time_ms=100.0),
        RequestArrival(timestamp_ms=1, function_id="g", execution_time_ms=100.0),
        RequestArrival(timestamp_ms=2, function_id="h", execution_time_ms=100.0),
    ]
    p_default = Policy(keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0)
    p_explicit = Policy(
        keep_alive_s=60.0, max_containers=1, prewarm_threshold=1.0, max_wait_ms=0.0
    )
    a = run(trace, p_default, rng=np.random.default_rng(0))
    b = run(trace, p_explicit, rng=np.random.default_rng(0))
    assert a.rejected == b.rejected
    assert a.warm_hits == b.warm_hits
    assert a.cold_starts == b.cold_starts
    assert a.latencies_ms == b.latencies_ms


def test_find_reusable_falls_back_to_free_then_stale_idle():
    free = _Container(id=0, state=ContainerState.FREE)
    stale_warm = _Container(id=1, state=ContainerState.IDLE, function_id="x")
    assert _find_reusable([free, stale_warm]) is free
    assert _find_reusable([stale_warm]) is stale_warm
    busy = _Container(id=0, state=ContainerState.BUSY, function_id="y")
    assert _find_reusable([busy]) is None
