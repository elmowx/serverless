from __future__ import annotations

import pytest

from core.types import RequestArrival
from datagen.flow import FlowGenerator

if not FlowGenerator.is_available():
    pytest.skip("flow weights not present; skipping FlowGenerator tests", allow_module_level=True)


@pytest.fixture(scope="module")
def gen() -> FlowGenerator:
    return FlowGenerator()


def test_deterministic_for_same_seed(gen):
    a = gen.generate(intensity=0.4, duration_minutes=30, n_functions=5, seed=7)
    b = gen.generate(intensity=0.4, duration_minutes=30, n_functions=5, seed=7)
    assert a == b


def test_intensity_monotonic(gen):
    low = gen.generate(intensity=0.1, duration_minutes=60, n_functions=10, seed=0)
    high = gen.generate(intensity=0.9, duration_minutes=60, n_functions=10, seed=0)
    assert len(high) > 3 * len(low)


def test_output_sorted_and_typed(gen):
    trace = gen.generate(intensity=0.3, duration_minutes=15, n_functions=5, seed=3)
    assert trace, "flow should produce at least some events at intensity 0.3"
    ts = [r.timestamp_ms for r in trace]
    assert ts == sorted(ts)
    for r in trace[:100]:
        assert isinstance(r, RequestArrival)
        assert r.timestamp_ms >= 0
        assert r.execution_time_ms >= 0.0
        assert r.function_id.startswith("f")


def test_zero_duration_raises(gen):
    with pytest.raises(ValueError):
        gen.generate(intensity=0.5, duration_minutes=0, n_functions=5, seed=0)


def test_is_available_returns_bool():
    assert FlowGenerator.is_available() is True


def test_intensity_thinning_preserves_exec_time_marginal(gen):
    """With per-arrival Bernoulli thinning (scale < 1) the exec_time marginal
    of the thinned trace must match the native trace — that's the whole
    point of the fix over post-hoc count-rescaling, which used to distort
    the bucket-level joint. We compare the mean of log1p(exec_ms)
    between two generations at different intensities; they should be
    close (within 5 %) because thinning is a stateless uniform subsample."""
    import statistics

    low = gen.generate(intensity=0.2, duration_minutes=120, n_functions=10, seed=1)
    high = gen.generate(intensity=0.9, duration_minutes=120, n_functions=10, seed=1)
    assert len(low) > 100 and len(high) > 100

    def log_mean(trace):
        import math as _m
        return statistics.fmean(_m.log1p(r.execution_time_ms) for r in trace)

    m_low, m_high = log_mean(low), log_mean(high)
    rel = abs(m_low - m_high) / max(m_low, m_high)
    assert rel < 0.05, (
        f"exec_time marginal drifted {rel:.3f} across intensity scaling "
        f"— thinning should preserve it (low={m_low:.3f}, high={m_high:.3f})"
    )


def test_exec_time_varies_within_minute_bucket(gen):
    """FlowGenerator used to reuse one mean value for every request in a
    (minute, function_id) bucket, which deflated latency variance and
    distorted p99/p999 in the simulator. The fix samples execution_time
    per request from an exponential around the bucket mean — we verify
    at least one minute-function bucket ends up with stdev > 0 here so
    a regression to the old behaviour is caught immediately.
    """
    trace = gen.generate(intensity=0.5, duration_minutes=30, n_functions=4, seed=0)
    assert trace, "flow should emit at least some events at intensity 0.5"
    import collections
    import statistics

    buckets: dict[tuple[int, str], list[float]] = collections.defaultdict(list)
    for r in trace:
        minute = r.timestamp_ms // 60_000
        buckets[(minute, r.function_id)].append(r.execution_time_ms)
    multi = [vals for vals in buckets.values() if len(vals) >= 3]
    assert multi, "need at least one (minute, function) bucket with ≥3 arrivals"
    max_std = max(statistics.pstdev(v) for v in multi)
    assert max_std > 0.0, (
        "execution_time_ms must vary within (minute, function) buckets "
        "after the per-request exponential sampling fix"
    )
