from __future__ import annotations

import numpy as np

from core.types import RequestArrival
from datagen import PoissonGenerator


def test_deterministic_for_same_seed():
    gen = PoissonGenerator()
    a = gen.generate(intensity=0.5, duration_minutes=60, n_functions=10, seed=42)
    b = gen.generate(intensity=0.5, duration_minutes=60, n_functions=10, seed=42)
    assert a == b


def test_different_seed_changes_output():
    gen = PoissonGenerator()
    a = gen.generate(intensity=0.5, duration_minutes=60, n_functions=10, seed=1)
    b = gen.generate(intensity=0.5, duration_minutes=60, n_functions=10, seed=2)
    assert a != b


def test_intensity_monotonicity():
    gen = PoissonGenerator()
    low = gen.generate(intensity=0.15, duration_minutes=120, n_functions=10, seed=0)
    high = gen.generate(intensity=0.9, duration_minutes=120, n_functions=10, seed=0)
    assert len(high) > 2 * len(low)


def test_diurnal_pattern_visible():
    gen = PoissonGenerator()
    trace = gen.generate(intensity=0.5, duration_minutes=1440, n_functions=10, seed=7)
    per_hour = np.zeros(24, dtype=int)
    for r in trace:
        per_hour[r.timestamp_ms // 3_600_000] += 1
    peak = per_hour[10:14].mean()
    trough = per_hour[0:4].mean()
    assert peak >= 2.0 * trough


def test_output_is_sorted_by_timestamp():
    gen = PoissonGenerator()
    trace = gen.generate(intensity=0.4, duration_minutes=30, n_functions=5, seed=3)
    ts = [r.timestamp_ms for r in trace]
    assert ts == sorted(ts)


def test_event_types_and_bounds():
    gen = PoissonGenerator()
    trace = gen.generate(intensity=0.3, duration_minutes=10, n_functions=5, seed=11)
    for r in trace:
        assert isinstance(r, RequestArrival)
        assert r.timestamp_ms >= 0
        assert r.execution_time_ms >= 0.0
        assert r.function_id.startswith("f")


def test_zipf_concentration_on_head():
    gen = PoissonGenerator()
    trace = gen.generate(intensity=0.7, duration_minutes=60, n_functions=20, seed=0)
    from collections import Counter

    counts = Counter(r.function_id for r in trace)
    top3_share = sum(c for _, c in counts.most_common(3)) / sum(counts.values())
    assert top3_share > 0.35


def test_zero_duration_raises():
    import pytest

    gen = PoissonGenerator()
    with pytest.raises(ValueError):
        gen.generate(intensity=0.5, duration_minutes=0, n_functions=5, seed=0)
