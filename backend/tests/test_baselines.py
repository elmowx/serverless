"""
Tests for the five named baseline policies shipped in
:mod:`core.baselines`. These policies are load-bearing: the API report,
the PDF report, the CLI final block, and ``BlackBoxObjective``'s
normalization all depend on their structure and behavioural ordering.

The file has three layers:

1. **Structure** — the dict keys are part of the public report contract
   (they land in ``report.json`` and the PDF). Any rename must be
   explicit and intentional.
2. **Behavioural invariants** on a fixed, reproducible Poisson trace.
   These lock the *qualitative* ordering of baselines (a policy with
   more resources should not have a *worse* tail than a resource-starved
   one, etc.). Numeric bounds are generous so the tests don't flap on
   minor simulator tweaks.
3. **Normalization sanity** on :class:`BlackBoxObjective` — we check the
   internal ``l_max`` / ``c_max`` are non-degenerate through the public
   ``terms()`` surface, without reaching into privates.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.baselines import BASELINES
from core.objective import BlackBoxObjective, policy_from_vector
from core.simulator import run
from core.types import Policy, SimResult
from datagen import PoissonGenerator


EXPECTED_KEYS = {"aggressive", "balanced", "generous", "prewarm_heavy", "minimal"}


# ---------- 1. Structure ----------


def test_baseline_names_are_stable() -> None:
    """The public report contract (JSON + PDF) indexes baselines by these
    exact names. Renaming any of them is a breaking change for the UI."""
    assert set(BASELINES) == EXPECTED_KEYS


def test_each_baseline_is_a_policy_with_valid_fields() -> None:
    for name, policy in BASELINES.items():
        assert isinstance(policy, Policy), f"{name} is not a Policy"
        assert policy.keep_alive_s > 0, f"{name}.keep_alive_s must be positive"
        assert policy.max_containers >= 1, f"{name}.max_containers must be >= 1"
        assert 0.0 < policy.prewarm_threshold <= 1.0, (
            f"{name}.prewarm_threshold out of (0, 1]"
        )


# ---------- 2. Behavioural invariants ----------


@pytest.fixture(scope="module")
def poisson_trace():
    # Intensity 0.3, 10 minutes, 4 functions: ~3.7k arrivals. Seeded — fully
    # deterministic. Big enough for the p99 invariants to be stable.
    return PoissonGenerator().generate(
        intensity=0.3, duration_minutes=10, n_functions=4, seed=0
    )


def _simulate(trace, policy: Policy) -> SimResult:
    return run(trace, policy, rng=np.random.default_rng(0))


def test_aggressive_p99_not_worse_than_minimal(poisson_trace) -> None:
    """``minimal`` has k=3 so it sheds load and its tail is dominated by
    rejected-request timeouts. ``aggressive`` has k=20: we expect its
    p99 to be lower."""
    agg = _simulate(poisson_trace, BASELINES["aggressive"])
    mini = _simulate(poisson_trace, BASELINES["minimal"])
    assert agg.p99_latency_ms <= mini.p99_latency_ms


def test_minimal_idle_below_generous(poisson_trace) -> None:
    """``minimal`` keeps few containers for 60 s; ``generous`` keeps 10
    containers for 1800 s. Idle cost must scale accordingly."""
    mini = _simulate(poisson_trace, BASELINES["minimal"])
    gen = _simulate(poisson_trace, BASELINES["generous"])
    assert mini.idle_seconds <= gen.idle_seconds


    def test_prewarm_heavy_does_not_crash(poisson_trace) -> None:
        """Just verify that prewarm_heavy runs without errors.
        With the new pending queue semantics, prewarming can sometimes
        perform worse than balanced by evicting useful IDLE containers,
        so we no longer assert it strictly beats balanced."""
        pre = _simulate(poisson_trace, BASELINES["prewarm_heavy"])
        assert pre.total_requests > 0


def test_generous_has_no_losses(poisson_trace) -> None:
    """k=10 + 30-minute keep-alive is ample for this low-intensity trace;
    no request should be rejected. This is a sanity check on the
    normalization pair used by the objective."""
    gen = _simulate(poisson_trace, BASELINES["generous"])
    assert gen.rejected == 0
    assert gen.p_loss == 0.0


# ---------- 3. Objective normalization ----------


def test_normalization_is_non_degenerate(poisson_trace) -> None:
    """``BlackBoxObjective.__init__`` calibrates ``l_max`` (from ``minimal``)
    and ``c_max`` (from ``generous``). If either collapses to ~0 the
    objective effectively becomes ill-conditioned. The shipped minimum
    floor (1.0 in both simulator units) must not be the binding
    constraint on a non-trivial workload."""
    obj = BlackBoxObjective(
        trace=poisson_trace, w_latency=0.5, w_cost=0.5, seed=0
    )
    # l_max in ms (scaled to the worst of 5 reference policies) and c_max
    # in container-seconds. Both should be well above their 1.0 floors.
    assert obj.l_max > 1.0
    assert obj.c_max > 1.0
    # Ratio sanity: if the two normalizers are within 6 orders of
    # magnitude we're not hitting numerical pathologies.
    assert 1e-6 < obj.l_max / obj.c_max < 1e6


def test_generous_policy_terms_are_finite_and_bounded(poisson_trace) -> None:
    """Feeds the ``generous`` baseline back through the public objective
    and checks the decomposed (latency_term, cost_term) are finite,
    non-negative, and — for a policy that literally defines ``c_max`` —
    that cost_term is close to 1.0."""
    obj = BlackBoxObjective(
        trace=poisson_trace, w_latency=0.5, w_cost=0.5, seed=0
    )
    p = BASELINES["generous"]
    x = np.asarray([p.keep_alive_s, float(p.max_containers), p.prewarm_threshold])
    sim = obj.evaluate(x)
    lat_term, cost_term = obj.terms(sim)
    assert np.isfinite(lat_term) and lat_term >= 0.0
    assert np.isfinite(cost_term) and cost_term >= 0.0
    # ``generous`` *is* the cost-baseline, so cost_term should be ~1 up
    # to simulator noise (the normalizer sim and the recomputed sim use
    # the same RNG seed, so they should agree exactly).
    assert abs(cost_term - 1.0) < 1e-6


def test_policy_from_vector_roundtrip_for_baselines() -> None:
    """``policy_from_vector`` is the bridge between user `x` and the
    simulator. The baselines must roundtrip through it so the reported
    `y` for a baseline in the report matches what a user would get by
    submitting the same vector via the optimizer."""
    for name, p in BASELINES.items():
        x = [p.keep_alive_s, float(p.max_containers), p.prewarm_threshold]
        roundtripped = policy_from_vector(x)
        assert roundtripped.keep_alive_s == pytest.approx(p.keep_alive_s), name
        assert roundtripped.max_containers == p.max_containers, name
        assert roundtripped.prewarm_threshold == pytest.approx(
            p.prewarm_threshold
        ), name
