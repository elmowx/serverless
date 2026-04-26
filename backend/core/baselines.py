from __future__ import annotations

from .types import Policy

_DEFAULT_MAX_WAIT_MS = 30_000.0

BASELINES: dict[str, Policy] = {
    "aggressive": Policy(
        keep_alive_s=30.0, max_containers=20, prewarm_threshold=1.0,
        max_wait_ms=_DEFAULT_MAX_WAIT_MS,
    ),
    "balanced": Policy(
        keep_alive_s=300.0, max_containers=10, prewarm_threshold=1.0,
        max_wait_ms=_DEFAULT_MAX_WAIT_MS,
    ),
    "generous": Policy(
        keep_alive_s=1800.0, max_containers=10, prewarm_threshold=1.0,
        max_wait_ms=_DEFAULT_MAX_WAIT_MS,
    ),
    "prewarm_heavy": Policy(
        keep_alive_s=600.0, max_containers=10, prewarm_threshold=0.6,
        max_wait_ms=_DEFAULT_MAX_WAIT_MS,
    ),
    "minimal": Policy(
        keep_alive_s=60.0, max_containers=3, prewarm_threshold=1.0,
        max_wait_ms=_DEFAULT_MAX_WAIT_MS,
    ),
}


def with_max_wait(policy: Policy, max_wait_ms: float) -> Policy:
    return Policy(
        keep_alive_s=policy.keep_alive_s,
        max_containers=policy.max_containers,
        prewarm_threshold=policy.prewarm_threshold,
        max_wait_ms=max_wait_ms,
    )
