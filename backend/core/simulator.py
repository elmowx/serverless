"""
Discrete-event simulator for a G/G/k/k serverless loss queue.

One call:
    run(trace, policy, rng=np.random.default_rng(seed)) -> SimResult

Models k function containers with a 4-state FSM
(FREE, WARMING_UP, BUSY, IDLE) and function affinity on warm hits.
Requests are rejected when no container slot is available.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .calibration import HUAWEI_P5_P95, ColdStartPhases
from .types import (
    ContainerStats,
    ContainerState,
    EventType,
    Policy,
    RequestArrival,
    SimEvent,
    SimResult,
    TimelineSegment,
)


@dataclass
class _Container:
    id: int
    state: ContainerState = ContainerState.FREE
    function_id: str = ""
    idle_since_ms: float = 0.0
    idle_token: int = 0
    state_since_ms: float = 0.0
    pending_queue: list[int] = field(default_factory=list)


def _transition(
    c: _Container,
    new_state: ContainerState,
    now_ms: float,
    stats: ContainerStats,
    timeline: list[list[TimelineSegment]] | None = None,
) -> None:
    """Close out time in the current state, then flip to ``new_state``.

    If ``timeline`` is provided, also append the closing interval as a
    :class:`TimelineSegment` to ``timeline[c.id]``. Zero-length segments are
    skipped to avoid visual clutter in the Gantt chart.
    """
    dt = max(0.0, now_ms - c.state_since_ms)
    if c.state is ContainerState.FREE:
        stats.free_ms += dt
    elif c.state is ContainerState.WARMING_UP:
        stats.warming_ms += dt
    elif c.state is ContainerState.BUSY:
        stats.busy_ms += dt
    elif c.state is ContainerState.IDLE:
        stats.idle_ms += dt
    if timeline is not None and dt > 0.0:
        timeline[c.id].append(
            TimelineSegment(
                state=c.state.value, t0_ms=c.state_since_ms, t1_ms=now_ms
            )
        )
    c.state = new_state
    c.state_since_ms = now_ms


def _sample_cold_start_ms(rng: np.random.Generator, phases: ColdStartPhases, mode: str = "full") -> float:
    total = 0.0
    for phase_name, (lo, hi) in phases.items():
        if mode == "prewarm_spinup" and phase_name not in ("env_init", "runtime_start"):
            continue
        if mode == "assign_prewarmed" and phase_name in ("env_init", "runtime_start"):
            continue
        total += float(rng.uniform(lo, hi))
    return total


def _find_warm(containers: list[_Container], function_id: str) -> _Container | None:
    for c in containers:
        if c.state is ContainerState.IDLE and c.function_id == function_id:
            return c
    return None


def _find_warming(containers: list[_Container], function_id: str) -> _Container | None:
    for c in containers:
        if c.state is ContainerState.WARMING_UP and c.function_id == function_id:
            return c
    return None


def _find_reusable(containers: list[_Container]) -> _Container | None:
    """Pick a container for a cold start.

    Preference order:
    1. A *prewarmed* IDLE container (state=IDLE, function_id=""). Using it
       lets us pay only the function-specific cold-start phases
       (``assign_prewarmed`` mode: code_loading + function_init) rather
       than the full stack. If we skipped prewarmed slots here in favour
       of FREE, the prewarm mechanism would be effectively wasted.
    2. A FREE slot (full cold start).
    3. Recycle any IDLE container belonging to another function (also
       full cold start; this evicts the previously-kept-warm image).
    """
    for c in containers:
        if c.state is ContainerState.IDLE and c.function_id == "":
            return c
    for c in containers:
        if c.state is ContainerState.FREE:
            return c
    for c in containers:
        if c.state is ContainerState.IDLE:
            return c
    return None


def _release_idle(c: _Container, now_ms: float, acc_ms: list[float]) -> None:
    if c.state is ContainerState.IDLE:
        acc_ms[0] += max(0.0, now_ms - c.idle_since_ms)


def _enter_idle(
    c: _Container,
    now_ms: float,
    queue: list[SimEvent],
    keep_alive_s: float,
    stats: ContainerStats,
    timeline: list[list[TimelineSegment]] | None = None,
) -> None:
    _transition(c, ContainerState.IDLE, now_ms, stats, timeline)
    c.idle_since_ms = now_ms
    c.idle_token += 1
    heapq.heappush(
        queue,
        SimEvent(
            time_ms=now_ms + keep_alive_s * 1000.0,
            kind=EventType.CONTAINER_EXPIRY,
            container_id=c.id,
            expiry_token=c.idle_token,
        ),
    )


def _maybe_prewarm(
    containers: list[_Container],
    stats_list: list[ContainerStats],
    queue: list[SimEvent],
    now_ms: float,
    policy: Policy,
    rng: np.random.Generator,
    phases: ColdStartPhases,
    timeline: list[list[TimelineSegment]] | None = None,
) -> None:
    if policy.prewarm_threshold >= 1.0:
        return
    busy = sum(1 for c in containers if c.state is ContainerState.BUSY)
    if busy / policy.max_containers < policy.prewarm_threshold:
        return
    free = next((c for c in containers if c.state is ContainerState.FREE), None)
    if free is None:
        return
    _transition(free, ContainerState.WARMING_UP, now_ms, stats_list[free.id], timeline)
    free.function_id = "__prewarm__"
    heapq.heappush(
        queue,
        SimEvent(
            time_ms=now_ms + _sample_cold_start_ms(rng, phases, mode="prewarm_spinup"),
            kind=EventType.SPIN_UP_DONE,
            container_id=free.id,
            request_idx=-1,
        ),
    )


def run(
    trace: list[RequestArrival],
    policy: Policy,
    *,
    rng: np.random.Generator | None = None,
    phases: ColdStartPhases | None = None,
    record_timeline: bool = False,
) -> SimResult:
    """Discrete-event simulation of the G/G/k/k loss queue.

    Semantics worth pinning down (all baseline experiments in the final
    report rely on these):

    - **Per-function pending queue for in-flight warmups.** When a request
      arrives for a function whose container is currently ``WARMING_UP``
      (cold start not yet finished), we queue the new request behind that
      warmup. It attaches to the container and waits for ``WARM_UP_DONE``,
      after which it is processed in FIFO order. This matches the AWS Lambda
      and Azure Functions published behaviour (provisioned concurrency / init phase).
      If no such container exists, we try to pick a ``FREE``/``IDLE``
      container and pay a cold start on it. If none are available, the request
      is counted as ``rejected``.
    - **Prewarming is opportunistic.** On every arrival we check if the
      fraction of busy containers has crossed ``prewarm_threshold``;
      if so and a ``FREE`` container exists, we kick off a cold start on
      it with a synthetic ``__prewarm__`` function id. The container
      then enters ``IDLE`` and is reusable on the next eligible arrival
      (same or different function — the simulator doesn't condition
      prewarm on function id, see §Method in the report).
    - **Keep-alive ticks are lazy** via stale ``expiry_token`` checks: a
      container that gets reclaimed (IDLE → BUSY → IDLE) increments its
      token, and any previously-scheduled ``CONTAINER_EXPIRY`` event
      with a stale token is silently ignored when popped.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if phases is None:
        phases = HUAWEI_P5_P95

    k = policy.max_containers
    containers = [_Container(id=i) for i in range(k)]
    stats_list = [ContainerStats(container_id=i) for i in range(k)]
    timeline: list[list[TimelineSegment]] | None = (
        [[] for _ in range(k)] if record_timeline else None
    )
    latencies: list[float] = []
    warm_hits = 0
    cold_starts = 0
    rejected = 0
    idle_acc = [0.0]

    queue: list[SimEvent] = []
    for idx, req in enumerate(trace):
        heapq.heappush(
            queue,
            SimEvent(
                time_ms=float(req.timestamp_ms),
                kind=EventType.REQUEST_ARRIVAL,
                request_idx=idx,
            ),
        )

    pending_cold: dict[int, int] = {}
    now = 0.0

    while queue:
        ev = heapq.heappop(queue)
        now = ev.time_ms

        if ev.kind is EventType.REQUEST_ARRIVAL:
            req = trace[ev.request_idx]
            warm = _find_warm(containers, req.function_id)
            if warm is not None:
                _release_idle(warm, now, idle_acc)
                _transition(warm, ContainerState.BUSY, now, stats_list[warm.id], timeline)
                stats_list[warm.id].warm_hits += 1
                warm_hits += 1
                heapq.heappush(
                    queue,
                    SimEvent(
                        time_ms=now + req.execution_time_ms,
                        kind=EventType.EXECUTION_DONE,
                        container_id=warm.id,
                        request_idx=ev.request_idx,
                    ),
                )
            else:
                warming = _find_warming(containers, req.function_id)
                if warming is not None:
                    warming.pending_queue.append(ev.request_idx)
                    stats_list[warming.id].warm_hits += 1
                    warm_hits += 1
                else:
                    reuse = _find_reusable(containers)
                    if reuse is None:
                        rejected += 1
                    else:
                        _release_idle(reuse, now, idle_acc)
                        is_prewarmed = (reuse.state is ContainerState.IDLE and reuse.function_id == "")
                        mode = "assign_prewarmed" if is_prewarmed else "full"
                        cold_ms = _sample_cold_start_ms(rng, phases, mode=mode)
                        _transition(reuse, ContainerState.WARMING_UP, now, stats_list[reuse.id], timeline)
                        reuse.function_id = req.function_id
                        stats_list[reuse.id].cold_starts += 1
                        pending_cold[reuse.id] = ev.request_idx
                        heapq.heappush(
                            queue,
                            SimEvent(
                                time_ms=now + cold_ms,
                                kind=EventType.SPIN_UP_DONE,
                                container_id=reuse.id,
                                request_idx=ev.request_idx,
                            ),
                        )
                        cold_starts += 1
            _maybe_prewarm(containers, stats_list, queue, now, policy, rng, phases, timeline)

        elif ev.kind is EventType.SPIN_UP_DONE:
            c = containers[ev.container_id]
            if ev.request_idx == -1:
                c.function_id = ""
                _enter_idle(c, now, queue, policy.keep_alive_s, stats_list[c.id], timeline)
            else:
                req = trace[ev.request_idx]
                _transition(c, ContainerState.BUSY, now, stats_list[c.id], timeline)
                pending_cold.pop(c.id, None)
                heapq.heappush(
                    queue,
                    SimEvent(
                        time_ms=now + req.execution_time_ms,
                        kind=EventType.EXECUTION_DONE,
                        container_id=c.id,
                        request_idx=ev.request_idx,
                    ),
                )

        elif ev.kind is EventType.EXECUTION_DONE:
            c = containers[ev.container_id]
            req = trace[ev.request_idx]
            latencies.append(now - req.timestamp_ms)
            
            if c.pending_queue:
                next_req_idx = c.pending_queue.pop(0)
                next_req = trace[next_req_idx]
                heapq.heappush(
                    queue,
                    SimEvent(
                        time_ms=now + next_req.execution_time_ms,
                        kind=EventType.EXECUTION_DONE,
                        container_id=c.id,
                        request_idx=next_req_idx,
                    ),
                )
            else:
                _enter_idle(c, now, queue, policy.keep_alive_s, stats_list[c.id], timeline)

        elif ev.kind is EventType.CONTAINER_EXPIRY:
            c = containers[ev.container_id]
            if c.state is ContainerState.IDLE and c.idle_token == ev.expiry_token:
                idle_acc[0] += max(0.0, now - c.idle_since_ms)
                _transition(c, ContainerState.FREE, now, stats_list[c.id], timeline)
                c.function_id = ""

    for c in containers:
        if c.state is ContainerState.IDLE:
            idle_acc[0] += max(0.0, now - c.idle_since_ms)
        # Close the final interval so the last in-flight state is visible
        # in the Gantt chart, even though no "new state" follows. We pass
        # the same state so aggregate stats stay correct.
        _transition(c, c.state, now, stats_list[c.id], timeline)

    return SimResult(
        warm_hits=warm_hits,
        cold_starts=cold_starts,
        rejected=rejected,
        latencies_ms=latencies,
        idle_ms=idle_acc[0],
        container_summary=stats_list,
        container_timeline=timeline if timeline is not None else [],
        timeline_end_ms=now,
    )
