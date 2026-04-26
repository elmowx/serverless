from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


class ContainerState(Enum):
    FREE = "free"
    WARMING_UP = "warming_up"
    BUSY = "busy"
    IDLE = "idle"


class EventType(Enum):
    REQUEST_ARRIVAL = "request_arrival"
    SPIN_UP_DONE = "spin_up_done"
    EXECUTION_DONE = "execution_done"
    CONTAINER_EXPIRY = "container_expiry"


class RequestArrival(NamedTuple):
    timestamp_ms: int
    function_id: str
    execution_time_ms: float


@dataclass(frozen=True, order=True)
class SimEvent:
    time_ms: float
    kind: EventType = field(compare=False)
    container_id: int = field(compare=False, default=-1)
    request_idx: int = field(compare=False, default=-1)
    expiry_token: int = field(compare=False, default=0)


@dataclass
class Policy:
    keep_alive_s: float = 600.0
    max_containers: int = 10
    prewarm_threshold: float = 1.0
    max_wait_ms: float = 0.0


@dataclass
class ContainerStats:
    container_id: int
    busy_ms: float = 0.0
    idle_ms: float = 0.0
    free_ms: float = 0.0
    warming_ms: float = 0.0
    cold_starts: int = 0
    warm_hits: int = 0

    @property
    def total_ms(self) -> float:
        return self.busy_ms + self.idle_ms + self.free_ms + self.warming_ms

    def _frac(self, value_ms: float) -> float:
        total = self.total_ms
        if total <= 0.0:
            return 0.0
        return value_ms / total

    @property
    def busy_frac(self) -> float:
        return self._frac(self.busy_ms)

    @property
    def idle_frac(self) -> float:
        return self._frac(self.idle_ms)

    @property
    def free_frac(self) -> float:
        return self._frac(self.free_ms)

    @property
    def warming_frac(self) -> float:
        return self._frac(self.warming_ms)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "container_id": self.container_id,
            "busy_frac": self.busy_frac,
            "idle_frac": self.idle_frac,
            "free_frac": self.free_frac,
            "warming_frac": self.warming_frac,
            "cold_starts": self.cold_starts,
            "warm_hits": self.warm_hits,
        }


@dataclass
class TimelineSegment:
    state: str
    t0_ms: float
    t1_ms: float


@dataclass
class SimResult:
    warm_hits: int
    cold_starts: int
    rejected: int
    latencies_ms: list[float]
    idle_ms: float
    container_summary: list[ContainerStats] = field(default_factory=list)
    container_timeline: list[list[TimelineSegment]] = field(default_factory=list)
    timeline_end_ms: float = 0.0

    @property
    def total_requests(self) -> int:
        return self.warm_hits + self.cold_starts + self.rejected

    @property
    def served(self) -> int:
        return self.warm_hits + self.cold_starts

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        xs = sorted(self.latencies_ms)
        idx = max(0, int(round(0.99 * (len(xs) - 1))))
        return xs[idx]

    @property
    def cvar99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        xs = sorted(self.latencies_ms)
        k = max(1, int(math.ceil(len(xs) * 0.01)))
        tail = xs[-k:]
        return sum(tail) / len(tail)

    @property
    def cold_start_rate(self) -> float:
        if self.served == 0:
            return 0.0
        return self.cold_starts / self.served

    @property
    def p_loss(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.rejected / self.total_requests

    @property
    def idle_seconds(self) -> float:
        return self.idle_ms / 1000.0
