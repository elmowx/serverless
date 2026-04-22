"""
Parametric synthetic trace generator.

    PoissonGenerator().generate(intensity, duration_minutes, n_functions, seed)

Models a non-homogeneous Poisson process with a diurnal sinusoidal rate
envelope, Zipf-weighted function popularity, and exponential execution
times with median ~64 ms (matching the Azure 2019 aggregate in Checkpoint 1).

Intensity slider maps to a base rate lambda_base:
    intensity=0.0 -> 1 RPS (low)
    intensity=0.5 -> 25 RPS (medium)
    intensity=1.0 -> 50 RPS (bursty)

Diurnal multiplier keeps peak/trough ratio at 4:1 (peak at hour 12,
trough at hour 0) regardless of intensity.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from core.types import RequestArrival

LAMBDA_MIN = 1.0
LAMBDA_MAX = 50.0

EXEC_TIME_MEAN_MS = 64.0 / math.log(2.0)

ZIPF_A = 1.2


def _lambda_at(hour: float, intensity: float) -> float:
    base = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * float(np.clip(intensity, 0.0, 1.0))
    modulation = 1.0 + 0.6 * math.sin(2.0 * math.pi * hour / 24.0 - math.pi / 2.0)
    return base * modulation


def _lambda_max(intensity: float) -> float:
    base = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * float(np.clip(intensity, 0.0, 1.0))
    return base * 1.6  # max of 1.0 + 0.6 * sin(...)


def _zipf_weights(n: int) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=float)
    w = 1.0 / np.power(ranks, ZIPF_A)
    return w / w.sum()


class PoissonGenerator:
    name = "poisson"

    def generate(
        self,
        *,
        intensity: float,
        duration_minutes: int,
        n_functions: int,
        seed: int,
    ) -> list[RequestArrival]:
        if duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        if n_functions <= 0:
            raise ValueError("n_functions must be positive")

        rng = np.random.default_rng(seed)
        weights = _zipf_weights(n_functions)
        func_ids = [f"f{i:03d}" for i in range(n_functions)]

        lam_max = _lambda_max(intensity)
        duration_seconds = duration_minutes * 60.0
        expected_total = lam_max * duration_seconds

        # Generate homogeneous Poisson process with rate lam_max
        n_candidates = rng.poisson(expected_total)
        if n_candidates == 0:
            return []

        # Uniformly distribute candidates in [0, duration_seconds]
        candidate_times_s = rng.uniform(0.0, duration_seconds, size=n_candidates)
        candidate_times_s.sort()

        # Lewis thinning: accept with probability lambda(t) / lambda_max
        hours = candidate_times_s / 3600.0
        # Vectorized lambda_at calculation
        base = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * float(np.clip(intensity, 0.0, 1.0))
        modulations = 1.0 + 0.6 * np.sin(2.0 * math.pi * hours / 24.0 - math.pi / 2.0)
        lams = base * modulations
        
        accept_probs = lams / lam_max
        accepted = rng.uniform(0.0, 1.0, size=n_candidates) < accept_probs

        accepted_times_s = candidate_times_s[accepted]
        n_accepted = len(accepted_times_s)

        if n_accepted == 0:
            return []

        idxs = rng.choice(n_functions, size=n_accepted, p=weights)
        execs = rng.exponential(scale=EXEC_TIME_MEAN_MS, size=n_accepted)

        out: list[RequestArrival] = []
        for t_s, fi, ex in zip(accepted_times_s, idxs, execs):
            out.append(
                RequestArrival(
                    timestamp_ms=int(t_s * 1000.0),
                    function_id=func_ids[int(fi)],
                    execution_time_ms=float(ex),
                )
            )
        return out


def _preview(args: argparse.Namespace) -> None:
    gen = PoissonGenerator()
    trace = gen.generate(
        intensity=args.intensity,
        duration_minutes=args.duration,
        n_functions=args.n_functions,
        seed=args.seed,
    )
    print(f"generated {len(trace)} requests over {args.duration} minutes")
    print(f"unique functions seen: {len({r.function_id for r in trace})}")
    per_hour = [0] * ((args.duration // 60) + 1)
    for r in trace:
        per_hour[r.timestamp_ms // 3_600_000] += 1
    max_v = max(per_hour) if per_hour else 1
    width = 40
    for h, v in enumerate(per_hour):
        bar = "#" * int(width * v / max_v) if max_v > 0 else ""
        print(f"h{h:02d}  {v:6d}  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(description="Preview PoissonGenerator output")
    p.add_argument("--preview", action="store_true", required=True)
    p.add_argument("--intensity", type=float, default=0.5)
    p.add_argument("--duration", type=int, default=1440, help="minutes")
    p.add_argument("--n-functions", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    _preview(args)


if __name__ == "__main__":
    main()
