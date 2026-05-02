from __future__ import annotations

import random


def optimize(
    objective,
    budget: int,
    bounds: list[tuple[float, float]],
) -> list[float]:
    rng = random.Random(0)
    best_x: list[float] | None = None
    best_y = float("inf")
    for _ in range(budget):
        x = [rng.uniform(lo, hi) for (lo, hi) in bounds]
        y = float(objective(x))
        if y < best_y:
            best_x, best_y = x, y
    if best_x is None:
        return [(lo + hi) / 2.0 for (lo, hi) in bounds]
    return best_x
