"""
Minimal random-search optimizer. Uploaded by the user as `solution.py`.

Contract:
    def optimize(objective, budget, bounds): ...

    objective: callable, objective(x: list[float]) -> float (to minimize).
    budget:    int, maximum allowed number of objective calls.
    bounds:    list[(low, high)] per dimension; here len(bounds)==3.

    Returns:   best x found (ignored by the runner; the sandbox tracks best
               internally from the trajectory of objective() calls).
"""

from __future__ import annotations

import random


def optimize(objective, budget: int, bounds: list[tuple[float, float]]) -> list[float]:
    rng = random.Random()
    best_x: list[float] | None = None
    best_y = float("inf")
    for _ in range(budget):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        y = objective(x)
        if y < best_y:
            best_x, best_y = x, y
    assert best_x is not None
    return best_x
