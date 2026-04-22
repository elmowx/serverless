"""
Grid-search optimizer. Uploaded by the user as `solution.py`.

Contract:
    def optimize(objective, budget, bounds): ...

    objective: callable, objective(x: list[float]) -> float (to minimize).
    budget:    int, maximum allowed number of objective calls.
    bounds:    list[(low, high)] per dimension; here len(bounds)==3.
"""

from __future__ import annotations


def optimize(objective, budget: int, bounds: list[tuple[float, float]]) -> list[float]:
    # Calculate grid size per dimension
    n = max(2, round(budget ** (1.0 / 3.0)))
    
    def linspace(lo: float, hi: float, steps: int) -> list[float]:
        if steps <= 1:
            return [lo]
        return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    
    pts = [linspace(lo, hi, n) for lo, hi in bounds]
    combos = [(a, b, c) for a in pts[0] for b in pts[1] for c in pts[2]]
    
    best_x: list[float] | None = None
    best_y = float("inf")
    
    for i, x in enumerate(combos):
        if i >= budget:
            break
        y = objective(list(x))
        if y < best_y:
            best_x, best_y = list(x), y
            
    assert best_x is not None
    return best_x