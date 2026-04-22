"""
Simulated Annealing optimizer. Uploaded by the user as `solution.py`.

Contract:
    def optimize(objective, budget, bounds): ...

    objective: callable, objective(x: list[float]) -> float (to minimize).
    budget:    int, maximum allowed number of objective calls.
    bounds:    list[(low, high)] per dimension; here len(bounds)==3.
"""

from __future__ import annotations

import math
import random


def optimize(objective, budget: int, bounds: list[tuple[float, float]]) -> list[float]:
    if budget <= 0:
        return [lo for lo, hi in bounds]

    rng = random.Random(42)
    
    # Start with a random point
    current_x = [rng.uniform(lo, hi) for lo, hi in bounds]
    current_y = objective(current_x)
    
    best_x, best_y = list(current_x), current_y
    
    t_init = 1.0
    t_min = 0.01
    
    for i in range(1, budget):
        # Exponential cooling schedule
        t = t_init * ((t_min / t_init) ** (i / float(budget - 1)))
        
        # Propose a neighbor (Gaussian step, scaled by 10% of bounds)
        new_x = []
        for j, (lo, hi) in enumerate(bounds):
            step = rng.gauss(0, (hi - lo) * 0.1)
            val = current_x[j] + step
            # Clip to bounds
            new_x.append(max(lo, min(hi, val)))
            
        new_y = objective(new_x)
        
        # Acceptance criterion
        if new_y < current_y or rng.random() < math.exp((current_y - new_y) / t):
            current_x, current_y = new_x, new_y
            if new_y < best_y:
                best_x, best_y = list(new_x), new_y
                
    return best_x