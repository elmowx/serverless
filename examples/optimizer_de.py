from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution


class _BudgetExhausted(Exception):
    pass


def optimize(
    objective,
    budget: int,
    bounds: list[tuple[float, float]],
) -> list[float]:
    popsize = 5
    maxiter = max(0, (budget // popsize) - 1)

    state = {"calls": 0, "best_x": None, "best_y": float("inf")}

    def wrapped(x):
        if state["calls"] >= budget:
            raise _BudgetExhausted()
        state["calls"] += 1
        x_list = [float(v) for v in np.asarray(x, dtype=float).ravel()]
        y = float(objective(x_list))
        if y < state["best_y"]:
            state["best_x"] = x_list
            state["best_y"] = y
        return y

    try:
        differential_evolution(
            wrapped,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            init="sobol",
            polish=False,
            tol=0,
            seed=0,
            updating="deferred",
            workers=1,
        )
    except _BudgetExhausted:
        pass

    if state["best_x"] is None:
        return [(lo + hi) / 2.0 for (lo, hi) in bounds]
    return state["best_x"]
