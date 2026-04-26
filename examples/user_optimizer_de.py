"""
Reference solver based on scipy.optimize.differential_evolution.

Strategy:
    Population-based stochastic search. With popsize=5 and a 3-dim cube,
    each generation costs 5 objective calls; we set maxiter so that the
    total number of evaluations equals exactly ``budget``. Uses a Sobol
    init for an even first generation, polish=False so we don't waste a
    final L-BFGS-B run on top, and seed=0 for reproducibility.

    DE has no native budget cap, so we wrap ``objective`` with a counter
    and raise ``_BudgetExhausted`` on the (budget+1)-th call. The exception
    propagates out of ``differential_evolution``; we catch it, return the
    best vector seen so far, and let the runner do the final
    re-simulation.

Why DE: the advisor wants a baseline reference optimizer that is not TPE.
DE is the canonical scipy black-box choice; it makes a fair comparison
against the Optuna-based ``user_optimizer_super.py``.
"""

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
