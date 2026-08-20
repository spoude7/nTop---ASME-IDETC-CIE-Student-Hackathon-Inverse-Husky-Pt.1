"""Warm-started differential evolution -- the shipped solver.

Signature:

    solve(objective, budget, seed) -> (x21, info)

`budget` is a surrogate-evaluation budget, not a time limit. That is the fair
currency: wall-clock reflects hardware, evaluations reflect the algorithm.

Only DE is included here. It was the strongest performer measured at equal
budget during development, and it is the only solver run_final.py calls; the
alternatives that were tried and rejected are not part of the submission.

The warm start is the substantive part: seeding the population from real,
stress-feasible rows of the provided dataset puts the search inside the feasible
manifold from generation zero, so the budget is spent improving a design rather
than hunting for one that satisfies the stress constraint at all.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import differential_evolution

from .features import (BOUNDS_LIST, INTEGER_IDX, LOWER, UPPER, decode,
                       dataset_population)

N_VAR = len(LOWER)


def _budget_stop(obj, budget):
    def cb(*args, **kwargs):
        return obj.n_evals >= budget
    return cb


def solve_de(obj, budget=20_000, seed=0, popsize=12, init=None):
    """Differential evolution -- produces the design we actually submit."""
    integrality = np.zeros(N_VAR, dtype=bool)
    integrality[INTEGER_IDX] = True
    pop = popsize * N_VAR
    maxiter = max(1, budget // pop)
    t0 = time.perf_counter()
    differential_evolution(
        obj.scipy_wrapper(), bounds=BOUNDS_LIST,
        integrality=integrality, vectorized=True,
        strategy="best1bin", popsize=popsize, maxiter=maxiter,
        mutation=(0.4, 1.0), recombination=0.85, tol=1e-10,
        polish=False, seed=seed,
        init=init if init is not None else "sobol",
        callback=_budget_stop(obj, budget),
    )
    return obj.best_x, {"seconds": time.perf_counter() - t0, "evals": obj.n_evals}


def solve_de_warm(obj, budget=20_000, seed=0, popsize=12, df=None, bias="light_feasible"):
    """DE seeded from real CSV designs instead of a uniform/Sobol draw."""
    if df is None:
        from .features import load_dataset
        df, _ = load_dataset()
    init = dataset_population(df, popsize * N_VAR, seed=seed, bias=bias)
    return solve_de(obj, budget=budget, seed=seed, popsize=popsize, init=init)
