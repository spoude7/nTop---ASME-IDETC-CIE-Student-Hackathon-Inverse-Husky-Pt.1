"""Mission profiles, the challenge loss, and the optimizer's guide-rail loss.

Two scoring functions, deliberately kept separate:

  score_exact()  reproduces the organisers' formula literally -- raw predicted
                 stress, hard infinity, no margin. This is what we report.

  Objective()    is what the optimizer sees: conformal stress margin, a *graded*
                 penalty instead of infinity (a flat wall of inf gives a
                 population-based search no direction home), and a novelty
                 penalty that keeps the search inside the surrogate's support.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .features import (DESIGN, FEATURES, MASS_REF, STRESS_ALLOWABLE, MassFrontier,
                       assemble, decode, to_physical)

_LD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "models", "ld_surrogate")
sys.path.insert(0, _LD)
from predict_ld import GEOM_KEYS, predict_ld_batch      # noqa: E402

# Column indices into a 21-wide design vector, in the order the aero model wants.
_AERO_IDX = np.array([DESIGN.index(k) for k in GEOM_KEYS])


@dataclass(frozen=True)
class Mission:
    """One mission profile. This is the pipeline's only input."""
    ld_target: float
    v_payload_target: float      # m^3
    v_fuel_target: float         # m^3
    alt_kft: float
    kcas: float
    aoa_deg: float
    name: str = "mission"

    @property
    def flight(self):
        return self.alt_kft, self.kcas, self.aoa_deg


# The three public practice cases. The scored missions are hidden until 24h out.
PUBLIC_CASES = [
    Mission(6.0, 0.75, 0.45, 15.0, 120.0, 1.0, "case1_high_speed_dash"),
    Mission(10.0, 0.80, 0.45, 15.0, 45.0, 8.0, "case2_max_endurance"),
    Mission(15.0, 1.00, 0.65, 5.0, 220.0, 4.5, "case3_max_capacity"),
]


def sample_missions(n: int, seed: int = 0) -> list[Mission]:
    """Random missions spanning the challenge's stated ranges.

    These are the held-out missions the generalization metric is measured on --
    a stand-in for the hidden test set. No pipeline should ever be tuned on them.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(Mission(
            ld_target=float(rng.uniform(5.0, 16.0)),
            v_payload_target=float(rng.uniform(0.5, 1.1)),
            v_fuel_target=float(rng.uniform(0.30, 0.70)),
            alt_kft=float(rng.uniform(0.5, 18.0)),
            kcas=float(rng.uniform(40.0, 245.0)),
            aoa_deg=float(rng.uniform(-6.0, 14.0)),
            name=f"rand{i:03d}",
        ))
    return out


def aero_ld(X: np.ndarray, mission: Mission) -> np.ndarray:
    geom = to_physical(X)[:, _AERO_IDX]
    with np.errstate(all="ignore"):
        return np.asarray(predict_ld_batch(geom, *mission.flight), dtype=float)


def metrics(X, surrogates, mission) -> dict[str, np.ndarray]:
    """Everything the loss needs, for a batch of designs under one mission."""
    x24 = assemble(X, *mission.flight)
    p = surrogates.predict(x24)
    return {
        "mass": p["mass"],
        "stress": p["stress"],
        "stress_conformal": p["stress_conformal"],
        "v_payload": p["vpay"] / 1e9,          # mm^3 -> m^3
        "v_fuel": p["vfuel"] / 1e9,
        "ld": aero_ld(X, mission),
    }


def score_exact(m: dict, mission: Mission) -> np.ndarray:
    """The organisers' loss, literally. Infinity above the allowable."""
    relu = lambda a: np.maximum(0.0, a)
    loss = (0.4 * m["mass"] / MASS_REF
            + 0.2 * relu((mission.ld_target - m["ld"]) / mission.ld_target)
            + 0.2 * relu((mission.v_fuel_target - m["v_fuel"]) / mission.v_fuel_target)
            + 0.2 * relu((mission.v_payload_target - m["v_payload"]) / mission.v_payload_target))
    return np.where(m["stress"] > STRESS_ALLOWABLE, np.inf, loss)


class Objective:
    """Callable, vectorized, evaluation-counting objective for any solver.

    The counter is what makes cross-pipeline comparison fair: wall-clock depends
    on hardware, but surrogate evaluations are the currency every search spends.
    """

    def __init__(self, surrogates, mission: Mission, X_train=None,
                 novelty_weight: float = 2.0, use_novelty: bool = True,
                 disagree_max: float = 0.10, disagree_weight: float = 5.0,
                 require_targets: bool = False, frontier=None,
                 frontier_alpha: float = 0.8, frontier_weight: float = 10.0):
        self.s = surrogates
        self.mission = mission
        self.n_evals = 0
        self.history: list[tuple[int, float]] = []     # (evals, best-so-far exact loss)
        self.best_loss = np.inf
        self.best_x = None
        self.use_novelty = use_novelty and X_train is not None
        self.novelty_weight = novelty_weight
        # Ensemble disagreement is a far better trust signal than kNN density:
        # measured, real dataset rows sit at ~0.018 (0.052 in the lightest bin),
        # while an unconstrained optimizer returns designs at ~0.74. In 21 free
        # dimensions kNN distance barely discriminates; fold spread does.
        self.disagree_max = disagree_max
        self.disagree_weight = disagree_weight
        # require_targets=True turns the three SOFT targets into HARD constraints.
        # The organisers' loss caps each shortfall at 0.2 while mass is unbounded,
        # so an unconstrained optimizer buys its way out of the requirements --
        # case 3 delivered 30% of the demanded payload. This flag produces the
        # engineering answer instead of the loss-optimal one, and the difference
        # between the two IS the trade study.
        self.require_targets = require_targets
        # Active during the WHOLE search, not just at initialisation. Measured:
        # dataset-seeded starts put 98% of the population inside the support, but
        # by the final generation the designs were no more credible than a random
        # start -- initialisation sets where you begin, not where you end.
        self.frontier = frontier
        self.frontier_alpha = frontier_alpha
        self.frontier_weight = frontier_weight
        if self.use_novelty:
            # Novelty is measured in the full 24-D feature space: a design can be
            # ordinary on its own yet unprecedented at this flight condition.
            X_train = np.asarray(X_train, dtype=float)   # RAW 24-col features
            self._mu = X_train.mean(0)
            self._sd = X_train.std(0) + 1e-12
            Z = (X_train - self._mu) / self._sd
            self._nn = NearestNeighbors(n_neighbors=10).fit(Z)
            d, _ = self._nn.kneighbors(Z)
            self._nov_ref = float(np.quantile(d.mean(axis=1), 0.95))

    def novelty(self, X) -> np.ndarray:
        if not self.use_novelty:
            return np.zeros(len(np.atleast_2d(X)))
        x24 = assemble(X, *self.mission.flight)
        d, _ = self._nn.kneighbors((x24 - self._mu) / self._sd)
        return np.maximum(0.0, d.mean(axis=1) - self._nov_ref)

    def __call__(self, X) -> np.ndarray:
        X = decode(X)
        self.n_evals += len(X)
        m = metrics(X, self.s, self.mission)

        relu = lambda a: np.maximum(0.0, a)
        loss = (0.4 * m["mass"] / MASS_REF
                + 0.2 * relu((self.mission.ld_target - m["ld"]) / self.mission.ld_target)
                + 0.2 * relu((self.mission.v_fuel_target - m["v_fuel"]) / self.mission.v_fuel_target)
                + 0.2 * relu((self.mission.v_payload_target - m["v_payload"]) / self.mission.v_payload_target))
        loss = np.where(np.isfinite(loss), loss, 1e6)

        # Graded stress penalty -- gives the population a gradient back to feasible.
        over = relu(m["stress_conformal"] - STRESS_ALLOWABLE)
        guided = loss + 10.0 * (over / STRESS_ALLOWABLE) + 100.0 * (over > 0)
        novelty_v = self.novelty(X)
        guided = guided + self.novelty_weight * novelty_v

        # Trust region in prediction space, not input space.
        x24 = assemble(X, *self.mission.flight)
        disagree = self.s.disagreement(x24, "mass")
        guided = guided + self.disagree_weight * relu(disagree - self.disagree_max)

        # Hard-constraint mode: shortfalls become expensive but still graded, so
        # the population can find its way back to the feasible region.
        short = (relu((self.mission.ld_target - m["ld"]) / self.mission.ld_target)
                 + relu((self.mission.v_fuel_target - m["v_fuel"]) / self.mission.v_fuel_target)
                 + relu((self.mission.v_payload_target - m["v_payload"]) / self.mission.v_payload_target))
        if self.require_targets:
            guided = guided + 20.0 * short + 2.0 * (short > 1e-6)

        # Empirical-plausibility gate: is this mass achievable at these volumes?
        if self.frontier is not None:
            fs = self.frontier.shortfall(m["mass"], m["v_payload"], m["v_fuel"],
                                         self.frontier_alpha)
            guided = guided + self.frontier_weight * fs
        else:
            fs = np.zeros(len(X))

        # Track the honest score -- but only accept designs the surrogate is
        # actually entitled to score. Without this the "best" design is simply
        # whichever one exploited the surrogate hardest.
        exact = score_exact(m, self.mission)
        if self.require_targets:
            exact = np.where(short > 1e-6, np.inf, exact)
        exact = np.where((fs > 0.0) |
                         (novelty_v > 0.0) |
                         (disagree > self.disagree_max) |
                         self.s.at_floor(x24, "mass"), np.inf, exact)
        k = int(np.argmin(exact))
        if np.isfinite(exact[k]) and exact[k] < self.best_loss:
            self.best_loss = float(exact[k])
            self.best_x = X[k].copy()
        self.history.append((self.n_evals, self.best_loss))
        return guided

    def scipy_wrapper(self):
        """scipy passes (21, popsize) when vectorized=True."""
        return lambda Xt: self(np.asarray(Xt).T)
