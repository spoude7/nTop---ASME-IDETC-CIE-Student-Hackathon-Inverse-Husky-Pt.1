"""Forward surrogates: mass, log-stress, log payload volume, log fuel volume.

Uses XGBoost when installed, otherwise falls back to sklearn's
HistGradientBoosting (same algorithm family, also supports monotonic
constraints) so the pipeline runs before `pip install xgboost`.

A K-fold ensemble is trained alongside the main models. Its disagreement on a
candidate design is the single most useful trust signal we have: the optimizer
systematically hunts for regions where the surrogate is least constrained by
data, and spread across folds is what exposes that.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from .features import (FEATURES, STRESS_ALLOWABLE, group_folds, group_split,
                       random_split)

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

# Physically motivated monotonicity. Mass must not fall when material is added.
# Deliberately NOT applied to stress: the hotspot can migrate between features,
# so stress-vs-thickness is not reliably monotonic. Verified before use.
MASS_MONOTONIC = {
    "Skin Thickness": 1, "Spar Thickness": 1, "Rib Thickness": 1,
    "# of Ribs": 1, "# of Fuselage Spars": 1,
    "Fuselage Struct Thickness": 1, "Fuselage Struct Width": 1, "C1": 1,
}


# Per-target backends were evaluated using group-split R2 and an extrapolation
# check over 20k random in-bound designs:
#
#   mass    mlp     R2 0.9978
#   vpay    poly2   R2 0.9999
#   vfuel   poly2   R2 0.9977
#   stress  histgb  R2 0.7460
#
# This mixed configuration was not adopted. It was substantially slower
# end-to-end, and the original loss comparison was not fair because each setup
# scored its own designs with different surrogate models.
#
# The default therefore remains all-HistGB until both configurations can be
# compared using one fixed referee model.
DEFAULT_BACKENDS = {"mass": "histgb", "stress": "histgb",
                    "vpay": "histgb", "vfuel": "histgb"}

# Only these can carry the verified monotonicity constraints.
MONOTONE_CAPABLE = {"histgb", "histgb_deep", "xgboost", "lightgbm"}


def _make(monotone: dict[str, int] | None = None, seed: int = 0,
          backend: str = "histgb"):
    """Build one regressor. Monotonic constraints are applied only where the
    backend supports them -- switching a target to mlp/poly2 trades that
    guarantee for accuracy and speed, which is a real choice, not a free win."""
    if monotone and backend in MONOTONE_CAPABLE:
        cst = [0] * len(FEATURES)
        for col, sign in monotone.items():
            cst[FEATURES.index(col)] = sign
        if backend == "xgboost" and HAVE_XGB:
            return xgb.XGBRegressor(
                n_estimators=1200, learning_rate=0.05, max_depth=7,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=2.0, n_jobs=-1, random_state=seed,
                monotone_constraints=tuple(cst))
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=48,
            min_samples_leaf=12, l2_regularization=1.0, random_state=seed,
            monotonic_cst=cst, early_stopping=False)

    from .models import MODEL_ZOO          # local import avoids a cycle
    return MODEL_ZOO[backend](seed)


# target name -> (column, forward transform, inverse transform, monotonic map)
SPECS = {
    "mass":   ("Aircraft Empty Weight", lambda y: y, lambda z: z, MASS_MONOTONIC),
    "stress": ("Max Hotspot Stress", np.log10, lambda z: 10.0 ** z, None),
    "vpay":   ("Payload Volume", np.log10, lambda z: 10.0 ** z, None),
    "vfuel":  ("Fuel Volume", np.log10, lambda z: 10.0 ** z, None),
}


class Surrogates:
    """Four forward models plus a K-fold ensemble and a conformal margin."""

    def __init__(self, n_folds: int = 4, conformal_q: float = 0.90,
                 backends: dict[str, str] | None = None):
        self.n_folds = n_folds
        self.conformal_q = conformal_q
        self.backends = dict(DEFAULT_BACKENDS if backends is None else backends)
        self.conformal_table: dict = {}
        self.models: dict = {}
        self.folds: dict = {}
        self.report: dict = {}
        self.delta_stress: float = 0.0
        self.train_seconds: float = 0.0
        self.backend = ",".join(f"{k}:{v}" for k, v in self.backends.items())

    def fit(self, df, groups, seed: int = 0, measure_leakage: bool = True):
        t0 = time.perf_counter()
        X = df[FEATURES].to_numpy(float)
        tr, te = group_split(df, groups, seed=seed)

        for name, (col, fwd, _inv, mono) in SPECS.items():
            bk = self.backends[name]
            y = fwd(df[col].to_numpy(float))
            m = _make(mono, seed, bk)
            m.fit(X[tr], y[tr])
            self.models[name] = m
            row = {"backend": bk, "r2_group": r2_score(y[te], m.predict(X[te]))}

            if measure_leakage:                    # the leakage gap, for the report
                rtr, rte = random_split(df, seed=seed)
                m2 = _make(mono, seed, bk)
                m2.fit(X[rtr], y[rtr])
                row["r2_random"] = r2_score(y[rte], m2.predict(X[rte]))
                row["leakage_gap"] = row["r2_random"] - row["r2_group"]
            self.report[name] = row

        # Conformal one-sided margin on log-stress: how far truth exceeds prediction.
        # The stress model is by far the weakest of the four, so this margin is
        # expensive -- the quantile is a genuine risk/mass dial, not a formality.
        resid = np.log10(df["Max Hotspot Stress"].to_numpy(float))[te] - \
            self.models["stress"].predict(X[te])
        self.conformal_table = {
            q: (float(np.quantile(resid, q)),
                STRESS_ALLOWABLE / (10.0 ** float(np.quantile(resid, q))))
            for q in (0.50, 0.75, 0.90, 0.95, 0.99)
        }
        self.delta_stress = float(np.quantile(resid, self.conformal_q))
        self.report["stress"]["conformal_q"] = self.conformal_q
        self.report["stress"]["conformal_delta_log10"] = self.delta_stress
        self.report["stress"]["effective_allowable_MPa"] = \
            STRESS_ALLOWABLE / (10.0 ** self.delta_stress)

        # K-fold ensemble -> per-design disagreement, our surrogate-trust metric.
        # Folds are fitted STRICTLY inside the training split. Splitting over the
        # whole dataset lets each fold see most of the held-out set, which makes
        # the ensemble look far more accurate and far more confident than it is.
        for name, (col, fwd, _inv, mono) in SPECS.items():
            yf = fwd(df[col].to_numpy(float))
            ms = []
            for k, (ktr, _kte) in enumerate(group_folds(groups[tr], self.n_folds)):
                mk = _make(mono, seed + 100 + k, self.backends[name])
                mk.fit(X[tr][ktr], yf[tr][ktr])
                ms.append(mk)
            self.folds[name] = ms

        # Observed output ranges. Tree models extrapolate flat, but a monotonic
        # constraint can still walk a prediction off the end of the data -- and a
        # negative predicted mass hands the optimizer an infinite free lunch.
        self.limits = {}
        for name, (col, _f, _i, _m) in SPECS.items():
            v = df[col].to_numpy(float)
            self.limits[name] = (float(v.min()), float(v.max()))

        self.train_seconds = time.perf_counter() - t0
        return self

    def predict(self, x24: np.ndarray) -> dict[str, np.ndarray]:
        """Score with the ENSEMBLE mean, not the single main model.

        The main model can be badly wrong in a region while the fold models
        agree with each other -- measured: main said 16 kg where three folds
        said ~98 kg. A gate that only checks fold-vs-fold spread cannot see
        that, because the folds are consistent. Averaging the main model in
        with them removes the single point of failure, and the spread over
        all K+1 members is what the gate should test.
        """
        out = {}
        for name, (_c, _f, inv, _m) in SPECS.items():
            lo, hi = self.limits.get(name, (-np.inf, np.inf))
            members = [self.models[name].predict(x24)]
            members += [f.predict(x24) for f in self.folds.get(name, [])]
            out[name] = np.clip(inv(np.mean(members, axis=0)), lo, hi)
        out["stress_conformal"] = 10.0 ** (
            self.models["stress"].predict(x24) + self.delta_stress)
        return out

    def disagreement(self, x24: np.ndarray, name: str = "mass") -> np.ndarray:
        """Relative spread across the main model AND the K folds.

        Including the main model is the point: it is the one whose prediction
        the loss is built from, so a design where it disagrees with the folds
        must be rejected even when the folds agree among themselves.
        """
        _c, _f, inv, _m = SPECS[name]
        members = [self.models[name]] + list(self.folds.get(name, []))
        preds = np.stack([inv(m.predict(x24)) for m in members])
        return preds.std(axis=0) / (np.abs(preds.mean(axis=0)) + 1e-9)

    def at_floor(self, x24: np.ndarray, name: str = "mass") -> np.ndarray:
        """True where the ensemble is pinned at the observed output floor.

        A design sitting on the clamp is an artefact of the clamp, not an
        optimum -- the search pushed the surrogate off the end of its data
        and the clip caught it."""
        lo, _hi = self.limits[name]
        _c, _f, inv, _m = SPECS[name]
        members = [self.models[name]] + list(self.folds.get(name, []))
        raw = inv(np.mean([m.predict(x24) for m in members], axis=0))
        return raw <= lo * 1.001

    def print_report(self):
        print(f"\n[surrogates] backend={self.backend}  train={self.train_seconds:.1f}s")
        print(f"{'model':8s} {'backend':>12s} {'R2 (group)':>12s} {'R2 (random)':>12s} {'leakage gap':>12s}")
        for k, v in self.report.items():
            print(f"{k:8s} {v.get('backend',''):>12s} {v['r2_group']:12.4f} "
                  f"{v.get('r2_random', float('nan')):12.4f} "
                  f"{v.get('leakage_gap', float('nan')):12.4f}")
        print(f"\n[stress] the stress model is the weak link -- margin is a real cost:")
        print(f"{'quantile':>10s} {'delta(log10)':>14s} {'effective allowable':>21s}")
        for q, (d, eff) in self.conformal_table.items():
            mark = "  <-- in use" if abs(q - self.conformal_q) < 1e-9 else ""
            print(f"{q:10.2f} {d:14.4f} {eff:18.1f} MPa{mark}")
