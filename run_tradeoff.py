#!/usr/bin/env python3
"""The two trade studies behind the technical summary's main figure.

    python run_tradeoff.py [--budget 40000] [--seeds 0 1]

Sweep 1 -- RISK. The stress acceptance gate is moved from the raw surrogate
prediction through the conformal quantiles q=0.75/0.90/0.95. Every arm is run
at the same evaluation budget against the same fitted surrogate instance, and
every returned design is re-audited at q=0.95 afterwards, so the arms are
comparable on one scale: what did conservatism cost in loss, and what did it
buy in true margin?

Sweep 2 -- COMPLIANCE. The published loss caps each volume/L-D shortfall at 0.2
while the mass term is unbounded, so an unconstrained optimizer can buy its way
out of a mission requirement. This sweep turns the three soft targets into hard
constraints at a stated fraction of the demanded value and prices the result.

One surrogate fit serves both sweeps: the conformal quantiles are all computed
from the same held-out residual vector at fit time, so moving the gate is a
change of scalar margin, not a refit.

Writes study/tradeoff.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from bwb.features import (FEATURES, STRESS_ALLOWABLE, MassFrontier, load_dataset)
from bwb.objective import PUBLIC_CASES, metrics
from bwb.solvers import solve_de_warm
from bwb.surrogates import Surrogates
from run_final import TracedObjective

GATES = ["raw", 0.75, 0.90, 0.95]
FRACS = [0.0, 0.60, 0.80, 0.90, 0.95, 1.00]
AUDIT_Q = 0.95


def _shortfall(m, mission, frac):
    """Fractional shortfall against frac x each demanded target."""
    relu = lambda a: np.maximum(0.0, a)
    return (relu((frac * mission.ld_target - m["ld"]) / mission.ld_target)
            + relu((frac * mission.v_fuel_target - m["v_fuel"]) / mission.v_fuel_target)
            + relu((frac * mission.v_payload_target - m["v_payload"]) / mission.v_payload_target))


class ComplianceObjective(TracedObjective):
    """Acceptance requires every mission target met to `frac` of its demand.

    Only the extra() hook differs from the shipped objective, so both sweeps are
    scored by exactly the same code path. target_frac=0 is the shipped behaviour:
    targets stay soft, graded by the published loss.
    """

    def __init__(self, *a, target_frac: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.target_frac = target_frac

    def extra(self, m, X):
        if self.target_frac <= 0.0:
            return 0.0, False
        short = _shortfall(m, self.mission, self.target_frac)
        missed = short > 1e-6
        # Graded so the population keeps a route back to the feasible set, plus a
        # step so "nearly compliant" never scores as compliant.
        return 20.0 * short + 2.0 * missed, missed


def _dump(out, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    """Write after every arm: a sweep this long should never be all-or-nothing."""
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="study/tradeoff.json")
    args = ap.parse_args()

    df, groups = load_dataset()
    X = df[FEATURES].to_numpy(float)
    print("fitting surrogates", flush=True)
    t0 = time.perf_counter()
    sur = Surrogates(n_folds=3, conformal_q=AUDIT_Q).fit(df, groups, measure_leakage=False)
    FR = MassFrontier(df)
    d_audit = sur.delta_stress
    print(f"  fitted in {time.perf_counter()-t0:.0f}s; conformal table: " +
          ", ".join(f"q={q}: {10**d:.2f}x -> {eff:.1f} MPa"
                    for q, (d, eff) in sur.conformal_table.items()), flush=True)

    out = {"budget": args.budget, "seeds": args.seeds, "audit_q": AUDIT_Q,
           "allowable_MPa": STRESS_ALLOWABLE,
           "conformal_table": {str(q): {"factor": 10 ** d, "effective_allowable": eff}
                               for q, (d, eff) in sur.conformal_table.items()},
           "risk": {}, "compliance": {}}

    # ---- Sweep 1: the stress acceptance gate -------------------------------
    for gate in GATES:
        key = "raw" if gate == "raw" else f"q{gate:.2f}"
        # The guided penalty always steers on the audit-q bound; only the
        # ACCEPTANCE rule changes between arms, which is the variable under test.
        sur.delta_stress = d_audit if gate == "raw" else sur.conformal_table[gate][0]
        out["risk"][key] = {}
        for m in PUBLIC_CASES:
            best = None
            for sd in args.seeds:
                o = TracedObjective(sur, m, X, frontier=FR, trace=False,
                                    stress_gate=("raw" if gate == "raw" else "conformal"))
                solve_de_warm(o, budget=args.budget, seed=sd, df=df)
                if o.best_x is None:
                    continue
                if best is None or o.best_loss < best[0]:
                    best = (o.best_loss, o.best_x)
            if best is None:
                out["risk"][key][m.name] = None
                print(f"  risk {key:6s} {m.name:24s} no accepted design", flush=True)
                continue
            L, x = best
            mm = metrics(x[None, :], sur, m)
            raw = float(mm["stress"][0])
            # Re-audit at q=0.95 regardless of the gate this arm ran under:
            # stress_conformal = 10^(pred + delta), so rescaling by the delta
            # ratio converts the bound without a second prediction.
            sig95 = float(mm["stress_conformal"][0]) * 10 ** (d_audit - sur.delta_stress)
            out["risk"][key][m.name] = {
                "loss": float(L), "mass": float(mm["mass"][0]),
                "ld": float(mm["ld"][0]), "stress_raw": raw,
                "stress_q95": sig95, "passes_q95": bool(sig95 <= STRESS_ALLOWABLE)}
            print(f"  risk {key:6s} {m.name:24s} loss={L:.4f}  raw={raw:6.1f}  "
                  f"q95={sig95:8.1f} MPa  {'PASS' if sig95 <= STRESS_ALLOWABLE else 'FAIL'}",
                  flush=True)
        _dump(out, args.out)

    # ---- Sweep 2: mission compliance ---------------------------------------
    sur.delta_stress = d_audit
    for frac in FRACS:
        key = f"{frac:.2f}"
        out["compliance"][key] = {}
        for m in PUBLIC_CASES:
            best = None
            for sd in args.seeds:
                o = ComplianceObjective(sur, m, X, frontier=FR, trace=False,
                                        target_frac=frac)
                solve_de_warm(o, budget=args.budget, seed=sd, df=df)
                if o.best_x is None:
                    continue
                if best is None or o.best_loss < best[0]:
                    best = (o.best_loss, o.best_x)
            if best is None:
                out["compliance"][key][m.name] = None
                print(f"  comply {frac:4.2f} {m.name:24s} INFEASIBLE", flush=True)
                continue
            L, x = best
            mm = metrics(x[None, :], sur, m)
            out["compliance"][key][m.name] = {
                "loss": float(L), "mass": float(mm["mass"][0]),
                "ld": float(mm["ld"][0]),
                "v_payload": float(mm["v_payload"][0]),
                "v_fuel": float(mm["v_fuel"][0]),
                "stress_q95": float(mm["stress_conformal"][0])}
            print(f"  comply {frac:4.2f} {m.name:24s} loss={L:.4f}  "
                  f"mass={mm['mass'][0]:6.1f} kg", flush=True)
        _dump(out, args.out)

    _dump(out, args.out)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
