#!/usr/bin/env python3
"""Final submission pipeline: mission profile in, 21-variable BWB design out.

Usage:
    python run_final.py
    python run_final.py --missions hidden_missions.json

The pipeline uses warm-started differential evolution with 200,000 evaluations
across seeds 0, 1, and 2. Initial populations are seeded from feasible dataset
designs.

The published loss is minimized directly. Stress feasibility is enforced using
a conformal margin on the surrogate prediction, with a final q=0.95 audit.

Outputs are written to:
    results/FINAL_DESIGNS.json
    results/final_designs.csv
    study/pareto_evolution.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from bwb.features import (DESIGN, FEATURES, FUSE_RIB_VALUES, LOWER, STRESS_ALLOWABLE,
                          UPPER, MassFrontier, assemble, decode, load_dataset,
                          to_physical)
from bwb.objective import PUBLIC_CASES, Mission, Objective, metrics, score_exact
from bwb.solvers import solve_de_warm
from bwb.surrogates import Surrogates

DEFAULT_OUT = "results/FINAL_DESIGNS.json"

MISSION_SCHEMA = """[
  {"name": "case1", "ld_target": 6.0, "v_payload_target": 0.75,
   "v_fuel_target": 0.45, "alt_kft": 15.0, "kcas": 120.0, "aoa_deg": 1.0}
]"""


class TracedObjective(Objective):
    """Objective with conformal stress gating and Pareto-front tracing.

    Changes from the base objective:

    1. Stress acceptance. Conformal stress is enforced as a hard feasibility
       gate instead of only contributing to the guided penalty. Use
       stress_gate="raw" to reproduce the original raw-prediction gate.

    2. Front tracing. After each DE population evaluation, only feasible
       Pareto-efficient points in mass and L/D are stored.

    __call__ is implemented directly to avoid evaluating the surrogates twice.
    """

    def __init__(self, *a, trace: bool = False, stress_gate: str = "conformal", **kw):
        super().__init__(*a, **kw)
        self.trace = trace
        self.stress_gate = stress_gate
        self.gens: list[dict] = []

    def extra(self, m, X):
        """Hook: (extra guided penalty, extra acceptance veto).

        Lets a subclass add a constraint without re-implementing __call__ and
        paying for a second surrogate evaluation. run_tradeoff.py uses it to
        make the mission targets hard.
        """
        return 0.0, False

    def __call__(self, X):
        X = decode(X)
        self.n_evals += len(X)
        m = metrics(X, self.s, self.mission)
        relu = lambda a: np.maximum(0.0, a)
        M = self.mission

        loss = (0.4 * m["mass"] / 50.0
                + 0.2 * relu((M.ld_target - m["ld"]) / M.ld_target)
                + 0.2 * relu((M.v_fuel_target - m["v_fuel"]) / M.v_fuel_target)
                + 0.2 * relu((M.v_payload_target - m["v_payload"]) / M.v_payload_target))
        loss = np.where(np.isfinite(loss), loss, 1e6)

        # Steer on the conformal stress so the population migrates to the
        # conservative side of the constraint rather than hugging the raw limit.
        over = relu(m["stress_conformal"] - STRESS_ALLOWABLE)
        guided = loss + 10.0 * (over / STRESS_ALLOWABLE) + 100.0 * (over > 0)
        nov = self.novelty(X)
        guided = guided + self.novelty_weight * nov
        x24 = assemble(X, *M.flight)
        dis = self.s.disagreement(x24, "mass")
        guided = guided + self.disagree_weight * relu(dis - self.disagree_max)
        fs = (self.frontier.shortfall(m["mass"], m["v_payload"], m["v_fuel"],
                                      self.frontier_alpha)
              if self.frontier is not None else np.zeros(len(X)))
        guided = guided + self.frontier_weight * fs

        penalty, veto = self.extra(m, X)
        guided = guided + penalty

        gate_sigma = m["stress"] if self.stress_gate == "raw" else m["stress_conformal"]
        exact = np.where(gate_sigma > STRESS_ALLOWABLE, np.inf,
                         score_exact(m, M))
        exact = np.where((fs > 0.0) | (nov > 0.0) | (dis > self.disagree_max) |
                         self.s.at_floor(x24, "mass") | veto, np.inf, exact)
        k = int(np.argmin(exact))
        if np.isfinite(exact[k]) and exact[k] < self.best_loss:
            self.best_loss = float(exact[k])
            self.best_x = X[k].copy()
        self.history.append((self.n_evals, self.best_loss))

        if self.trace:
            ok = np.isfinite(exact)
            mass, ld = m["mass"][ok], m["ld"][ok]
            front_m, front_l = [], []
            if len(mass):
                order = np.argsort(mass)
                best = -np.inf
                for i in order:
                    if ld[i] > best:
                        front_m.append(float(mass[i])); front_l.append(float(ld[i]))
                        best = ld[i]
            self.gens.append({
                "gen": len(self.gens),
                "evals": int(self.n_evals),
                "n_feasible": int(ok.sum()),
                "front_mass": front_m,
                "front_ld": front_l,
                "best_loss": float(self.best_loss) if np.isfinite(self.best_loss) else None,
            })
        return guided


def verify(row: dict) -> list[str]:
    """Every check here is a way to lose points on a technicality."""
    bad = []
    v = np.array([row[c] for c in DESIGN], dtype=float)
    # "# of Fuselage Ribs" is optimized as an index (0-4) but exported as the
    # corresponding odd rib count (3-11). Validate the exported value using the
    # allowed-value check below rather than the search-space bounds.
    keep = [i for i, c in enumerate(DESIGN) if c != "# of Fuselage Ribs"]
    if not np.all((v[keep] >= LOWER[keep] - 1e-9) & (v[keep] <= UPPER[keep] + 1e-9)):
        bad.append("out of bounds")
    if row["# of Ribs"] % 1 or row["# of Fuselage Spars"] % 1:
        bad.append("non-integer rib/spar count")
    if row["# of Fuselage Ribs"] not in FUSE_RIB_VALUES:
        bad.append("fuselage ribs not in {3,5,7,9,11}")
    if row["Front Spar Chord %"] >= row["Rear Spar Chord %"]:
        bad.append("front spar aft of rear spar")
    if row["max_stress_MPa"] > STRESS_ALLOWABLE:
        bad.append("STRESS VIOLATION")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", help=f"JSON file of mission profiles, shaped like:\n{MISSION_SCHEMA}")
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--q", type=float, default=0.95,
                    help="conformal quantile used as the HARD stress acceptance margin")
    ap.add_argument("--stress-gate", default="conformal", choices=("conformal", "raw"),
                    dest="stress_gate",
                    help="accept on the conformal bound (default) or the raw prediction")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    missions = ([Mission(**d) for d in json.load(open(args.missions))]
                if args.missions else PUBLIC_CASES)

    df, groups = load_dataset()
    X = df[FEATURES].to_numpy(float)

    print(f"[1/4] fitting surrogates at conformal q={args.q}", flush=True)
    sur = Surrogates(n_folds=3, conformal_q=args.q).fit(df, groups, measure_leakage=False)
    FR = MassFrontier(df)
    eff = STRESS_ALLOWABLE / 10 ** sur.delta_stress
    print(f"      conformal factor {10**sur.delta_stress:.2f}x  ->  effective allowable "
          f"{eff:.1f} MPa on the prediction", flush=True)

    out, rows, evo = {}, [], {}
    for m in missions:
        print(f"\n[2/4] {m.name}   L/D>={m.ld_target:g}  Vp>={m.v_payload_target:g}  "
              f"Vf>={m.v_fuel_target:g}  @ {m.alt_kft:g}kft/{m.kcas:g}kt/{m.aoa_deg:g}deg", flush=True)
        pool = []
        for sd in args.seeds:
            o = TracedObjective(sur, m, X, require_targets=False, frontier=FR,
                                trace=True, stress_gate=args.stress_gate)
            t0 = time.perf_counter()
            x, _ = solve_de_warm(o, budget=args.budget, seed=sd, df=df)
            dt = time.perf_counter() - t0
            if x is None:
                print(f"      de_warm seed {sd}: no accepted design ({dt:.0f}s)", flush=True)
                continue
            mm = metrics(x[None, :], sur, m)
            L = float(score_exact(mm, m)[0])
            print(f"      de_warm seed {sd}: loss={L:.4f}  mass={mm['mass'][0]:6.1f} kg  "
                  f"sigma={mm['stress'][0]:5.1f} MPa  ({dt:.0f}s)", flush=True)
            pool.append((L, sd, x, mm, o.gens))

        if not pool:
            print(f"      FAILED on {m.name}", flush=True)
            continue

        L, sd, x, mm, gens = min(pool, key=lambda t: t[0])
        evo[m.name] = gens          # the WINNING seed's trace, not seed 0's
        phys = to_physical(x[None, :])[0]
        fmin = float(FR.min_mass(mm["v_payload"][0], mm["v_fuel"][0]))
        x24 = assemble(x[None, :], *m.flight)
        met = int(sum([bool(mm["ld"][0] >= m.ld_target),
                       bool(mm["v_payload"][0] >= m.v_payload_target),
                       bool(mm["v_fuel"][0] >= m.v_fuel_target)]))
        out[m.name] = {
            "solver": f"de_warm@{args.budget//1000}k", "seed": sd, "loss": L,
            "mass": float(mm["mass"][0]), "ld": float(mm["ld"][0]),
            "v_payload": float(mm["v_payload"][0]), "v_fuel": float(mm["v_fuel"][0]),
            "stress": float(mm["stress"][0]),
            "stress_conformal_at_run_q": float(mm["stress_conformal"][0]),
            "run_conformal_q": args.q,
            "frontier_min_mass": fmin, "frontier_ratio": float(mm["mass"][0] / fmin),
            "disagree": float(sur.disagreement(x24, "mass")[0]),
            "targets_met": met,
            "loss_terms": {
                "mass": 0.4 * float(mm["mass"][0]) / 50.0,
                "ld_shortfall": 0.2 * max(0.0, (m.ld_target - float(mm["ld"][0])) / m.ld_target),
                "fuel_shortfall": 0.2 * max(0.0, (m.v_fuel_target - float(mm["v_fuel"][0])) / m.v_fuel_target),
                "payload_shortfall": 0.2 * max(0.0, (m.v_payload_target - float(mm["v_payload"][0])) / m.v_payload_target),
            },
            "design": dict(zip(DESIGN, phys.tolist())),
        }
        rows.append({"case": m.name, "solver": f"de_warm@{args.budget//1000}k",
                     **dict(zip(DESIGN, phys)),
                     "mass_kg": float(mm["mass"][0]), "LD": float(mm["ld"][0]),
                     "payload_volume_m3": float(mm["v_payload"][0]),
                     "fuel_volume_m3": float(mm["v_fuel"][0]),
                     "max_stress_MPa": float(mm["stress"][0]),
                     "loss": L, "targets_met": met})

    # ---- re-audit every winner at the strict quantile, so the risk is explicit
    print(f"\n[3/4] re-auditing the winners at conformal q=0.95", flush=True)
    sur95 = Surrogates(n_folds=3, conformal_q=0.95).fit(df, groups, measure_leakage=False)
    for m in missions:
        if m.name not in out:
            continue
        xv = np.array([[out[m.name]["design"][c] for c in DESIGN]])
        mm = metrics(xv, sur95, m)
        c95 = float(mm["stress_conformal"][0])
        out[m.name]["stress_conformal_q095"] = c95
        out[m.name]["passes_q095"] = bool(c95 <= STRESS_ALLOWABLE)
        print(f"      {m.name:24} q=0.95 bound {c95:6.1f} MPa  "
              f"{'PASS' if c95 <= STRESS_ALLOWABLE else 'OVER'}", flush=True)

    # The CSV and the front trace follow --out, so a hidden-mission run cannot
    # overwrite the submitted results/ files. The defaults are unchanged.
    out_dir = os.path.dirname(args.out) or "."
    stem = os.path.splitext(os.path.basename(args.out))[0].lower()
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    trace_path = ("study/pareto_evolution.json" if args.out == DEFAULT_OUT
                  else os.path.join(out_dir, f"{stem}_pareto_evolution.json"))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    json.dump({"solver": "de_warm", "budget": args.budget, "seeds": args.seeds,
               "conformal_q": args.q, "cases": out},
              open(args.out, "w"), indent=1)
    json.dump(evo, open(trace_path, "w"))
    sub = pd.DataFrame(rows)
    sub.to_csv(csv_path, index=False)

    print(f"\n[4/4] verifying {csv_path} as written to disk", flush=True)
    ok = True
    for i, r in pd.read_csv(csv_path).iterrows():
        problems = verify(r)
        if problems:
            ok = False
            print(f"      row {i} ({r['case']}): {', '.join(problems)}")
    print("      ALL CHECKS PASSED" if ok else "      FIX THE ABOVE BEFORE SUBMITTING")
    print(f"\nmean loss = {sub.loss.mean():.4f}    wrote {args.out}, {csv_path}, "
          f"{trace_path}")


if __name__ == "__main__":
    main()
