"""Figure 2 -- the mass/L-D Pareto front the search explored, one panel per case."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from .utils import ALLOWABLE, INK, SERIES, TEXT_W, cases, save, tidy

# The cloud is context, so it stays neutral; the things that carry meaning get
# the accents. Blue/orange separate cleanly under colour-vision simulation.
CLOUD = "#c9c8c1"
FRONT, OURS = SERIES[0], SERIES[1]


def dataset_benchmark():
    """Per case, the lightest real dataset aircraft that meets the whole mission.

    Mission-aware on purpose: counted only if it clears the allowable AND
    delivers this mission's payload and fuel minima AND reaches its L/D target.
    Anything looser would flatter us.
    """
    sys.path.insert(0, ".")
    sys.path.insert(0, os.path.join("models", "ld_surrogate"))
    try:
        from bwb.features import load_dataset
        from bwb.objective import PUBLIC_CASES
        from predict_ld import GEOM_KEYS, predict_ld_batch
    except Exception as exc:
        print(f"  skipped dataset benchmark: {exc}")
        return {}

    df, _ = load_dataset()
    geom = df[list(GEOM_KEYS)].to_numpy(float)
    mass = df["Aircraft Empty Weight"].to_numpy(float)
    vpay = df["Payload Volume"].to_numpy(float) / 1e9          # mm^3 -> m^3
    vfuel = df["Fuel Volume"].to_numpy(float) / 1e9
    stress_ok = df["Max Hotspot Stress"].to_numpy(float) <= ALLOWABLE

    out = {}
    for m in PUBLIC_CASES:
        ld = np.asarray(predict_ld_batch(geom, *m.flight), float)
        keep = (stress_ok & np.isfinite(ld) & (ld >= m.ld_target)
                & (vpay >= m.v_payload_target) & (vfuel >= m.v_fuel_target))
        if keep.any():
            j = int(np.argmin(np.where(keep, mass, np.inf)))
            out[m.name] = (float(mass[j]), float(ld[j]))
    return out


def explored(evolution, key):
    """Every accepted design that sat on some generation's non-dominated front."""
    return [(m, l) for gen in evolution[key]
            for m, l in zip(gen["front_mass"], gen["front_ld"])]


def front_of(points):
    """Non-dominated set across the whole run: minimise mass, maximise L/D."""
    front, best = [], -np.inf
    for mass, ld in sorted(points):
        if ld > best:
            front.append((mass, ld))
            best = ld
    return front


def build(evolution, final, alt=None):
    benchmark = dataset_benchmark()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.98))
    for i, (ax, (key, label, _short, _col, target)) in enumerate(zip(axes, cases())):
        points = explored(evolution, key)
        ax.plot(*zip(*points), ls="none", marker="o", ms=2.4, mfc=CLOUD,
                mec="none", alpha=0.45, zorder=2, label="explored designs")
        ax.plot(*zip(*front_of(points)), color=FRONT, lw=1.6, marker="o", ms=3.2,
                mfc=FRONT, mec="none", zorder=4, label="Pareto front")
        ax.axhline(target, color=INK, lw=0.9, ls=(0, (3, 2.5)), zorder=3)
        ax.annotate(f"L/D target {target:g}", (0.97, target), xytext=(0, 3),
                    xycoords=("axes fraction", "data"),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.0, color=INK)

        design = final["cases"][key]
        ax.plot([design["mass"]], [design["ld"]], marker="s", ms=6.2, color=OURS,
                mec="white", mew=1.0, ls="none", zorder=6, label="submitted")
        # Case 1's compliant variant sits just above its submitted marker, so
        # that label goes underneath; the others have room to the right.
        off, ha, va = ((0, -9), "center", "top") if i == 0 else ((9, 0), "left", "center")
        ax.annotate(f"{design['mass']:.1f} kg\nL/D {design['ld']:.2f}",
                    (design["mass"], design["ld"]), xytext=off,
                    textcoords="offset points", ha=ha, va=va, fontsize=6.0,
                    color=INK, fontweight="bold", linespacing=1.3,
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
        if alt and key in alt:
            ax.plot([alt[key]["mass"]], [alt[key]["ld"]], marker="^", ms=6.0,
                    mfc="white", mec=OURS, mew=1.4, ls="none", zorder=6,
                    label="compliant variant")
        if key in benchmark:
            ax.plot(*benchmark[key], marker="x", ms=5.6, color=INK, mew=1.5,
                    ls="none", zorder=5, label="best real design")

        ax.set_xscale("log")
        ax.set_xticks([20, 30, 50, 100, 200, 400])
        ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0f}")
        ax.xaxis.set_minor_formatter(lambda v, _p: "")
        ax.set_xlabel("structural mass (kg)")
        ax.set_title(f"({'abc'[i]})  {label}")
        tidy(ax)

    axes[0].set_ylabel("L/D")
    # Case 2 has no real-aircraft benchmark, so build the key where all five appear.
    axes[2].legend(frameon=True, facecolor="white", edgecolor="none",
                   framealpha=0.92, loc="lower right", handlelength=1.4,
                   borderpad=0.3, labelspacing=0.28, fontsize=5.8)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    save(fig, "fig1_pareto")
