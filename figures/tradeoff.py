"""Figure 3 -- how the stress and volumetric constraints were handled."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .utils import ALLOWABLE, CRIT, INK, MUTED, TEXT_W, cases, save, tidy

GATES = ["raw", "q=0.75", "q=0.90", "q=0.95"]
GATE_LABELS = {"raw": "raw\nprediction", "q=0.75": "$q$=0.75",
               "q=0.90": "$q$=0.90", "q=0.95": "$q$=0.95"}


def _panel_stress(ax, tradeoff):
    rows = tradeoff["stress_gate"]
    x = np.arange(len(GATES))
    for key, lab, _short, col, _t in cases():
        y = [next((r["stress_q095"] for r in rows
                   if r["gate"] == g and r["case"] == key), np.nan) for g in GATES]
        ax.plot(x, y, color=col, lw=1.5, marker="o", ms=3.8, mfc="white",
                mec=col, mew=1.2, zorder=3, label=lab)
    ax.axhline(ALLOWABLE, color=INK, lw=1.0, ls=(0, (3.5, 2.5)), zorder=2)
    # Bottom-left is the only region no series passes through.
    ax.annotate(f"{ALLOWABLE:.0f} MPa allowable", (0, ALLOWABLE), xytext=(0, -10),
                textcoords="offset points", ha="left", va="top", fontsize=6.0,
                color=INK, fontweight="bold")

    # Mean loss and pass count above each gate. Text, not a second y-scale.
    for xi, gate in zip(x, GATES):
        arm = [r for r in rows if r["gate"] == gate]
        passes = sum(r["passes_q095"] for r in arm)
        clean = passes == len(arm)
        ax.annotate(f"{np.mean([r['loss'] for r in arm]):.3f}\n{passes}/3 pass",
                    (xi, 1.005), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=5.8,
                    color=INK if clean else MUTED,
                    fontweight="bold" if clean else "normal", linespacing=1.3)

    ax.set_yscale("log")
    ax.set_yticks([200, 300, 500, 1000])
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}")
    ax.yaxis.set_minor_formatter(lambda v, _p: "")
    ax.set_xticks(x, [GATE_LABELS[g] for g in GATES])
    ax.set_xlim(-0.4, len(GATES) - 0.6)
    ax.set_xlabel("stress acceptance gate, looser to stricter")
    ax.set_ylabel("true $q$=0.95 stress bound (MPa)")
    ax.set_title("(a)  the stress constraint")
    ax.legend(frameon=False, loc="upper right", handlelength=1.3, borderpad=0.1,
              labelspacing=0.22, fontsize=6.0)
    tidy(ax, title_pad=16)


def _panel_volume(ax, tradeoff):
    rows = tradeoff["tolerance"]
    taus = sorted({r["tau"] for r in rows})
    x = np.arange(len(taus))
    for key, _lab, short, col, _t in cases():
        y = [next((r["loss"] for r in rows if r["tau"] == t and r["case"] == key),
                  np.nan) for t in taus]
        ax.plot(x, y, color=col, lw=1.5, marker="o", ms=3.8, mfc="white",
                mec=col, mew=1.2, zorder=3)
        ax.annotate(short, (x[-1], y[-1]), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.0,
                    color=col, fontweight="bold")
    ax.set_xticks(x, ["soft" if t == 0 else f"{t:.0%}" for t in taus])
    ax.set_xlim(-0.3, len(taus) - 0.35)
    ax.set_xlabel("required fraction of every L/D and volume target")
    ax.set_ylabel("loss achieved")
    ax.set_title("(b)  the volumetric constraint")
    tidy(ax)


def build(tradeoff):
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_W, 1.80))
    _panel_stress(a, tradeoff)
    _panel_volume(b, tradeoff)
    fig.tight_layout(pad=0.4, w_pad=2.2)
    save(fig, "fig2_constraints")
