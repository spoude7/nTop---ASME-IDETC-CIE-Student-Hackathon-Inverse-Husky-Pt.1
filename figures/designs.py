"""Figure 2 -- the submitted designs: loss decomposition and stress margin."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .utils import (ALLOWABLE, COL_W, CRIT, INK, INK2, KEYS, LABELS, MUTED,
                    SERIES, cases, save, tidy)

TERMS = [("mass", "mass"), ("payload_shortfall", "payload short"),
         ("fuel_shortfall", "fuel short"), ("ld_shortfall", "L/D short")]


def _panel_loss(ax, designs, rows):
    left = np.zeros(len(rows))
    for (term, label), col in zip(TERMS, SERIES):
        width = np.array([designs[k]["loss_terms"][term] for k in KEYS])
        # White edges keep adjacent segments apart.
        ax.barh(rows, width, left=left, height=0.52, color=col, lw=1.1,
                edgecolor="white", label=label, zorder=3)
        for y, w, l in zip(rows, width, left):
            if w > 0.045:
                ax.text(l + w / 2, y, f"{w:.2f}", ha="center", va="center",
                        fontsize=5.8, color="white", fontweight="bold")
        left = left + width
    for y, total in zip(rows, left):
        ax.text(total + 0.008, y, f"{total:.3f}", va="center", fontsize=6.4,
                color=INK, fontweight="bold")

    ax.set_yticks(rows, LABELS)
    ax.set_xlim(0, left.max() * 1.16)
    ax.set_xlabel("loss contribution")
    ax.set_title("(a)  where the loss comes from")
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.40), handlelength=1.0, handleheight=0.9,
              columnspacing=0.8, borderpad=0.0, fontsize=6.0)
    tidy(ax, ygrid=False, xgrid=True)


def _panel_margin(ax, designs, rows):
    for y, (key, _lab, _short, col, _t) in zip(rows, cases()):
        raw = designs[key]["stress"]
        bound = designs[key]["stress_conformal_q095"]
        ax.plot([raw, bound], [y, y], color=col, lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot([raw], [y], "o", ms=4.0, color="white", mec=col, mew=1.3, zorder=4)
        ax.plot([bound], [y], "o", ms=5.0, color=col, mec="white", mew=1.0, zorder=4)
        ax.annotate(f"{raw:.0f}", (raw, y), xytext=(-4, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=6.0, color=MUTED)
        # Case 3's bound nearly touches the limit line, so the label gets a
        # surface-coloured backing to stay readable over it.
        ax.annotate(f"{bound:.0f}", (bound, y), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.0, color=INK, fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=0.6))

    ax.axvline(ALLOWABLE, color=INK, lw=1.0, ls=(0, (3.5, 2.5)), zorder=2)
    ax.annotate(f"{ALLOWABLE:.0f} MPa\nallowable", (ALLOWABLE, rows[0] + 0.72),
                xytext=(4, 0), textcoords="offset points", ha="left", va="top",
                fontsize=6.0, color=INK, fontweight="bold", linespacing=1.3)

    top = designs[KEYS[0]]
    for x, text, col in ((top["stress"], "raw prediction", MUTED),
                         (top["stress_conformal_q095"], "$q$=0.95 bound", INK2)):
        ax.annotate(text, (x, rows[0]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=5.8,
                    color=col)

    ax.set_yticks(rows, LABELS)
    ax.set_xlim(0, ALLOWABLE * 1.42)
    ax.set_ylim(-0.55, 3.05)
    ax.set_xlabel("hot-spot stress  (MPa)")
    ax.set_title("(b)  margin held after the conformal bound")
    tidy(ax, ygrid=False, xgrid=True)


def build(final):
    designs = final["cases"]
    rows = np.arange(len(KEYS))[::-1]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 2.90),
                               gridspec_kw={"height_ratios": [1.0, 0.95]})
    _panel_loss(a, designs, rows)
    _panel_margin(b, designs, rows)
    fig.tight_layout(pad=0.4, h_pad=2.0)
    save(fig, "fig3_designs")
