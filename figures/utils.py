"""Palette, plot style, axis chrome and result-file IO shared by every figure."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bwb.features import STRESS_ALLOWABLE
from bwb.objective import PUBLIC_CASES

RESULTS = "results"      # the submitted deliverable
STUDY = "study"          # supporting measurements behind the report
FIGDIR = "report"        # generated figures

# Inches. The report is A4, two columns, 1.6 cm margins, 0.7 cm column gap.
# Sizing figures to these exactly means \includegraphics never rescales them and
# every label lands at the point size it was set in.
TEXT_W, COL_W = 7.01, 3.37

ALLOWABLE = STRESS_ALLOWABLE

# Categorical slots, used in this order and never cycled. Checked for
# colour-vision separation on a white surface: worst adjacent pair 9.1 simulated,
# 22.9 unsimulated. Two slots fall below 3:1 contrast, so every series also
# carries a direct label -- colour alone never carries identity.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
CRIT = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# One source of truth for case order and targets: the pipeline's own case list.
KEYS = [m.name for m in PUBLIC_CASES]
TARGETS = {m.name: m.ld_target for m in PUBLIC_CASES}
LABELS = ["case 1 · dash", "case 2 · endurance", "case 3 · capacity"]
SHORT = ["case 1", "case 2", "case 3"]
COLORS = list(SERIES[:3])


def use_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 7.3,
        "axes.titlesize": 7.6, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.9, "ytick.labelsize": 6.9,
        "legend.fontsize": 6.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.grid": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def tidy(ax, ygrid=True, xgrid=False, title_pad=4):
    """Drop the top and right rules, put a hairline grid behind the marks, and
    left-align the title.

    set_title(loc="left") makes a second title object rather than moving the
    centred one, so the centred one has to be cleared first.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.tick_params(length=2.5, pad=2)
    title = ax.get_title()
    if title:
        ax.set_title("")
        ax.set_title(title, color=INK, fontweight="bold", loc="left",
                     pad=title_pad)


def cases():
    """(key, label, short label, colour, L/D target) per case, in order."""
    return zip(KEYS, LABELS, SHORT, COLORS, (TARGETS[k] for k in KEYS))


def load(name):
    """Read <name> from results/ or study/, or None if it is not there yet."""
    for folder in (RESULTS, STUDY):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    print(f"  skipped: {name} not found in {RESULTS}/ or {STUDY}/")
    return None


def save(fig, name):
    """Write a vector PDF for the report and a 400 dpi PNG for slides."""
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=400)
    plt.close(fig)
    print(f"  wrote {path}.pdf / .png")
