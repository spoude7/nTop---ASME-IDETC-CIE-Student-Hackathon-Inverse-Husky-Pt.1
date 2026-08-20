#!/usr/bin/env python3
"""Build every report figure.

    python make_figures.py

Reads results/FINAL_DESIGNS.json (run_final.py) plus study/pareto_evolution.json,
study/tradeoff.json and study/reference_compliant_alternative.json, and writes a
vector PDF plus a 400 dpi PNG per figure into report/. A figure whose input is
missing is skipped rather than failing the run.
"""
from figures import designs, pareto, tradeoff, utils


def main():
    utils.use_style()
    print("building figures")
    sweeps = utils.load("tradeoff.json")
    final = utils.load("FINAL_DESIGNS.json")
    compliant = utils.load("reference_compliant_alternative.json")
    evolution = utils.load("pareto_evolution.json")
    if evolution and final:
        pareto.build(evolution, final, compliant)
    if sweeps:
        tradeoff.build(sweeps)
    if final:
        designs.build(final)


if __name__ == "__main__":
    main()
