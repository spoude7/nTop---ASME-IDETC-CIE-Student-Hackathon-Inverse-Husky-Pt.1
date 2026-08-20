# BWB Inverse Design

Automated inverse design for a Blended Wing Body aircraft: a mission profile in,
a 21-variable design vector out — external planform and internal structure
chosen together.

**ASME IDETC/CIE 2026 Student Hackathon** · nTop / MIT DeCoDE

```
mission profile              21-variable design vector
  L/D target                   10 planform  (chord & span ratios,
  payload volume minimum   ->    sweep angles, centreline chord)
  fuel volume minimum          11 structural (skin, spars, ribs,
  altitude / KCAS / AoA          wingbox cutout, fuselage members)
```

---

## Installation

Pure Python, CPU only. No GPU, no external CFD or FEA solver.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Four dependencies for the pipeline itself: `numpy`, `scipy`, `pandas`,
`scikit-learn`. `matplotlib` is also listed, and is used only by
`make_figures.py` — nothing in `bwb/` or `run_final.py` imports it.
Tested on Python 3.12; 3.10+ should work.

## Usage

Run everything from the repository root.

**Reproduce the submitted design candidates** for the three public test cases:

```bash
python run_final.py
```

**Run the hidden test missions.** Write them to a JSON file shaped like the
schema below and pass `--missions`. Nothing else changes:

```bash
python run_final.py --missions hidden_missions.json
```

```json
[
  {"name": "case1", "ld_target": 6.0, "v_payload_target": 0.75,
   "v_fuel_target": 0.45, "alt_kft": 15.0, "kcas": 120.0, "aoa_deg": 1.0}
]
```

**Options**

| flag | default | meaning |
|---|---|---|
| `--missions` | the 3 public cases | JSON file of mission profiles |
| `--budget` | `200000` | surrogate evaluations per seed |
| `--seeds` | `0 1 2` | random seeds; the best design across them is kept |
| `--q` | `0.95` | conformal quantile used as the stress acceptance margin |
| `--out` | `results/FINAL_DESIGNS.json` | output path |

`python run_final.py --help` prints the full schema.

**Runtime.** About **5 minutes per seed per case** — so ~50 minutes for the
default 3 seeds × 3 cases on a normal desktop CPU. Reduce with
`--seeds 0 1` (~35 min) or `--budget 50000` (~13 min), at some cost in quality.

**Outputs**

`results/` holds the submission and nothing else:

| file | contents |
|---|---|
| `results/FINAL_DESIGNS.json` | the 21 design variables per case, plus predicted performance and the loss decomposition |
| `results/final_designs.csv` | the same design vectors as a flat table |

Everything else is kept out of the way: `study/` holds the supporting
measurements (`pareto_evolution.json`, `tradeoff.json`, `benchmark.json`,
`cvae_benchmark.json`, `reference_compliant_alternative.json`) and `report/`
holds the generated figures.

The script re-reads the CSV it just wrote and verifies every design for parameter
bounds, integer rib and spar counts, odd-only fuselage ribs, and spar ordering,
then prints `ALL CHECKS PASSED` or names the offending row.

**The two trade studies and the report figures** are separate entry points: they
cost far more than one design run, and nothing in the submitted designs depends
on them.

```bash
python run_tradeoff.py        # -> study/tradeoff.json
python make_figures.py        # -> report/fig*.pdf and .png
```

## Repository structure

```
run_final.py              the pipeline: mission -> design candidates
run_tradeoff.py           the stress and volume trade studies (figure 2)
make_figures.py           builds every figure
figures/
  utils.py                palette, plot style, axis chrome, result IO
  tradeoff.py             figure 1 -- the three trade-off spaces
  designs.py              figure 2 -- the submitted designs
bwb/
  features.py             column order, bounds, integer/odd snapping,
                          group splits, empirical mass frontier
  surrogates.py           the four fitted forward models, K-fold ensemble,
                          conformal stress margin
  objective.py            Mission, the challenge loss, the guided loss
  solvers.py              warm-started differential evolution
models/ld_surrogate/      the provided L/D aerodynamic surrogate
data/                     the provided BlendedNet++ structures dataset
assets/                   the hackathon logo used on the report header
results/                  THE SUBMISSION: design variables, nothing else
study/                    supporting measurements behind the report
report/                   generated figures
TECHNICAL_SUMMARY.tex     the engineering report (LaTeX source)
```

## Method

There is no analytical inverse, so the pipeline fits a fast forward model of the
physics and searches the design space against it.

**Five forward models.** Four are fitted here from the provided 13,720-row
structures dataset — mass, hot-spot stress, payload volume, fuel volume. The
fifth is the provided L/D aerodynamic surrogate, used unmodified.

**Validation splits on planform group, never at random.** Each planform appears
2–3 times in the dataset with different internal structure, and always at one
flight condition. A random split therefore leaves near-twins of every test row in
training. Splitting on planform identity holds out unseen geometry *and* unseen
flight conditions which is what a hidden mission is. Held-out R²: mass 0.989,
payload volume 0.998, fuel volume 0.994, hot-spot stress 0.746.

**Search is warm-started differential evolution**, 200,000 evaluations per seed
across seeds 0/1/2. The search space is mixed-integer, non-differentiable
through the acceptance gates, and cheap to evaluate in batch, which is the
regime a population method is for. "Warm-started" is one concrete change: the
initial population is built from real, stress-feasible rows of the provided
dataset rather than a Sobol draw, so generation zero already sits inside the
feasible region. Measured against a cold start at equal budget under one
referee, that single change moves mean loss from 0.436 to 0.383 (see
`study/benchmark.json`).

**Stress is enforced against a conformal bound, not the raw prediction.** The
stress model is the weak one, and it under-predicts on 44.8% of held-out designs.
The held-out residual distribution is measured and its 95th percentile becomes an
explicit margin (a factor of 3.65×), which acceptance tests directly. This is a
hard gate rather than a penalty: with the raw prediction as the gate, the
optimizer returns designs predicted near 300 MPa whose 95th-percentile bound
exceeds 1,100 MPa.

**Four further trust gates** reject candidates the surrogate is not entitled to
score — an empirical mass frontier, K-fold ensemble disagreement above 10%,
novelty outside the data support, and bound-pinning. These *raise* the reported
loss rather than lowering it, and the gated number is the one reported.

## Results

Mean loss **0.3831** across the three public cases. All three clear the
335.3 MPa allowable *after* the 3.65× conformal margin.

| case | loss | mass (kg) | L/D | payload (m³) | fuel (m³) | σ raw | σ at q=0.95 |
|---|---|---|---|---|---|---|---|
| 1 · high-speed dash | 0.3450 | 33.9 | 6.76 | 0.698 | 0.316 | 61 | 234 |
| 2 · max endurance | 0.3690 | 38.0 | 9.15 | 0.755 | 0.367 | 63 | 281 |
| 3 · max capacity | 0.4352 | 19.5 | 15.41 | 0.321 | 0.184 | 89 | 317 |

Stresses in MPa. Full detail, including the per-case loss decomposition, is in
`results/FINAL_DESIGNS.json` and in the technical summary.

## Technical summary

`TECHNICAL_SUMMARY.tex` is the LaTeX source for the **3-page** engineering
report required by the problem statement: abstract, the pipeline as a flowchart,
the optimization strategy, how the stress and volumetric constraints are handled,
and the trade-off spaces. It uses only standard TeX Live packages (the flowchart
is TikZ, drawn in the document) and pulls two figures from `results/`. Build it
with:

```bash
python make_figures.py                                    # if report/fig*.pdf are missing
pdflatex TECHNICAL_SUMMARY.tex && pdflatex TECHNICAL_SUMMARY.tex
```

The second pass resolves the figure and table cross-references. To build on
Overleaf, upload `TECHNICAL_SUMMARY.tex` with `report/fig*.pdf` and `assets/`.
The compiled `TECHNICAL_SUMMARY.pdf` is checked in.

## Notes and limitations

- Hot-spot stress is noise-limited. A near-duplicate probe puts the achievable R²
  ceiling near 0.80, and every model backend tried lands at 0.72–0.76. The
  conformal margin manages that uncertainty; it does not remove it, and the
  guarantee is marginal rather than conditional on any individual design.
- 123 rows of the provided dataset are divergent FE solves and are dropped at
  10⁴ MPa. There is no clean gap at that threshold — the largest kept value is
  9,902 MPa and the smallest dropped one is 10,022 MPa — so the cutoff is a
  judgement call. The ~40% of rows above the 335.3 MPa allowable are kept: they
  are the constraint boundary, not bad data.
- The published loss caps each shortfall term at 0.2 while the mass term is
  unbounded, so it rewards a smaller aircraft. Since performance is ranked per
  case on that loss, the pipeline minimises it directly rather than imposing a
  compliance requirement the formula does not contain.
- Predicted stresses are surrogate estimates of a linear-elastic FE result. No
  independent FE validation was possible within the scope of the challenge.
