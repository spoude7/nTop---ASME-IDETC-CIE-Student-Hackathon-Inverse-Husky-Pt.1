DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2026 Massachusetts Institute of Technology.

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

---

# BlendedNet++ Structures Dataset

`blendednet++_structural_dataset.csv` — **13,720** unique blended-wing-body (BWB) aircraft
structural designs, each with its FE-evaluated mass, deflection, internal volumes, and a
physically-anchored hot-spot **stress**.

Each row is one design: **24 input parameters** (aircraft shape + flight condition +
internal structure) + `load_case` → **5 simulated outputs**. **30 columns total.**

---

## How the designs were generated

| Stage | Tool / method |
|---|---|
| **Geometry + structure** | Parametric BWB model built in **nTop 5.49.2**. |
| **Design of experiments** | 24-D design space sampled by scrambled **Sobol** sequences (planform+flight seed 0, structural per-tier seed 1), plus stratified tiers biasing toward the near-constraint region. |
| **Aerodynamic loads** | Surface pressure & friction fields predicted by a **FiLM neural-network surrogate** from the exported wing STL at the sampled flight condition, applied as the FE surface load. |
| **Structural solve** | **Linear-elastic Static Structural FE** in nTop, on an adaptive tetrahedral mesh (mesh tolerance = 0.001). Material **aluminium 7075-T6** (yield ≈ 333 MPa; the allowable is **335.3 MPa** = 503 MPa / 1.5 SF). Payload mass = 70 kg. |
| **Load case** | `combined` worst-case inertial vector — **3 g forward + 2.5 g lateral + 5 g down applied simultaneously** (all 13,720 rows). |

A design is "successful" when nTop builds the full FE result. High-but-physical stresses are
**kept on purpose**: the FE is linear-elastic, so over-yield designs report valid elastic
stresses — intentional near-constraint / negative data for ML models.

---

## How the stress is obtained

The FE solve exports a **full stress point cloud** (`FullStressPointMap`: `x, y, z, stress`
per node — coordinates in mm, stress in Pa) together with the mesh nodes of each structural
component (skin, spar/wingbox, rib, fuselage). The `stress` column is a **hot-spot stress
averaged over a fixed 5 mm physical length**, computed from that cloud:

1. **Attribute stress to each component's nodes** by nearest-neighbour lookup into the full
   cloud (`scipy.cKDTree`).
2. **Radius-smooth and take the peak** — for each node, average the stress of *all nodes
   within a 5 mm ball* (`query_ball_point`), then take the **max** of that smoothed field.
   (Evaluated only at the hottest candidate nodes, which is exact for the maximum.)
3. **`stress` = the maximum across the 4 components** (the binding one), reported in **MPa**.

Averaging over a fixed physical length makes the metric **mesh-density-independent** and
robust to numerical **stress singularities**: a single hot node at a sharp corner (whose
value is mesh-dependent and, in the continuum limit, unbounded) is diluted by its cold
neighbours inside the 5 mm ball and drops out of the maximum, while a genuine over-stressed
region (≥ a few mm wide) survives. The 5 mm length is ≈ 3–4 element widths at the 0.001 mesh
tolerance — the classic "hot-spot stress over a characteristic length" of structural practice.
About **41 %** of designs exceed the 335 MPa allowable under this metric.

---

## Columns (30)

### Inputs — planform / aircraft shape (10)
| Column | Units | Meaning |
|---|---|---|
| `C1` | mm | Centreline chord length (overall size; ≈ 2500–4000). |
| `C2/C1`,`C3/C1`,`C4/C1` | – | Chord ratios along the span, relative to C1. |
| `B1/C1`,`B2/C1`,`B3/C1` | – | Span-station ratios, relative to C1. |
| `X3/C1` | – | Longitudinal station ratio. |
| `S1`,`S3` | degrees | Leading / trailing sweep angles. |

### Inputs — flight condition (3)
| Column | Units | Meaning |
|---|---|---|
| `Altitude` | kft | Altitude. |
| `KCAS` | knots | Calibrated airspeed. |
| `AOA` | degrees | Angle of attack. |

### Inputs — internal structure (11)
| Column | Units | Meaning |
|---|---|---|
| `Skin Thickness` | m | Wing skin thickness. |
| `Front Spar Chord %`,`Rear Spar Chord %` | fraction | Chordwise position of front / rear spar. |
| `Spar Thickness` | m | Spar wall thickness. |
| `# of Ribs` | integer | Number of wing ribs. |
| `Rib Thickness` | m | Rib thickness. |
| `Wingbox Cutout` | m | Wingbox cutout size. |
| `# of Fuselage Ribs` | integer | Number of fuselage ribs. |
| `# of Fuselage Spars` | integer | Number of fuselage spars. |
| `Fuselage Struct Thickness` | m | Fuselage structural thickness. |
| `Fuselage Struct Width` | m | Fuselage structural member width. |

### Meta (1)
| Column | Description |
|---|---|
| `load_case` | Inertial load case (`combined` for all 13,720 rows). |

### Outputs — FE results (4)
| Column | Units | Meaning |
|---|---|---|
| `Aircraft Empty Weight` | kg | Structural / empty mass (**minimisation target**). Range ≈ 19–1639. |
| `stress` | MPa | Hot-spot stress over a 5 mm length (**constraint** vs the 335 MPa allowable). Median ≈ 229; ~41 % exceed 335. |
| `Payload Volume` | mm³ | Usable internal payload-bay volume (≈ 0.08–2.0 m³). |
| `Fuel Volume` | mm³ | Usable internal fuel volume. |

---

## Notes & caveats

- **Over-constraint designs are intentional:** ~41 % of rows exceed the 335 MPa allowable —
  the DOE deliberately probes past the constraint to provide near-boundary negative data.
  Filter `stress <= 335.3` for the stress-feasible designs.
- **FE is linear-elastic** (no plasticity), so `stress` is an elastic extrapolation for
  infeasible designs, not a failure prediction.
- **123 numerical-artifact rows** have `stress > 10⁴ MPa` — divergent elastic solves on
  wildly infeasible geometries. Filter `stress < 1e4` (13,597 rows) for statistics or training.
- `stress` is a surface hot-spot averaged over 5 mm; the exact length scale, together with a
  full mesh-convergence check, is the recommended next validation step.
