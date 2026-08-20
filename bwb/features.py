"""Column definitions, design-space bounds, and the group split.

Everything downstream imports FEATURES from here so that training and the
optimizer can never disagree about column order -- the classic silent bug in
surrogate pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

PLANFORM = ["C2/C1", "C3/C1", "C4/C1", "B1/C1", "B2/C1",
            "B3/C1", "X3/C1", "S1", "S3", "C1"]
FLIGHT = ["Altitude", "KCAS", "AOA"]
STRUCT = ["Skin Thickness", "Front Spar Chord %", "Rear Spar Chord %",
          "Spar Thickness", "# of Ribs", "Rib Thickness", "Wingbox Cutout",
          "# of Fuselage Ribs", "# of Fuselage Spars",
          "Fuselage Struct Thickness", "Fuselage Struct Width"]

FEATURES = PLANFORM + FLIGHT + STRUCT          # 24 columns, fixed order
DESIGN = PLANFORM + STRUCT                     # 21 free variables
TARGETS = ["Aircraft Empty Weight", "Max Hotspot Stress",
           "Payload Volume", "Fuel Volume"]

STRESS_ALLOWABLE = 335.0      # MPa, Al 7075-T6
MASS_REF = 50.0               # kg, fixed normalizer in the challenge loss
DIVERGENT_STRESS = 1e4        # MPa, above this the FE solve did not converge

# Published design-space bounds. Order matches DESIGN.
BOUNDS = {
    "C2/C1": (0.55, 0.85), "C3/C1": (0.18, 0.28), "C4/C1": (0.06, 0.09),
    "B1/C1": (0.10, 0.20), "B2/C1": (0.05, 0.20), "B3/C1": (0.35, 0.70),
    "X3/C1": (0.50, 0.65), "S1": (40.0, 60.0), "S3": (20.0, 40.0),
    "C1": (2500.0, 4000.0),
    "Skin Thickness": (0.0003, 0.0050),
    "Front Spar Chord %": (0.18, 0.35),
    "Rear Spar Chord %": (0.55, 0.75),
    "Spar Thickness": (0.00098, 0.0080),
    "# of Ribs": (3, 14),
    "Rib Thickness": (0.0015, 0.015),
    "Wingbox Cutout": (0.01, 0.05),
    "# of Fuselage Ribs": (0, 4),        # INDEX into FUSE_RIB_VALUES
    "# of Fuselage Spars": (3, 12),
    "Fuselage Struct Thickness": (0.002, 0.025),
    "Fuselage Struct Width": (0.0010, 0.015),
}

FUSE_RIB_VALUES = np.array([3, 5, 7, 9, 11])   # odd-only constraint

I_RIBS = DESIGN.index("# of Ribs")
I_FUSE_RIBS = DESIGN.index("# of Fuselage Ribs")
I_FUSE_SPARS = DESIGN.index("# of Fuselage Spars")
INTEGER_IDX = [I_RIBS, I_FUSE_RIBS, I_FUSE_SPARS]

LOWER = np.array([BOUNDS[c][0] for c in DESIGN], dtype=float)
UPPER = np.array([BOUNDS[c][1] for c in DESIGN], dtype=float)
BOUNDS_LIST = list(zip(LOWER, UPPER))


def decode(X: np.ndarray) -> np.ndarray:
    """(n, 21) raw search vector -> (n, 21) legal design vector.

    Snapping happens here, inside the objective, so that every design the
    optimizer *scores* is a design it could legally *submit*.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    X = np.clip(X, LOWER, UPPER)
    X[:, I_RIBS] = np.round(X[:, I_RIBS])
    X[:, I_FUSE_SPARS] = np.round(X[:, I_FUSE_SPARS])
    idx = np.clip(np.round(X[:, I_FUSE_RIBS]), 0, 4).astype(int)
    X[:, I_FUSE_RIBS] = idx                       # kept as index internally
    return X


def to_physical(X: np.ndarray) -> np.ndarray:
    """Same as decode, but the fuselage-rib index becomes the real odd count."""
    X = decode(X)
    X[:, I_FUSE_RIBS] = FUSE_RIB_VALUES[X[:, I_FUSE_RIBS].astype(int)]
    return X


def assemble(X: np.ndarray, alt: float, kcas: float, aoa: float) -> np.ndarray:
    """(n, 21) design + scalar flight condition -> (n, 24) in FEATURES order."""
    X = to_physical(X)
    n = X.shape[0]
    out = np.empty((n, len(FEATURES)), dtype=float)
    for j, col in enumerate(FEATURES):
        if col in FLIGHT:
            out[:, j] = {"Altitude": alt, "KCAS": kcas, "AOA": aoa}[col]
        else:
            out[:, j] = X[:, DESIGN.index(col)]
    return out


def load_dataset(path: str = "data/bwb_structures_dataset.csv",
                 drop_divergent: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the CSV and return (dataframe, group_ids).

    Groups are planform identity. Each planform appears at exactly one flight
    condition, so grouping on the planform also holds out unseen flight
    conditions -- which is the generalization the hidden missions demand.
    """
    df = pd.read_csv(path)
    if drop_divergent:
        n0 = len(df)
        df = df[df["Max Hotspot Stress"] < DIVERGENT_STRESS].reset_index(drop=True)
        print(f"[data] dropped {n0 - len(df)} divergent rows (stress > {DIVERGENT_STRESS:g} MPa)")
    groups = df.groupby(PLANFORM, sort=False).ngroup().to_numpy()
    print(f"[data] {len(df)} rows across {groups.max() + 1} planform groups")
    return df, groups


def group_split(df, groups, test_size=0.2, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=groups))


def random_split(df, test_size=0.2, seed=0):
    """Deliberately leaky split, kept so the leakage gap can be measured."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    return idx[:cut], idx[cut:]


def group_folds(groups, n_splits=4):
    return list(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups))


def dataset_population(df, n, seed=0, jitter=0.02, bias="light_feasible"):
    """Build the DE initial population from real dataset designs.

    Seeding from dataset rows keeps the initial population within the region
    covered by the surrogate models.

    bias: "none"           -- uniform sampling from dataset rows
          "light_feasible" -- favor low-mass, stress-feasible designs
    jitter: added noise as a fraction of each variable's range.
    """
    rng = np.random.default_rng(seed)
    d = df
    if bias == "light_feasible":
        d = df[df["Max Hotspot Stress"] <= STRESS_ALLOWABLE]
        if len(d) == 0:
            d = df
        w = 1.0 / (d["Aircraft Empty Weight"].to_numpy(float) + 1.0)
        w = w / w.sum()
        idx = rng.choice(len(d), size=n, replace=True, p=w)
    else:
        idx = rng.choice(len(d), size=n, replace=True)

    rows = d.iloc[idx]
    P = np.empty((n, len(DESIGN)), dtype=float)
    for j, col in enumerate(DESIGN):
        P[:, j] = rows[col].to_numpy(float)

    # The CSV stores the PHYSICAL fuselage-rib count; the search space uses an
    # index into {3,5,7,9,11}, so map it back.
    phys = P[:, I_FUSE_RIBS]
    P[:, I_FUSE_RIBS] = np.argmin(np.abs(phys[:, None] - FUSE_RIB_VALUES[None, :]), axis=1)

    if jitter > 0:
        P = P + rng.normal(0.0, jitter, P.shape) * (UPPER - LOWER)
    return np.clip(P, LOWER, UPPER)


class MassFrontier:
    """Empirical mass lower bound based on delivered payload and fuel volumes.

    The bound is taken from the lightest dataset design providing at least the
    requested volumes. Mass and volume are geometric quantities, so no flight
    condition filtering is required.

    Implemented as a 2-D suffix-minimum grid for O(1) lookup during optimization.
    """

    def __init__(self, df, n_grid: int = 80):
        vp = (df["Payload Volume"].to_numpy(float) / 1e9)
        vf = (df["Fuel Volume"].to_numpy(float) / 1e9)
        m = df["Aircraft Empty Weight"].to_numpy(float)
        self.gvp = np.linspace(vp.min(), vp.max(), n_grid)
        self.gvf = np.linspace(vf.min(), vf.max(), n_grid)

        G = np.full((n_grid, n_grid), np.inf)
        i = np.clip(np.searchsorted(self.gvp, vp, "right") - 1, 0, n_grid - 1)
        j = np.clip(np.searchsorted(self.gvf, vf, "right") - 1, 0, n_grid - 1)
        np.minimum.at(G, (i, j), m)
        # suffix minimum: cell (i,j) = min mass over all rows with vp>=gvp[i], vf>=gvf[j]
        for a in range(n_grid - 2, -1, -1):
            G[a, :] = np.minimum(G[a, :], G[a + 1, :])
        for b in range(n_grid - 2, -1, -1):
            G[:, b] = np.minimum(G[:, b], G[:, b + 1])
        self.G = G
        self.global_min = float(m.min())

    def min_mass(self, v_payload, v_fuel):
        """Lightest REAL design delivering at least these volumes."""
        i = np.clip(np.searchsorted(self.gvp, np.asarray(v_payload), "right") - 1,
                    0, len(self.gvp) - 1)
        j = np.clip(np.searchsorted(self.gvf, np.asarray(v_fuel), "right") - 1,
                    0, len(self.gvf) - 1)
        out = self.G[i, j]
        return np.where(np.isfinite(out), out, self.global_min)

    def shortfall(self, mass, v_payload, v_fuel, alpha=0.8):
        """Relative amount by which a design undercuts the empirical bound.

        0 means the claim is supported by the data; 0.5 means it claims to be
        half the mass of anything real at those volumes.
        """
        bound = alpha * self.min_mass(v_payload, v_fuel)
        return np.maximum(0.0, (bound - np.asarray(mass)) / np.maximum(bound, 1e-9))
