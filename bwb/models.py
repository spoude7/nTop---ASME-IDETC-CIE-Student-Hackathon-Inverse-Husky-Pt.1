"""The regressor backends the surrogates can be built from.

`MODEL_ZOO` maps a backend name to a constructor. The shipped pipeline uses
gradient boosting for every target; the other entries were used during model
selection and are kept because `bwb/surrogates.py` looks each target's backend
up here by name.

Optional extras (xgboost, lightgbm) register themselves only if installed --
nothing in the pipeline depends on them.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


warnings.filterwarnings("ignore")


def _mlp(hidden=(256, 128), seed=0, iters=400):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                     learning_rate_init=3e-3, max_iter=iters, early_stopping=True,
                     n_iter_no_change=20, random_state=seed),
    )


MODEL_ZOO = {
    "ridge":     lambda s=0: make_pipeline(StandardScaler(), RidgeCV()),
    "poly2":     lambda s=0: make_pipeline(StandardScaler(),
                                           PolynomialFeatures(2, include_bias=False),
                                           RidgeCV()),
    "knn10":     lambda s=0: make_pipeline(StandardScaler(),
                                           KNeighborsRegressor(n_neighbors=10, weights="distance")),
    "rf":        lambda s=0: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                   n_jobs=-1, random_state=s),
    "extratrees": lambda s=0: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                                  n_jobs=-1, random_state=s),
    "histgb":    lambda s=0: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                                           max_leaf_nodes=48, min_samples_leaf=12,
                                                           l2_regularization=1.0, random_state=s,
                                                           early_stopping=False),
    "histgb_deep": lambda s=0: HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.03,
                                                             max_leaf_nodes=128, min_samples_leaf=5,
                                                             l2_regularization=0.5, random_state=s,
                                                             early_stopping=False),
    "mlp":       lambda s=0: _mlp(seed=s),
    "mlp_wide":  lambda s=0: _mlp((512, 256, 128), seed=s, iters=600),
}

try:                                            # used automatically if installed
    import xgboost as xgb
    MODEL_ZOO["xgboost"] = lambda s=0: xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, n_jobs=-1, random_state=s)
except ImportError:
    pass

try:
    import lightgbm as lgb
    MODEL_ZOO["lightgbm"] = lambda s=0: lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.04, num_leaves=64, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=s, verbose=-1)
except ImportError:
    pass


TARGETS = {
    "mass":   ("Aircraft Empty Weight", lambda y: y),
    "stress": ("Max Hotspot Stress", np.log10),
    "vpay":   ("Payload Volume", np.log10),
    "vfuel":  ("Fuel Volume", np.log10),
}
