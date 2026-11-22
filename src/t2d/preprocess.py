"""Leakage-safe preprocessing pipeline v1 (T06).

Two kinds of step:

1. ``prepare_X`` - RULE-BASED, row by row, nothing learnt from data, so it can
   run before any split without leakage:
     * diet features -> missing when the energy report is implausible (EDA §6)
     * drop total_fat (VIF 23.5; ~ sat + MUFA + PUFA; EDA §7)
     * waist_missing indicator (missing waist is not random: 22% vs 13% prevalence)
2. ``build_preprocessor`` - FITTED steps, inside a scikit-learn Pipeline so they
   are learnt on the training folds only:
     * winsorise at the 1st/99th percentile
     * log1p of skewed non-negative features   (linear models only)
     * impute: median (baseline) or iterative   (compared, proposal 8.3)
     * standardise                               (linear models only)
     * categorical: fill "missing", one-hot encode

Use ``make_pipeline(model, feature_set, flavour)`` so preprocessing and model are
fitted together in every CV fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config
from .data import BINARY, CATEGORICAL

# ---- switches (decisions pending supervisor, 24 Nov) ------------------------
MASK_IMPLAUSIBLE_ENERGY = True   # Tuesday Q6: False = keep reported values
DROP_FEATURES = ["total_fat"]    # EDA §7: [] to keep it
ADD_WAIST_MISSING = True         # EDA §3

WINSOR_QUANTILES = (0.01, 0.99)
ORDINAL = ["education"]
# EDA §2: skew > 1 and non-negative -> log1p for linear models
LOG_FEATURES = ["bmi", "hdl_mg_dl", "energy_kcal", "protein", "fibre", "pufa",
                "diet_cholesterol", "sodium", "potassium", "magnesium", "calcium",
                "caffeine", "alcohol_g", "drinking_freq", "drinks_per_day",
                "leisure_mvpa_min_week"]


# --------------------------------------------------------------------------
# Feature sets for the ablation (proposal 5.2 / T12)
# --------------------------------------------------------------------------
def feature_sets() -> dict[str, list[str]]:
    f = config.FEATURES
    sets = {
        "clinical": f["demographic"] + f["clinical"],
        "clinical+lifestyle": f["demographic"] + f["clinical"] + f["lifestyle"],
        "all": f["demographic"] + f["clinical"] + f["lifestyle"] + f["diet"],
    }
    out = {}
    for name, cols in sets.items():
        cols = [c for c in cols if c not in DROP_FEATURES]
        if ADD_WAIST_MISSING and "waist_cm" in cols:
            cols.append("waist_missing")
        out[name] = cols
    return out


# --------------------------------------------------------------------------
# 1. Rule-based preparation (no fitting)
# --------------------------------------------------------------------------
def prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select ``features`` and apply the row-wise rules. Never looks at y."""
    d = df.copy()
    if MASK_IMPLAUSIBLE_ENERGY and "qc_energy_implausible" in d:
        bad = d["qc_energy_implausible"] == 1
        diet = [c for c in config.FEATURES["diet"] if c in d]
        d.loc[bad, diet] = np.nan
    if ADD_WAIST_MISSING:
        d["waist_missing"] = d["waist_cm"].isna().astype(float)
    X = d[features].copy()
    for c in CATEGORICAL:
        if c in X:   # pd.NA -> np.nan, plain object dtype for sklearn
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


# --------------------------------------------------------------------------
# 2. Fitted steps
# --------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learnt in ``fit`` (NaN ignored and kept)."""

    def __init__(self, quantiles=WINSOR_QUANTILES):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.quantiles[0]).to_numpy()
        self.upper_ = X.quantile(self.quantiles[1]).to_numpy()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        cols = X.columns if isinstance(X, pd.DataFrame) else None
        out = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        return pd.DataFrame(out, columns=cols, index=X.index) if cols is not None else out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None
                          else self.feature_names_in_, dtype=object)


def _imputer(kind: str, seed: int):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "iterative":
        return IterativeImputer(max_iter=25, random_state=seed, sample_posterior=False,
                                initial_strategy="median", skip_complete=True)
    raise ValueError(kind)


def build_preprocessor(features: list[str], flavour: str = "linear",
                       imputer: str = "median", seed: int = config.SEED
                       ) -> ColumnTransformer:
    """ColumnTransformer for ``features``.

    flavour: "linear" (log1p + scaling, for logistic regression) or
             "tree" (no log/scaling, for random forest / boosting).
    imputer: "median" or "iterative".
    """
    if flavour not in ("linear", "tree"):
        raise ValueError(flavour)
    cats = [c for c in features if c in CATEGORICAL]
    binary = [c for c in features if c in BINARY + ["waist_missing"]]
    numeric = [c for c in features if c not in cats + binary]
    logged = [c for c in numeric if flavour == "linear" and c in LOG_FEATURES]
    plain = [c for c in numeric if c not in logged]

    def num_pipe(log: bool) -> Pipeline:
        steps = [("winsor", Winsorizer())]
        if log:
            steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        steps.append(("impute", _imputer(imputer, seed)))
        if flavour == "linear":
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = []
    if imputer == "iterative":
        # one imputer across all numeric columns so they can inform each other
        transformers.append(("num", Pipeline(
            [("winsor", Winsorizer()),
             ("log1p", FunctionTransformer(
                 _log_some, kw_args={"idx": [numeric.index(c) for c in logged]},
                 feature_names_out="one-to-one")),
             ("impute", _imputer("iterative", seed))]
            + ([("scale", StandardScaler())] if flavour == "linear" else [])), numeric))
    else:
        if logged:
            transformers.append(("num_log", num_pipe(True), logged))
        if plain:
            transformers.append(("num", num_pipe(False), plain))
    if binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if cats:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def _log_some(X, idx):
    X = np.array(X, dtype=float, copy=True)
    if idx:
        X[:, idx] = np.log1p(X[:, idx])
    return X


def make_pipeline(model, features: list[str], flavour: str = "linear",
                  imputer: str = "median", seed: int = config.SEED) -> Pipeline:
    """Preprocessing + model, fitted together inside each CV fold."""
    return Pipeline([("prep", build_preprocessor(features, flavour, imputer, seed)),
                     ("model", model)])
