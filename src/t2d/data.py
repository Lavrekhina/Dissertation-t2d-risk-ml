"""Load the analytic cohort and the fixed train/test split.

The split is made once (stratified by outcome and cycle, seed from config) and
saved as ``data/processed/split.csv`` so that EDA (T04), preprocessing (T06) and
every model use exactly the same 20 % hold-out test set, which is used once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CATEGORICAL = ["sex", "race_ethnicity", "marital_status", "smoking_status"]
BINARY = ["born_in_us", "told_high_bp", "told_high_cholesterol", "history_cvd",
          "told_liver_condition"]
TEST_SIZE = 0.20


def feature_list() -> list[str]:
    return sum(config.FEATURES.values(), [])


def numeric_features() -> list[str]:
    """Continuous and ordinal features (education is ordinal 1-5)."""
    return [f for f in feature_list() if f not in CATEGORICAL + BINARY]


def group_of(feature: str) -> str:
    for g, cols in config.FEATURES.items():
        if feature in cols:
            return g
    raise KeyError(feature)


def load_cohort(processed: Path = config.PROCESSED) -> pd.DataFrame:
    """Read the T03 output (parquet if present, else CSV)."""
    pq, csv = processed / "analytic_cohort.parquet", processed / "analytic_cohort.csv"
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except ImportError:
            df = pd.read_csv(csv)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Run scripts/03_build_cohort.py first")
    for c in CATEGORICAL:
        df[c] = df[c].astype("string")
    df["diabetes"] = df["diabetes"].astype(int)
    return df


def make_split(df: pd.DataFrame, seed: int = config.SEED,
               test_size: float = TEST_SIZE) -> pd.Series:
    """Stratified (diabetes x cycle) random split. Returns 'train'/'test' per SEQN."""
    rng = np.random.default_rng(seed)
    labels = pd.Series("train", index=df["SEQN"].values, name="split")
    for _, g in df.groupby(["diabetes", "cycle"]):
        ids = g["SEQN"].to_numpy()
        n_test = int(round(len(ids) * test_size))
        labels.loc[rng.choice(ids, n_test, replace=False)] = "test"
    labels.index.name = "SEQN"
    return labels


def load_split(df: pd.DataFrame | None = None, processed: Path = config.PROCESSED,
               create: bool = True) -> pd.Series:
    """Read the saved split; create it the first time. Never re-drawn once saved."""
    path = processed / "split.csv"
    if path.exists():
        s = pd.read_csv(path, index_col="SEQN")["split"]
        if df is not None and set(s.index) != set(df["SEQN"]):
            raise ValueError("split.csv does not match the cohort: rebuild deliberately "
                             "by deleting split.csv (this changes the test set)")
        return s
    if not create or df is None:
        raise FileNotFoundError(path)
    s = make_split(df)
    s.to_csv(path)
    return s


def train_frame(processed: Path = config.PROCESSED) -> pd.DataFrame:
    """The training rows only (EDA and all fitting decisions use these)."""
    df = load_cohort(processed)
    split = load_split(df, processed)
    return df[df["SEQN"].map(split) == "train"].reset_index(drop=True)
