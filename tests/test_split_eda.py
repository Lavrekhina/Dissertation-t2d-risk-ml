"""Tests for the fixed train/test split (data.py) and EDA helpers (eda.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2d import config, eda  # noqa: E402
from t2d.data import feature_list, load_split, make_split  # noqa: E402


def _cohort(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"SEQN": np.arange(n), "cycle": rng.choice(["2017-2020", "2021-2023"], n),
                       "diabetes": (rng.random(n) < 0.14).astype(int)})
    for f in feature_list():
        df[f] = rng.normal(size=n)
    df["sex"] = rng.choice(["male", "female"], n)
    df["race_ethnicity"] = rng.choice(["a", "b", "c"], n)
    df["marital_status"] = rng.choice(["x", "y"], n)
    df["smoking_status"] = rng.choice(["never", "former", "current"], n)
    for f in ["born_in_us", "told_high_bp", "told_high_cholesterol", "history_cvd",
              "told_liver_condition"]:
        df[f] = rng.integers(0, 2, n).astype(float)
    df.loc[rng.random(n) < 0.1, "bmi"] = np.nan
    return df


def test_split_is_reproducible_and_stratified():
    df = _cohort()
    a, b = make_split(df), make_split(df)
    assert a.equals(b)                                   # same seed -> same split
    assert abs((a == "test").mean() - 0.20) < 0.01
    d = df.assign(split=df["SEQN"].map(a))
    prev = d.groupby("split")["diabetes"].mean()
    assert abs(prev["train"] - prev["test"]) < 0.01
    for _, g in d.groupby("cycle"):
        assert abs((g["split"] == "test").mean() - 0.20) < 0.02


def test_saved_split_is_never_redrawn(tmp_path):
    df = _cohort()
    first = load_split(df, processed=tmp_path)
    assert (tmp_path / "split.csv").exists()
    again = load_split(df, processed=tmp_path)
    assert first.sort_index().equals(again.sort_index())
    with pytest.raises(ValueError):                      # cohort changed -> refuse silently reusing
        load_split(df.iloc[:-5], processed=tmp_path)


def test_eda_tables_run_on_synthetic_data():
    df = _cohort()
    assert len(eda.univariate(df)) == len(eda.numeric_features())
    miss = eda.missingness(df)
    assert miss.loc[miss.feature == "bmi", "pct_missing"].iloc[0] == pytest.approx(10, abs=2)
    biv = eda.bivariate_tests(df)
    assert set(biv["feature"]) == set(feature_list())
    _, _, vif = eda.collinearity(df)
    assert (vif["vif"] > 0.9).all()


def test_leakage_screen_flags_a_proxy_but_not_noise():
    from t2d import leakage
    df = _cohort(3000, seed=1)
    rng = np.random.default_rng(2)
    df["proxy"] = df["diabetes"] + rng.normal(0, 0.05, len(df))   # near-perfect proxy
    auc = leakage.single_feature_auc(df, ["proxy", "bmi", "sex"]).set_index("feature")
    assert auc.loc["proxy", "flag"] == "LEAK"
    assert auc.loc["bmi", "flag"] == "ok" and auc.loc["sex", "flag"] == "ok"
    assert 0.4 < auc.loc["bmi", "auc"] < 0.6
