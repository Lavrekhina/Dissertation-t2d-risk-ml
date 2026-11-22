"""Tests for the leakage-safe preprocessing pipeline (T06)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from t2d import config, preprocess  # noqa: E402
from test_split_eda import _cohort  # noqa: E402


@pytest.fixture
def df():
    d = _cohort(1500, seed=3)
    rng = np.random.default_rng(4)
    for f in preprocess.LOG_FEATURES:            # skewed, non-negative like the real data
        d[f] = rng.lognormal(2, 1, len(d))
    d["qc_energy_implausible"] = (rng.random(len(d)) < 0.05).astype(int)
    d.loc[rng.random(len(d)) < 0.05, "waist_cm"] = np.nan
    d.loc[rng.random(len(d)) < 0.05, "sex"] = pd.NA
    return d


def test_prepare_x_rules(df):
    feats = preprocess.feature_sets()["all"]
    X = preprocess.prepare_X(df, feats)
    assert "total_fat" not in X and "waist_missing" in X
    bad = df["qc_energy_implausible"] == 1
    assert X.loc[bad, "sodium"].isna().all() and X.loc[~bad, "sodium"].notna().all()
    assert (X["waist_missing"] == df["waist_cm"].isna()).all()


def test_feature_sets_nest():
    s = preprocess.feature_sets()
    assert set(s["clinical"]) < set(s["clinical+lifestyle"]) < set(s["all"])
    assert not set(s["all"]) & {"hba1c", "diabetes", "chk_diagnosed_diabetes"}


@pytest.mark.parametrize("flavour", ["linear", "tree"])
@pytest.mark.parametrize("imputer", ["median", "iterative"])
def test_output_complete(df, flavour, imputer):
    feats = preprocess.feature_sets()["all"]
    X = preprocess.prepare_X(df, feats)
    Xt = preprocess.build_preprocessor(feats, flavour, imputer).fit_transform(X)
    assert Xt.isna().sum().sum() == 0
    assert len(Xt) == len(X)
    if flavour == "linear":   # scaled
        num = [c for c in Xt if c in ("age", "bmi")]
        assert np.allclose(Xt[num].mean(), 0, atol=1e-6)


def test_fitted_on_training_rows_only(df):
    """Parameters must depend on the rows passed to fit, never on the test rows."""
    feats = preprocess.feature_sets()["clinical"]
    X = preprocess.prepare_X(df, feats)
    train, test = X.iloc[:1000], X.iloc[1000:].copy()
    pre = preprocess.build_preprocessor(feats, "linear", "median").fit(train)
    num = pre.named_transformers_["num"]
    win, imp = num.named_steps["winsor"], num.named_steps["impute"]
    cols = list(num.feature_names_in_)
    assert np.allclose(win.lower_, train[cols].quantile(0.01).to_numpy())
    assert np.allclose(imp.statistics_, train[cols].clip(win.lower_, win.upper_,
                                                         axis=1).median().to_numpy())
    # poison the test rows: transforming them must not change fitted parameters
    before = imp.statistics_.copy()
    test[cols] = 1e9
    pre.transform(test)
    assert np.array_equal(before, imp.statistics_)
    # and extreme test values are capped at the TRAINING 99th percentile
    capped = win.transform(test[cols])
    assert np.allclose(capped.max().to_numpy(), win.upper_)
    assert np.isfinite(pre.transform(test).to_numpy()).all()


def test_cross_validation_runs(df):
    feats = preprocess.feature_sets()["all"]
    X = preprocess.prepare_X(df, feats)
    pipe = preprocess.make_pipeline(LogisticRegression(max_iter=2000), feats, "linear")
    auc = cross_val_score(pipe, X, df["diabetes"],
                          cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED),
                          scoring="roc_auc")
    assert len(auc) == 3 and np.isfinite(auc).all()


def test_unseen_category_in_test_fold(df):
    feats = preprocess.feature_sets()["clinical"]
    X = preprocess.prepare_X(df, feats)
    pre = preprocess.build_preprocessor(feats, "tree").fit(X)
    new = X.iloc[:5].copy()
    new["race_ethnicity"] = "never_seen"
    assert pre.transform(new).isna().sum().sum() == 0
