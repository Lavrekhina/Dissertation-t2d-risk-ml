"""Leakage screen (T05), on the TRAINING split only.

1. Single-feature AUC, out-of-fold: each feature alone, scored with its direction
   (and category prevalences) learnt on the training folds. A near-perfect AUC
   means the feature is a proxy for HbA1c/glucose (proposal 8.2).
   Positive controls (HbA1c, fasting glucose/insulin, diagnosis, treatment) are
   screened too: they SHOULD be flagged, which shows the screen works.
2. Whole-model check: a gradient-boosting model on all features, then with each
   positive control added. A plausible AUC for the features (~0.75-0.85 in prior
   NHANES work) and a jump when a control is added show nothing hidden leaks.
3. Manual review: each feature is labelled by how it relates to the outcome.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .data import CATEGORICAL, feature_list, group_of

Y = "diabetes"
FLAG_AUC = 0.90     # near-perfect: treat as leakage, remove
REVIEW_AUC = 0.75   # strong for a single variable: review by hand

POSITIVE_CONTROLS = {
    "hba1c": "HbA1c itself (defines the outcome)",
    "chk_fasting_glucose": "Fasting glucose (fasting subsample)",
    "chk_fasting_insulin": "Fasting insulin (fasting subsample)",
    "chk_diagnosed_diabetes": "Doctor-diagnosed diabetes (DIQ010)",
    "chk_treated_diabetes": "Insulin or diabetes pills (DIQ050/DIQ070)",
}

# Manual review: how each feature could relate to the outcome.
#   ok         - measured independently of diabetes status
#   treatment  - can be changed by diagnosis, advice or medication (reverse causation)
#   contact    - reflects contact with healthcare (diagnosed conditions)
#   design     - survey design / measurement artefact
REVIEW = {
    "age": ("ok", ""), "sex": ("ok", ""), "race_ethnicity": ("ok", ""),
    "education": ("ok", ""), "marital_status": ("ok", ""),
    "income_poverty_ratio": ("ok", ""), "born_in_us": ("ok", ""),
    "bmi": ("treatment", "weight loss advice / GLP-1 drugs after diagnosis"),
    "waist_cm": ("treatment", "as BMI"), "height_cm": ("ok", ""),
    "waist_hip_ratio": ("treatment", "as BMI"),
    "sbp_mean": ("treatment", "antihypertensives are common in diabetes"),
    "dbp_mean": ("treatment", "as SBP"),
    "pulse_mean": ("ok", ""),
    "hdl_mg_dl": ("ok", "lowered by insulin resistance (a cause, not leakage)"),
    "total_chol_mg_dl": ("treatment", "statins after diagnosis (EDA: 167 vs 188 mg/dL)"),
    "told_high_bp": ("contact", "diagnosed with diabetes -> more BP checks"),
    "told_high_cholesterol": ("contact", "as above"),
    "history_cvd": ("contact", "complication and risk factor"),
    "told_liver_condition": ("contact", "fatty liver is linked to diabetes; watch AUC"),
    "smoking_status": ("ok", ""),
    "drinking_freq": ("treatment", "advice to cut alcohol (EDA: lower in diagnosed)"),
    "drinks_per_day": ("treatment", "as above"),
    "leisure_mvpa_min_week": ("treatment", "advice to exercise (EDA: 0 vs 30 min/wk)"),
    "sedentary_min_day": ("ok", ""), "sleep_hours": ("ok", ""),
}
for _f in config.FEATURES["diet"]:
    REVIEW.setdefault(_f, ("treatment", "dietary advice after diagnosis (EDA §5)"))


def _encode_fold(x_tr: pd.Series, y_tr: np.ndarray, x_te: pd.Series, categorical: bool
                 ) -> np.ndarray:
    """Score the test fold with a direction / mapping learnt on the training fold."""
    if categorical:
        prev = pd.Series(y_tr, index=x_tr.index).groupby(x_tr.fillna("__NA__")).mean()
        return x_te.fillna("__NA__").map(prev).fillna(y_tr.mean()).to_numpy(float)
    xt = x_tr.astype(float)
    med = xt.median()
    sign = np.sign(stats.spearmanr(xt.fillna(med), y_tr)[0]) or 1.0
    return sign * x_te.astype(float).fillna(med).to_numpy()


def single_feature_auc(df: pd.DataFrame, features: list[str], n_splits: int = 5,
                       seed: int = config.SEED) -> pd.DataFrame:
    """Out-of-fold AUC of each feature on its own (missing -> training-fold median).

    For variables measured only in a subsample (fasting glucose/insulin) the AUC is
    computed on rows with a value, and n is reported.
    """
    rows = []
    for f in features:
        d = df[[f, Y]]
        subsample = f in ("chk_fasting_glucose", "chk_fasting_insulin")
        if subsample:
            d = d[d[f].notna()]
        y = d[Y].to_numpy()
        oof = np.empty(len(d))
        fold_auc = []
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(d, y):
            s = _encode_fold(d[f].iloc[tr], y[tr], d[f].iloc[te], f in CATEGORICAL)
            oof[te] = s
            fold_auc.append(roc_auc_score(y[te], s))
        auc = roc_auc_score(y, oof)
        rows.append({"feature": f, "auc": auc, "fold_sd": np.std(fold_auc),
                     "n": len(d), "pct_missing": 100 * df[f].isna().mean(),
                     "flag": ("LEAK" if auc >= FLAG_AUC else
                              "review" if auc >= REVIEW_AUC else "ok")})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def _design(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    X = df[feature_list() + (extra or [])].copy()
    for c in CATEGORICAL:   # pd.NA -> np.nan so sklearn's encoder accepts the column
        X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), np.nan))
    return X


def model_check(df: pd.DataFrame, n_splits: int = 5, seed: int = config.SEED) -> pd.DataFrame:
    """5-fold CV AUC of a default gradient-boosting model, alone and with each control."""
    y = df[Y].to_numpy()
    rows = []
    for label, extra in [("all model features", [])] + [
            (f"+ {c}", [c]) for c in POSITIVE_CONTROLS]:
        X = _design(df, extra)
        clf = HistGradientBoostingClassifier(categorical_features="from_dtype",
                                             random_state=seed)
        p = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True,
                                                            random_state=seed),
                              method="predict_proba")[:, 1]
        rows.append({"model": label, "cv_auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


def outcome_vs_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """How the HbA1c outcome lines up with self-reported diagnosis and treatment."""
    d = df.assign(
        diagnosed=df["chk_diagnosed_diabetes"].map({1: "diagnosed", 0: "not diagnosed"})
        .fillna("unknown"),
        treated=df["chk_treated_diabetes"].map({1: "treated", 0: "not treated"})
        .fillna("not asked / unknown"))
    t = d.groupby(["diabetes", "diagnosed", "treated"]).size().rename("n").reset_index()
    t["pct_of_cohort"] = 100 * t["n"] / len(df)
    return t


def review_table(auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = auc.set_index("feature")
    for f in feature_list():
        cat, note = REVIEW.get(f, ("ok", ""))
        rows.append({"feature": f, "group": group_of(f), "single_feature_auc": a.loc[f, "auc"],
                     "auc_flag": a.loc[f, "flag"], "review": cat, "note": note,
                     "decision": "remove" if a.loc[f, "flag"] == "LEAK" else "keep"})
    return pd.DataFrame(rows)


def plot_auc(auc: pd.DataFrame, fig_path: Path) -> None:
    t = auc.sort_values("auc")
    colours = ["#b91c1c" if f in POSITIVE_CONTROLS else
               {"demographic": "#6b7280", "clinical": "#2563eb", "lifestyle": "#059669",
                "diet": "#d97706"}[group_of(f)] for f in t["feature"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(t) + 1.2))
    ax.barh(t["feature"], t["auc"] - 0.5, left=0.5, color=colours,
            xerr=t["fold_sd"], error_kw={"lw": .6})
    for x, ls, lab in [(REVIEW_AUC, ":", f"review ≥ {REVIEW_AUC}"),
                       (FLAG_AUC, "--", f"leak ≥ {FLAG_AUC}")]:
        ax.axvline(x, color="#111827", ls=ls, lw=1)
        ax.text(x, len(t) - 0.2, lab, fontsize=7, ha="center", va="bottom")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Single-feature AUC (5-fold out-of-fold, training set)")
    ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#b91c1c", label="positive control (not a feature)")] +
              [Patch(color=c, label=g) for g, c in
               {"demographic": "#6b7280", "clinical": "#2563eb", "lifestyle": "#059669",
                "diet": "#d97706"}.items()], fontsize=7, loc="lower right")
    ax.set_title("Leakage screen: no model feature comes close to the controls")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
