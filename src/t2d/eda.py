"""Exploratory data analysis (T04) on the TRAINING split only (proposal 8.2).

Each function returns a table (DataFrame) and, where useful, saves a figure.
Nothing here changes the data: the output is evidence for decisions that the
in-fold pipeline (T06) then applies, fitted on training folds only.

Sections follow proposal 8.2:
  class balance · univariate · missing data · bivariate · multicollinearity ·
  dietary-recall quality · descriptive "Table 1"
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from . import config
from .data import BINARY, CATEGORICAL, feature_list, group_of, numeric_features  # noqa: F401

Y = "diabetes"
GROUP_COLOURS = {"demographic": "#6b7280", "clinical": "#2563eb",
                 "lifestyle": "#059669", "diet": "#d97706"}

# Physiological plausibility limits (adults). Values outside are flagged, not removed.
PLAUSIBLE = {
    "bmi": (12, 80), "waist_cm": (50, 200), "height_cm": (120, 220),
    "waist_hip_ratio": (0.5, 1.6), "sbp_mean": (70, 270), "dbp_mean": (30, 150),
    "pulse_mean": (30, 200), "hdl_mg_dl": (10, 200), "total_chol_mg_dl": (50, 500),
    "sleep_hours": (2, 16), "sedentary_min_day": (0, 1440),
    "leisure_mvpa_min_week": (0, 4200),       # > 10 h/day of MVPA every day
    "energy_kcal": (500, 5000), "drinks_per_day": (0, 25),
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Class balance
# --------------------------------------------------------------------------
def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in [("all", df)] + list(df.groupby("cycle")):
        w = g["wt_lab_pooled"] if name == "all" else g["wt_lab"]
        rows.append({"subset": name, "n": len(g), "cases": int(g[Y].sum()),
                     "prevalence_unweighted": g[Y].mean(),
                     "prevalence_weighted": np.average(g[Y], weights=w.fillna(0)),
                     "imbalance_ratio": (1 - g[Y].mean()) / g[Y].mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Univariate
# --------------------------------------------------------------------------
def univariate(df: pd.DataFrame, fig_path: Path | None = None) -> pd.DataFrame:
    rows = []
    for f in numeric_features():
        x = df[f].dropna()
        q1, q3 = x.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = PLAUSIBLE.get(f, (-np.inf, np.inf))
        skew = stats.skew(x)
        rows.append({
            "feature": f, "group": group_of(f), "n": len(x),
            "pct_missing": 100 * df[f].isna().mean(),
            "median": x.median(), "q1": q1, "q3": q3,
            "p1": x.quantile(.01), "p99": x.quantile(.99), "min": x.min(), "max": x.max(),
            "skew": skew,
            "pct_iqr_outliers": 100 * ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean(),
            "pct_extreme_3iqr": 100 * ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean(),
            "n_implausible": int(((x < lo) | (x > hi)).sum()),
            "pct_zero": 100 * (x == 0).mean(),
            "log_candidate": bool(skew > 1 and x.min() >= 0),
        })
    out = pd.DataFrame(rows)
    if fig_path is not None:
        feats = numeric_features()
        ncol = 6
        nrow = int(np.ceil(len(feats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.3 * nrow))
        for ax, f in zip(axes.flat, feats):
            x = df[f].dropna()
            clip = x.clip(upper=x.quantile(.995))
            ax.hist(clip, bins=40, color=GROUP_COLOURS[group_of(f)], alpha=.85)
            ax.set_title(f, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_yticks([])
        for ax in list(axes.flat)[len(feats):]:
            ax.axis("off")
        fig.suptitle("Distributions of numeric features (training set; clipped at 99.5th pct "
                     "for display). Colour = feature group", fontsize=11)
        fig.tight_layout()
        _save(fig, fig_path)
    return out


# --------------------------------------------------------------------------
# Missing data
# --------------------------------------------------------------------------
def missingness(df: pd.DataFrame, fig_path: Path | None = None) -> pd.DataFrame:
    feats = feature_list()
    tab = pd.DataFrame({"feature": feats, "group": [group_of(f) for f in feats]})
    tab["pct_missing"] = [100 * df[f].isna().mean() for f in feats]
    for cyc, g in df.groupby("cycle"):
        tab[f"pct_missing_{cyc}"] = [100 * g[f].isna().mean() for f in feats]
    tab = tab.sort_values("pct_missing", ascending=False).reset_index(drop=True)
    if fig_path is not None:
        show = tab[tab["pct_missing"] > 0]
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(show) + 1))
        ypos = np.arange(len(show))
        cycles = [c for c in tab.columns if c.startswith("pct_missing_")]
        for i, c in enumerate(cycles):
            ax.barh(ypos + (i - .5) * .38, show[c], height=.38,
                    label=c.replace("pct_missing_", ""), color=["#94a3b8", "#1e3a8a"][i])
        ax.set_yticks(ypos, show["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% missing (training set)")
        ax.legend(title="cycle", fontsize=8)
        ax.set_title("Missing values per feature, by cycle")
        _save(fig, fig_path)
    return tab


def missing_vs_observed(df: pd.DataFrame, min_pct: float = 1.0) -> pd.DataFrame:
    """Compare participants with and without a value (proposal 8.2: is data MAR?).

    For each feature missing in >= min_pct % of rows: diabetes prevalence, median
    age, % female and % from 2021-2023 among missing vs observed, plus a chi-squared
    test of outcome x missingness. Large differences mean the data are not missing
    completely at random, so imputation should use the other features (iterative
    imputer) and/or a missing-indicator.
    """
    rows = []
    for f in feature_list():
        miss = df[f].isna()
        if 100 * miss.mean() < min_pct:
            continue
        ct = pd.crosstab(miss, df[Y])
        p = stats.chi2_contingency(ct)[1] if ct.shape == (2, 2) else np.nan
        rows.append({
            "feature": f, "pct_missing": 100 * miss.mean(),
            "prev_missing": df.loc[miss, Y].mean(), "prev_observed": df.loc[~miss, Y].mean(),
            "age_missing": df.loc[miss, "age"].median(), "age_observed": df.loc[~miss, "age"].median(),
            "pct_female_missing": 100 * (df.loc[miss, "sex"] == "female").mean(),
            "pct_female_observed": 100 * (df.loc[~miss, "sex"] == "female").mean(),
            "pct_2021_missing": 100 * (df.loc[miss, "cycle"] == "2021-2023").mean(),
            "pct_2021_observed": 100 * (df.loc[~miss, "cycle"] == "2021-2023").mean(),
            "chi2_p_outcome": p,
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


# --------------------------------------------------------------------------
# Bivariate
# --------------------------------------------------------------------------
def bivariate_tests(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Association of every feature with the outcome.

    Numeric: medians by class, Mann-Whitney U, rank-biserial r, Spearman rho.
    Binary/categorical: prevalence by level, chi-squared, Cramer's V.
    All: mutual information (median-imputed numeric, label-encoded categorical).
    With ~9,500 rows almost every p-value is tiny, so rank by effect size.
    """
    y = df[Y].to_numpy()
    rows = []
    for f in feature_list():
        x = df[f]
        ok = x.notna()
        r = {"feature": f, "group": group_of(f)}
        if f in CATEGORICAL or f in BINARY:
            ct = pd.crosstab(x[ok], y[ok])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            n = ct.to_numpy().sum()
            r |= {"type": "categorical" if f in CATEGORICAL else "binary", "test": "chi2",
                  "p_value": p, "effect": np.sqrt(chi2 / (n * (min(ct.shape) - 1))),
                  "effect_name": "Cramer's V",
                  "detail": "; ".join(f"{k}: {v:.1%}" for k, v in
                                      df[ok].groupby(f)[Y].mean().items())}
        else:
            a, b = x[ok & (y == 1)], x[ok & (y == 0)]
            u, p = stats.mannwhitneyu(a, b)
            rbc = 2 * u / (len(a) * len(b)) - 1          # rank-biserial correlation
            rho = stats.spearmanr(x[ok], y[ok])[0]
            r |= {"type": "numeric", "test": "Mann-Whitney U", "p_value": p,
                  "effect": abs(rbc), "effect_name": "|rank-biserial r|",
                  "spearman_rho": rho, "median_cases": a.median(),
                  "median_noncases": b.median()}
        rows.append(r)
    out = pd.DataFrame(rows)

    # mutual information on a simple complete matrix (EDA only, not a model input)
    X = pd.DataFrame(index=df.index)
    discrete = []
    for f in feature_list():
        if f in CATEGORICAL:
            X[f] = df[f].astype("category").cat.codes      # NA -> -1 as its own level
            discrete.append(True)
        else:
            X[f] = df[f].fillna(df[f].median())
            discrete.append(f in BINARY)
    mi = mutual_info_classif(X, y, discrete_features=np.array(discrete), random_state=seed)
    out["mutual_info"] = mi
    return out.sort_values("mutual_info", ascending=False).reset_index(drop=True)


def plot_associations(tab: pd.DataFrame, fig_path: Path) -> None:
    t = tab.sort_values("mutual_info")
    fig, ax = plt.subplots(figsize=(7, 0.24 * len(t) + 1))
    ax.barh(t["feature"], t["mutual_info"], color=[GROUP_COLOURS[g] for g in t["group"]])
    ax.set_xlabel("Mutual information with diabetes (nats)")
    ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLOURS.items()],
              fontsize=8, loc="lower right")
    ax.set_title("Univariate association with HbA1c-defined diabetes (training set)")
    _save(fig, fig_path)


def prevalence_profiles(df: pd.DataFrame, fig_path: Path) -> dict[str, pd.DataFrame]:
    """Diabetes prevalence by age band, sex, BMI category and race/ethnicity."""
    d = df.copy()
    d["age_band"] = pd.cut(d["age"], [19, 39, 59, 79, 120],
                           labels=["20-39", "40-59", "60-79", "80+"])
    d["bmi_cat"] = pd.cut(d["bmi"], [0, 25, 30, 35, 200],
                          labels=["<25", "25-29.9", "30-34.9", ">=35"], right=False)
    specs = [("age_band", "Age (years)"), ("sex", "Sex"), ("bmi_cat", "BMI (kg/m²)"),
             ("race_ethnicity", "Race / ethnicity")]
    out = {}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, specs):
        t = d.groupby(col, observed=True)[Y].agg(["mean", "size"]).rename(
            columns={"mean": "prevalence", "size": "n"})
        se = np.sqrt(t["prevalence"] * (1 - t["prevalence"]) / t["n"])
        ax.bar(t.index.astype(str), 100 * t["prevalence"], yerr=196 * se,
               color="#2563eb", alpha=.85, capsize=3)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30 if col == "race_ethnicity" else 0, labelsize=8)
        out[col] = t
    axes[0].set_ylabel("% with HbA1c ≥ 6.5% (95% CI)")
    fig.suptitle("Diabetes prevalence by subgroup (training set, unweighted)")
    fig.tight_layout()
    _save(fig, fig_path)
    return out


def diet_quintiles(df: pd.DataFrame, fig_path: Path) -> pd.DataFrame:
    """Prevalence across quintiles of each energy-adjusted diet feature."""
    rows = []
    for f in config.FEATURES["diet"]:
        x = df[f]
        try:
            q = pd.qcut(x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            continue
        prev = df.groupby(q, observed=True)[Y].mean()
        rows.append({"feature": f, **{f"Q{k}": v for k, v in prev.items()},
                     "Q5_minus_Q1": prev.iloc[-1] - prev.iloc[0]})
    tab = pd.DataFrame(rows).sort_values("Q5_minus_Q1")
    fig, ax = plt.subplots(figsize=(7, 5))
    mat = 100 * tab[[f"Q{k}" for k in range(1, 6)]].to_numpy()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=np.nanmin(mat), vmax=np.nanmax(mat), aspect="auto")
    ax.set_yticks(range(len(tab)), tab["feature"], fontsize=8)
    ax.set_xticks(range(5), [f"Q{k}" for k in range(1, 6)])
    for i in range(mat.shape[0]):
        for j in range(5):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% with diabetes")
    ax.set_title("Diabetes prevalence by quintile of diet feature\n"
                 "(per 1,000 kcal except energy; training set)")
    _save(fig, fig_path)
    return tab


# --------------------------------------------------------------------------
# Multicollinearity
# --------------------------------------------------------------------------
def collinearity(df: pd.DataFrame, fig_path: Path | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Spearman matrix, pairs with |rho| >= 0.7, and VIF on numeric + binary features.

    VIF = diagonal of the inverse Pearson correlation matrix on the median-imputed,
    rank-transformed features (robust to skew; EDA only).
    """
    feats = numeric_features() + BINARY
    X = df[feats]
    rho = X.corr(method="spearman")
    pairs = (rho.where(np.triu(np.ones(rho.shape, bool), 1)).stack()
             .rename("spearman_rho").reset_index()
             .rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
    pairs = pairs[pairs["spearman_rho"].abs() >= 0.7].sort_values(
        "spearman_rho", key=np.abs, ascending=False)
    R = X.fillna(X.median()).rank().corr().to_numpy()
    vif = pd.DataFrame({"feature": feats, "vif": np.diag(np.linalg.inv(R))}
                       ).sort_values("vif", ascending=False)
    if fig_path is not None:
        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7)
        ax.set_yticks(range(len(feats)), feats, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=.7, label="Spearman ρ")
        ax.set_title("Spearman correlation between features (training set)")
        _save(fig, fig_path)
    return rho, pairs, vif


# --------------------------------------------------------------------------
# Dietary-recall quality
# --------------------------------------------------------------------------
def diet_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in [("all", df)] + list(df.groupby("cycle")) + list(df.groupby("sex")):
        rows.append({
            "subset": name, "n": len(g),
            "pct_two_recalls": 100 * (g["diet_recall_days"] == 2).mean(),
            "pct_energy_implausible": 100 * g["qc_energy_implausible"].mean(),
            "pct_day1_not_usual": 100 * g["qc_day1_not_usual"].mean(),
            "median_energy_kcal": g["energy_kcal"].median(),
            "prev_if_implausible": g.loc[g["qc_energy_implausible"] == 1, Y].mean(),
            "prev_if_plausible": g.loc[g["qc_energy_implausible"] == 0, Y].mean(),
        })
    return pd.DataFrame(rows)


def implausible_energy_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Median nutrient density in implausible vs plausible energy reporters."""
    bad = df["qc_energy_implausible"] == 1
    rows = []
    for f in config.FEATURES["diet"]:
        rows.append({"feature": f, "median_implausible": df.loc[bad, f].median(),
                     "median_plausible": df.loc[~bad, f].median(),
                     "p99_plausible": df.loc[~bad, f].quantile(.99),
                     "max_implausible": df.loc[bad, f].max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reverse causation: diagnosed vs undiagnosed cases
# --------------------------------------------------------------------------
def diagnosed_vs_undiagnosed(df: pd.DataFrame) -> pd.DataFrame:
    """Among HbA1c-defined cases, compare people who already know (DIQ010 = 1)
    with undiagnosed people. Lifestyle and diet that differ in the direction of
    advice (less sugar, less alcohol) suggest behaviour changed after diagnosis,
    so these features may reflect diagnosis rather than cause (affects RQ2).
    Non-cases are shown for reference.
    """
    cases = df[df[Y] == 1]
    dx = cases["chk_diagnosed_diabetes"] == 1
    undx = cases["chk_diagnosed_diabetes"] == 0
    rows = []
    for f in config.FEATURES["lifestyle"] + config.FEATURES["diet"] + ["bmi", "total_chol_mg_dl"]:
        if f in CATEGORICAL:
            continue
        a, b = cases.loc[dx, f].dropna(), cases.loc[undx, f].dropna()
        rows.append({"feature": f, "group": group_of(f),
                     "median_diagnosed": a.median(), "median_undiagnosed": b.median(),
                     "median_noncases": df.loc[df[Y] == 0, f].median(),
                     "mann_whitney_p": stats.mannwhitneyu(a, b)[1],
                     "n_diagnosed": len(a), "n_undiagnosed": len(b)})
    return pd.DataFrame(rows).sort_values("mann_whitney_p")


# --------------------------------------------------------------------------
# Descriptive table ("Table 1")
# --------------------------------------------------------------------------
def table_one(df: pd.DataFrame) -> pd.DataFrame:
    """Median (IQR) or n (%) by diabetes status, unweighted."""
    rows = []
    groups = {"Overall": df, "No diabetes": df[df[Y] == 0], "Diabetes": df[df[Y] == 1]}
    rows.append({"characteristic": "n", **{k: f"{len(v):,}" for k, v in groups.items()}})
    for f in feature_list():
        if f in CATEGORICAL:
            for level in sorted(df[f].dropna().unique()):
                rows.append({"characteristic": f"{f}: {level}", **{
                    k: f"{(v[f] == level).sum():,} ({100 * (v[f] == level).mean():.1f}%)"
                    for k, v in groups.items()}})
        elif f in BINARY:
            rows.append({"characteristic": f"{f} (yes)", **{
                k: f"{int(v[f].sum()):,} ({100 * v[f].mean():.1f}%)" for k, v in groups.items()}})
        else:
            rows.append({"characteristic": f, **{
                k: f"{v[f].median():.1f} ({v[f].quantile(.25):.1f}–{v[f].quantile(.75):.1f})"
                for k, v in groups.items()}})
    return pd.DataFrame(rows)
