"""T06: smoke test of the preprocessing pipeline on the training split.

Run from the repo root:  python scripts/06_pipeline_check.py

Fits preprocessing + a default model inside 5-fold CV for each ablation feature
set and imputer, and checks the output has no missing values. The AUCs are a
SANITY CHECK of the pipeline, not results: untuned models, training folds only.
Tuned baselines are T07. Writes reports/pipeline/pipeline_check.csv.
"""
import sys
import time
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

# IterativeImputer does not meet its strict stopping rule within 25 rounds on these
# data; AUC is identical at 10 and 25 rounds, so the warning is silenced here
# (documented in docs/T06_pipeline.md).
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.impute")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from t2d import config, preprocess  # noqa: E402
from t2d.data import train_frame  # noqa: E402

OUT = config.ROOT / "reports" / "pipeline"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = train_frame()
    y = tr["diabetes"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    models = {
        "logistic (linear prep)": (LogisticRegression(max_iter=5000), "linear"),
        "hist gradient boosting (tree prep)": (
            HistGradientBoostingClassifier(random_state=config.SEED), "tree"),
    }
    rows = []
    for set_name, feats in preprocess.feature_sets().items():
        X = preprocess.prepare_X(tr, feats)
        for imp in ("median", "iterative"):
            for m_name, (model, flavour) in models.items():
                pipe = preprocess.make_pipeline(model, feats, flavour, imp)
                Xt = preprocess.build_preprocessor(feats, flavour, imp).fit_transform(X)
                t0 = time.time()
                auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
                rows.append({"feature_set": set_name, "imputer": imp, "model": m_name,
                             "n_input": len(feats), "n_output_columns": Xt.shape[1],
                             "nan_after_prep": int(Xt.isna().sum().sum()),
                             "cv_auc_mean": auc.mean(), "cv_auc_sd": auc.std(),
                             "seconds": round(time.time() - t0, 1)})
                print(f"{set_name:20s} {imp:9s} {m_name:36s} AUC {auc.mean():.3f} ± {auc.std():.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pipeline_check.csv", index=False)
    assert (res["nan_after_prep"] == 0).all(), "missing values left after preprocessing"
    print(f"\nPipeline OK: no missing values after preprocessing. Saved {OUT / 'pipeline_check.csv'}")


if __name__ == "__main__":
    main()
