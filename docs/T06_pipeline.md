# T06 – Leakage-safe preprocessing pipeline v1

Code: `src/t2d/preprocess.py`. Check: `python scripts/06_pipeline_check.py` (about 1 min). Tests: `tests/test_preprocess.py`.

## How to use it (T07 onwards)

```python
from t2d import preprocess
from t2d.data import train_frame

tr = train_frame()
feats = preprocess.feature_sets()["all"]          # or "clinical", "clinical+lifestyle"
X = preprocess.prepare_X(tr, feats)               # rule-based, no fitting
pipe = preprocess.make_pipeline(model, feats, flavour="tree", imputer="median")
# cross_val_score / RandomizedSearchCV(pipe, ...) -> preprocessing refitted in every fold
```

## Steps

| Step | Kind | What | Why (EDA section) |
|---|---|---|---|
| Implausible energy → diet features missing | rule | outside 500–3,500 kcal (women) or 800–4,000 kcal (men) | §6; switch `MASK_IMPLAUSIBLE_ENERGY` |
| Drop `total_fat` | rule | removed from every feature set | §7, VIF 23.5; switch `DROP_FEATURES` |
| `waist_missing` indicator | rule | 1 if waist not measured | §3, missing not at random (22% vs 13%) |
| Winsorise 1st/99th percentile | **fitted** | quantiles learnt per fold | §2, MVPA up to 13,920 min/wk; extreme nutrients |
| log1p (16 skewed features) | fixed | logistic regression only | §2, skew > 1 |
| Impute | **fitted** | median (baseline) or iterative (compared) | §3, proposal 8.3 |
| Standardise | **fitted** | logistic regression only | for the elastic-net penalty |
| Categorical | **fitted** | fill "missing", then one-hot (unknown levels ignored) | sex, race/ethnicity, marital status, smoking |

Rule-based steps use only the row itself, so they can run before the split. Fitted steps live inside the scikit-learn `Pipeline` and are learnt on the training folds only.

## Feature sets (ablation, T12)

| Set | Input features | Columns after encoding |
|---|---|---|
| clinical (demographic + clinical + waist_missing) | 21 | 30 |
| clinical + lifestyle | 27 | 39 |
| all (+ 15 diet features; total_fat dropped) | 42 | 54 |

## Leakage tests (`tests/test_preprocess.py`, all passing)

* Winsorising limits and imputation medians equal those of the **training rows only**.
* Transforming "poisoned" test rows (values set to 10⁹) does not change any fitted parameter, and those values are capped at the **training** 99th percentile.
* No missing values after preprocessing for every set × imputer × flavour combination. An unseen category in a test fold does not break the pipeline.

## Sanity check on the training set (5-fold CV, untuned, not results)

| Feature set | Logistic regression | Gradient boosting (sklearn default) |
|---|---|---|
| clinical | 0.838 | 0.823 |
| clinical + lifestyle | 0.839 | 0.828 |
| all | 0.843 | 0.834 |

The median and iterative imputers give the same AUC to within 0.002. Early signs for the presentation: logistic regression is at least as good as untuned boosting, and lifestyle and diet add about +0.005 AUC. Tuned baselines with confidence intervals come in T07.

**Note:** `IterativeImputer` does not meet scikit-learn's strict stopping rule within 25 rounds on these data, but AUC is the same at 10 and 25 rounds. The convergence warning is silenced in the check script. Mention this in Implementation §4.2.
