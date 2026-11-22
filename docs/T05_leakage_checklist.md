# T05 – Leakage screen and checklist

Run: `python scripts/05_leakage_screen.py` (about 15 s, training split only). It writes `reports/leakage/*.csv` and `reports/figures/leakage_single_feature_auc.png`, and **exits with an error if any model feature reaches the leakage threshold**, so it can be re-run before the final experiments (risk K2).

## Result: no leakage found

| Check | Result | Verdict |
|---|---|---|
| Highest single-feature AUC among the 42 model features | waist-hip ratio **0.71** (then waist 0.69, age 0.67, told high BP 0.66, HDL 0.66, BMI 0.66) | Below the review (0.75) and leak (0.90) thresholds |
| Positive controls (screened to prove the method works) | HbA1c 1.00 · fasting glucose 0.94 · diagnosed 0.86 · treated 0.85 · fasting insulin 0.68 | Controls flagged as expected |
| Default gradient-boosting model, all features, 5-fold CV | AUC **0.84** | In line with prior NHANES work (0.80–0.83, Table 1 of the proposal) |
| Same model + one control | + HbA1c 1.00 · + glucose 0.91 · + diagnosed 0.93 · + treated 0.93 | A clear jump, so no hidden proxy among the features |

AUCs are out-of-fold: each feature's direction (or category prevalence) is learnt on 4 folds and scored on the 5th.

## Checklist (proposal Table 4, "Leakage-safe pipeline")

- [x] HbA1c, fasting glucose, fasting insulin and diabetes medication are excluded from features. `check_cohort()` fails the build if a `chk_*` column or `hba1c` is listed as a feature.
- [x] No single feature has a near-perfect AUC (maximum 0.71).
- [x] Derived diabetes variables are excluded: DIQ items, "diabetic diet" (DRQSDT7), prediabetes diagnosis (DIQ160), recent blood test (DIQ180), doctor's advice items (MCQ366/371). All were dropped in the frozen dictionary (T02).
- [x] Train/test split fixed before EDA; EDA and this screen use training rows only.
- [ ] All fitting inside CV folds: imputation, scaling, winsorising and encoding. Due in T06.
- [ ] Repeat this screen before the final run (T17).
- [ ] Feature list reviewed with supervisor (24 Nov).

## Manual review of features (`reports/leakage/feature_review.csv`)

None are removed. Three kinds are kept but noted, because they can be **consequences** of diabetes rather than causes. This is reverse causation, not leakage: the information is available before the HbA1c test, but it can be shaped by an earlier diagnosis.

| Kind | Features | Why |
|---|---|---|
| Treatment / advice | BMI, waist, waist-hip ratio, SBP, DBP, total cholesterol, alcohol, MVPA, all diet features | Changed by medication (statins, antihypertensives, GLP-1) or by lifestyle advice after diagnosis. EDA §5 shows diagnosed cases differ from undiagnosed cases in this direction. |
| Healthcare contact | told high BP, told high cholesterol, history of CVD, told liver condition | People with diabetes see doctors more, so they are more often told about other conditions. |
| Independent | age, sex, race/ethnicity, education, marital status, income, born in US, height, pulse, HDL, smoking, sedentary time, sleep | No plausible route from diabetes status. |

This supports the proposed sensitivity analysis that **excludes people with diagnosed diabetes** (Tuesday question 5).

## Outcome vs self-report (training set, `outcome_vs_diagnosis.csv`)

| | Diagnosed | Not diagnosed |
|---|---|---|
| HbA1c ≥ 6.5% (cases) | 998 (10.5%) | 292 (3.1%), undiagnosed diabetes |
| HbA1c < 6.5% (non-cases) | 420 (4.4%), of whom 299 treated | 7,773 (82.0%) |

The 420 diagnosed people below 6.5% are mostly **treated, well-controlled diabetes classed as negative**. This is the outcome limitation already stated in proposal §8.1. Report it in the Methodology and Discussion chapters, using these numbers.
