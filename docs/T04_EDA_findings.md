# T04 – EDA findings and decisions for the pipeline

Run: `python scripts/04_eda.py` (about 10 s). Tables are written to `reports/eda/` and figures to `reports/figures/eda_*.png`.
Everything below uses the **training split only** (n = 9,484). The test set (n = 2,370) is untouched.

## 0. Train/test split (fixed)

* 80/20 split, stratified by diabetes × cycle, seed 42. It is saved in `data/processed/split.csv` and is **never re-drawn**: `load_split()` refuses to run if the cohort changes.
* Prevalence: 13.6% in train and 13.6% in test. By cycle, 14.4% in 2017–2020 and 12.4% in 2021–2023 in both splits.

## 1. Class balance

| | n | Cases | Unweighted | Weighted (lab weight) |
|---|---|---|---|---|
| All (train) | 9,484 | 1,290 | 13.6% | 10.4% |
| 2017–2020 | 5,830 | 838 | 14.4% | 10.3% |
| 2021–2023 | 3,654 | 452 | 12.4% | 10.6% |

Imbalance is about 6:1, which is moderate. Class weights or SMOTE (T10) are both reasonable. Report PR-AUC alongside ROC-AUC.

## 2. Univariate

* **Skewed (skew > 1, non-negative), so log-transform for logistic regression:** bmi, hdl, energy, protein, fibre, pufa, diet cholesterol, sodium, potassium, magnesium, calcium, caffeine, alcohol, drinking frequency, drinks per day, leisure MVPA. Tree models don't need it.
* **Zero-inflated:** alcohol_g is 0 for 73% of people, leisure MVPA for 39%, and drinking frequency for 27%. Consider adding a binary "any" indicator alongside each.
* **Implausible values:** 28 people report more than 4,200 min/week of MVPA (maximum 13,920), and 4 have BMI above 80. Plan: winsorise at the 1st and 99th percentiles, **fitted on training folds**.
* **Extreme nutrient densities** (for example, sodium up to 62,857 mg per 1,000 kcal) come almost entirely from implausible energy reports (§6).

## 3. Missing data

* Missingness is low. Only income-to-poverty ratio (11.9%) and pulse (9.9%) are above 5%. Diet variables are complete because of the day-1 recall rule.
* **Not completely at random.** Diabetes prevalence is 22% when waist or waist-hip ratio is missing, against 13% when present (χ² p < 0.001), and those people are older (median 64 vs 55). This is probably people who could not stand for the measurement. Missing pulse is concentrated in 2017–2020 (92% of missing values).
* **Decision for T06:** use median imputation as the baseline and compare it with iterative imputation, as in the proposal. Add **missing-indicator** columns for waist and waist-hip ratio. All imputation is fitted inside the folds.

## 4. Bivariate (rank by effect size; nearly every p is below 0.05 at this n)

* Strongest univariate signals, by mutual information: waist-hip ratio, waist, age, told high BP, HDL, BMI, told high cholesterol. These are all clinical or demographic.
* Prevalence rises with age (2.9% at 20–39 to 20.2% at 60–79) and BMI (5.6% below 25 to 22.7% at 35 and over). It is higher in men (15.2% vs 12.2%) and in Mexican American (18.6%) and non-Hispanic Black (16.7%) adults than in non-Hispanic White adults (10.8%).
* **Diet signals are weak.** Mutual information is below 0.005 for every nutrient except alcohol. This is an early hint that diet may add little beyond clinical variables, which would still be a valid answer to RQ1 (risk K5).

## 5. Reverse causation (important for RQ2)

Several lifestyle and diet features point the "wrong" way. Cases report **less** sugar and alcohol and have **lower** total cholesterol than non-cases. Comparing cases who already know they have diabetes with undiagnosed cases explains why:

| Median | Diagnosed cases (n = 998) | Undiagnosed cases (n = 292) | Non-cases |
|---|---|---|---|
| Total sugars, g/1,000 kcal | 43.6 | 47.5 | 47.0 |
| Carbohydrate, g/1,000 kcal | 113.4 | 118.8 | 115.9 |
| Total fat, g/1,000 kcal | 43.3 | 40.1 | 40.9 |
| Total cholesterol, mg/dL | 167 | 188 | 186 |
| Leisure MVPA, min/week | 0 | 30 | 90 |

Undiagnosed cases look like non-cases. Diagnosed cases have changed their diet in the direction of standard advice (lower carbohydrate, higher fat), and their cholesterol is lower, probably from statins. So **some diet and lifestyle features partly reflect having a diagnosis, not causes of diabetes.** SHAP rankings of "modifiable factors" (RQ2) must be read with this in mind.

**Proposed (for the supervisor):** keep the main analysis as planned, and add a sensitivity analysis that **excludes people with diagnosed diabetes**. That analysis predicts undiagnosed diabetes, as in Liu et al. (2025) and Choi et al. (2023). It has about 292 cases in training, so it is smaller but free of this bias.

## 6. Dietary-recall quality

* 87% of people have two valid recalls. 23% said day-1 intake was not usual.
* **4.0% have implausible energy intake** (outside 500–3,500 kcal for women or 800–4,000 kcal for men): 6.2% of men and 2.0% of women. Their diabetes prevalence is higher (17.0% vs 13.5%), and they produce the extreme nutrient densities.
* **Proposed:** set the diet features to missing for these people (imputed in-fold) rather than dropping them. That keeps the cohort, the split and the ablation sample unchanged. Run a sensitivity analysis that excludes them.

## 7. Multicollinearity

* **|ρ| ≥ 0.9:** BMI and waist (0.91). Above 0.7: total fat with MUFA (0.87), saturated fat (0.71) and carbohydrate (−0.70); potassium with magnesium (0.75); fibre with magnesium (0.70); drinking frequency with drinks per day (0.72).
* **VIF > 10:** total fat (23.5) and waist (12.1).
* **Proposed:** drop **total_fat**, because it is close to the sum of saturated, MUFA and PUFA and has the highest VIF. Keep BMI and waist for the tree models. For logistic regression, the elastic-net penalty handles them, and SHAP stability (RQ2) is checked with this pair noted. This keeps the "no |r| > 0.9 pairs among nutrients" rule from proposal 8.2.

## Decisions passed to T06

1. Winsorise at the 1st/99th percentile, fitted in-fold (MVPA, BMI, nutrients).
2. Log1p-transform skewed variables for logistic regression only.
3. Median imputation vs iterative imputation, compared, in-fold, plus missing indicators for waist and waist-hip ratio.
4. Set diet features to missing when energy is implausible (rule-based, so there is no leakage).
5. Drop total_fat.
6. One-hot encode sex, race/ethnicity, marital status and smoking status. Education stays ordinal.

## Figures

`eda_distributions.png`, `eda_missingness.png`, `eda_associations.png`, `eda_prevalence_profiles.png`, `eda_diet_quintiles.png`, `eda_correlation.png`
