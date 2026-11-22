# T03 – Cohort build (NHANES 2017–2020 + 2021–2023)

## Run it

```bash
python -m pytest tests/test_cohort.py   # synthetic-data tests, no real data needed
python scripts/03_build_cohort.py       # reads data/raw/<cycle>/*.xpt(.txt)
```

Code: `src/t2d/cohort.py` (merge, derive, exclusions), `src/t2d/cohort_settings.py`
(columns, recodes, constants), `src/t2d/flow_diagram.py`. Paths, cycles, the HbA1c
threshold and the feature list come from `config.py` (frozen in T02); files are found
with `load.find_file`.

## Outputs

| File | What it is |
|---|---|
| `data/processed/analytic_cohort.parquet` (or `.csv` without pyarrow) | One row per adult; features, target, weights, QC flags, leakage-check columns |
| `reports/figures/cohort_flow.png/.svg` | Exclusion flow diagram (TRIPOD+AI item 20a) for the Methodology chapter and slides |
| `reports/cohort_flow.json` | The same counts as numbers, by cycle |
| `reports/cohort_columns.csv` | Every column with its group and % missing per cycle (starting point for T04 EDA) |

## Exclusion steps (in order)

1. Aged under 20
2. Interviewed but not examined (RIDSTATR ≠ 2)
3. Pregnant at examination (RIDEXPRG = 1)
4. No valid HbA1c (LBXGH missing)
5. No reliable day-1 dietary recall (DR1DRSTZ ≠ 1). This keeps one cohort for all three ablation feature sets, so the change in AUC is not affected by a different sample.

Target: `diabetes = 1` if HbA1c ≥ 6.5 %.

## Column naming

* Feature groups are listed in `config.FEATURES` (demographic, clinical, lifestyle, diet). The "clinical-only" ablation set includes the demographic group.
* `hba1c` and `chk_*` columns (diagnosed diabetes, insulin, pills, fasting glucose and insulin) are kept **only** for the T05 leakage screen and for describing treated cases. They are never features. `check_cohort()` fails the build if one is listed as a feature.
* `qc_*` flags (implausible energy, unusual day-1 intake) are flags only. T04 decides whether to exclude those rows.
* Weights: `wt_mec`, `wt_lab` (2021–2023 phlebotomy weight; MEC weight in 2017–2020), `wt_diet1`, `wt_diet2`, plus `*_pooled` versions scaled by each cycle's share of years.

## Decisions to confirm with the supervisor (marked `# DECISION` in `cohort_settings.py`)

1. ~~Total cholesterol~~: already settled in T02 (TCHOL added in `config.py`).
2. **Pooled weights scaled by years (3.2/5.2 and 2.0/5.2)** instead of "divide by number of cycles" (proposal 8.2), because the 2017–March 2020 file covers 3.2 years. Check against the CDC weighting tutorial before T16.
3. **MVPA counts vigorous minutes double** (`VIGOROUS_WEIGHT = 2`).
4. **ALQ121 frequency mapped to drinking days per year.** Verify the codes against both codebooks.
5. **Energy plausibility limits** 500–3500 kcal (women), 800–4000 kcal (men). Flag only.
6. **Day-1 recall required** for the cohort (exclusion step 5).
7. **Family history of diabetes (MCQ300C) left out.** It is a strong predictor but missing in 2021–2023 (see the dictionary note).

## Known checks for the first real run

* The flow counts should be close to published NHANES numbers (roughly 15,560 participants in P_DEMO and 11,930 in DEMO_L).
* Look at `reports/cohort_columns.csv`: in 2021–2023, missingness for exam and diet variables is higher (~22–39 %) because fewer people were examined. This is expected, and it is mostly removed by exclusion steps 2 and 5.
