# Machine learning for type 2 diabetes risk prediction (NHANES)

BSc Computer Science dissertation project (Sofia Lavrekhina).
Predicts laboratory-defined diabetes (HbA1c >= 6.5 %) from health, lifestyle and dietary
data, compares four models (elastic-net logistic regression, random forest, XGBoost,
LightGBM), runs a feature-group ablation and tests how stable SHAP explanations are.

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
(or `conda env create -f environment.yml`)

## Data
Public NHANES files are **not** stored in this repo. Download the .XPT files from CDC
(https://wwwn.cdc.gov/nchs/nhanes/) and place them in:
- `data/raw/2017-2020/`  (files starting `P_`, e.g. `P_GHB.xpt`)
- `data/raw/2021-2023/`  (files ending `_L`, e.g. `GHB_L.xpt`)

See `src/t2d/config.py` for the list of files used.

## Layout
- `src/t2d/` code (loading, cohort, features, models, explainability)
- `notebooks/` exploratory analysis
- `tests/` unit tests (`pytest`)
- `reports/figures/` generated figures
- `data/` raw and processed data (git-ignored)

## Reproduce
Run the notebooks and scripts in numeric order; all random seeds are set in `config.py`.
