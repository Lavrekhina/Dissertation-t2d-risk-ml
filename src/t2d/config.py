"""Project-wide settings. Change the cycle lists here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SEED = 42  # fixed seed for reproducibility

# NHANES file names per cycle (cycle label -> suffix rule).
# 2017-Mar 2020 pre-pandemic files start with "P_"; 2021-2023 files end with "_L".
CYCLES = {
    "2017-2020": {"folder": "2017-2020", "name": lambda f: f"P_{f}"},
    "2021-2023": {"folder": "2021-2023", "name": lambda f: f"{f}_L"},
}

# Component files (base names) used in the project.
# T02 audit: TRIGLY dropped (fasting subsample only); TCHOL added for total cholesterol.
FEATURE_FILES = ["DEMO", "BMX", "BPXO", "HDL", "TCHOL", "DR1TOT", "DR2TOT",
                 "PAQ", "SLQ", "SMQ", "ALQ", "BPQ", "MCQ"]
TARGET_FILE = "GHB"                              # HbA1c -> target (>= 6.5 %)
LEAKAGE_FILES = ["DIQ", "RXQ_RX", "GLU", "INS"]  # checks only, never features

HBA1C_THRESHOLD = 6.5
MIN_AGE = 20
HIGH_MISSING = 0.50  # audit flags variables missing for >= 50 % of adults

# Feature group of each component file (used by the audit).
FILE_GROUPS = {
    "DEMO": "demographic",
    "BMX": "clinical", "BPXO": "clinical", "HDL": "clinical", "TCHOL": "clinical",
    "BPQ": "clinical", "MCQ": "clinical",
    "PAQ": "lifestyle", "SLQ": "lifestyle", "SMQ": "lifestyle", "ALQ": "lifestyle",
    "DR1TOT": "diet", "DR2TOT": "diet",
    "GHB": "target",
    "DIQ": "leakage", "RXQ_RX": "leakage", "GLU": "leakage", "INS": "leakage",
}

# ---- Frozen feature list (T02, 15 Nov 2026) ---------------------------------
# Full decisions and reasons: reports/variable_dictionary_frozen.csv

# Final model features by group (names after derivation in T03).
FEATURES = {
    "demographic": ["age", "sex", "race_ethnicity", "education", "marital_status",
                    "income_poverty_ratio", "born_in_us"],
    "clinical": ["bmi", "waist_cm", "height_cm", "waist_hip_ratio", "sbp_mean", "dbp_mean",
                 "pulse_mean", "hdl_mg_dl", "total_chol_mg_dl", "told_high_bp",
                 "told_high_cholesterol", "history_cvd", "told_liver_condition"],
    "lifestyle": ["leisure_mvpa_min_week", "sedentary_min_day", "sleep_hours",
                  "smoking_status", "drinking_freq", "drinks_per_day"],
    "diet": ["energy_kcal", "protein", "carbohydrate", "total_sugars", "fibre", "total_fat",
             "sat_fat", "mufa", "pufa", "diet_cholesterol", "sodium", "potassium",
             "magnesium", "calcium", "caffeine", "alcohol_g"],  # per 1,000 kcal except energy
}

# Raw NHANES variables each derived feature comes from.
SOURCE_VARS = {
    "age": ["RIDAGEYR"], "sex": ["RIAGENDR"], "race_ethnicity": ["RIDRETH3"],
    "education": ["DMDEDUC2"], "marital_status": ["DMDMARTZ"],
    "income_poverty_ratio": ["INDFMPIR"], "born_in_us": ["DMDBORN4"],
    "bmi": ["BMXBMI"], "waist_cm": ["BMXWAIST"], "height_cm": ["BMXHT"],
    "waist_hip_ratio": ["BMXWAIST", "BMXHIP"],
    "sbp_mean": ["BPXOSY1", "BPXOSY2", "BPXOSY3"],
    "dbp_mean": ["BPXODI1", "BPXODI2", "BPXODI3"],
    "pulse_mean": ["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"],
    "hdl_mg_dl": ["LBDHDD"], "total_chol_mg_dl": ["LBXTC"],
    "told_high_bp": ["BPQ020"], "told_high_cholesterol": ["BPQ080"],
    "history_cvd": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "told_liver_condition": ["MCQ160L"],
    "leisure_mvpa_min_week": {
        "2017-2020": ["PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675"],
        "2021-2023": ["PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820"]},
    "sedentary_min_day": ["PAD680"], "sleep_hours": ["SLD012", "SLD013"],
    "smoking_status": ["SMQ020", "SMQ040"],
    "drinking_freq": ["ALQ111", "ALQ121"], "drinks_per_day": ["ALQ111", "ALQ130"],
    # diet: DR1T<x> and DR2T<x>, averaged when both recalls are valid
}

FILTER_VARS = ["RIDSTATR", "RIDEXPRG", "DR1DRSTZ", "DR2DRSTZ", "DRDINT", "DR1_300", "DR2_300"]
DESIGN_VARS = ["SDMVPSU", "SDMVSTRA"]
WEIGHTS = {"2017-2020": {"mec": "WTMECPRP", "lab": "WTMECPRP", "diet1": "WTDRD1PP"},
           "2021-2023": {"mec": "WTMEC2YR", "lab": "WTPH2YR", "diet1": "WTDRD1"}}
TARGET_VAR = "LBXGH"
LEAKAGE_VARS = ["DIQ010", "DIQ050", "DIQ070", "RXDDRUG", "LBXGLU", "LBXIN"]  # checks only