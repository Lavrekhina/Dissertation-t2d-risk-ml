"""Settings for the T03 cohort build (columns, recodes, derivation constants).

Paths, cycles, HbA1c threshold, minimum age and the feature list come from
``config.py`` (frozen in T02); this file only adds what the cohort build needs.
Choices not in the frozen dictionary are marked ``# DECISION``.
"""
from __future__ import annotations

# Years of data collection per cycle, for pooled survey weights.
CYCLE_YEARS = {"2017-2020": 3.2, "2021-2023": 2.0}

# Component files to merge on SEQN (DEMO is the base table).
FILES = [
    "DEMO", "BMX", "BPXO", "HDL", "TCHOL", "GHB",
    "DR1TOT", "DR2TOT",
    "PAQ", "SLQ", "SMQ", "ALQ", "BPQ", "MCQ",
    "DIQ", "GLU", "INS",
]

# Columns to take from each file. Columns missing in one cycle are allowed
# (e.g. the PAQ questions were redesigned in 2021-2023); a warning is logged.
COLUMNS = {
    "DEMO": ["RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDBORN4",
             "DMDEDUC2", "DMDMARTZ", "INDFMPIR", "RIDEXPRG",
             "SDMVPSU", "SDMVSTRA", "WTMECPRP", "WTMEC2YR"],
    "BMX": ["BMXBMI", "BMXHT", "BMXWAIST", "BMXHIP"],
    "BPXO": ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3",
             "BPXOPLS1", "BPXOPLS2", "BPXOPLS3"],
    "HDL": ["LBDHDD"],
    "TCHOL": ["LBXTC"],
    "GHB": ["LBXGH", "WTPH2YR"],
    "DR1TOT": ["DR1DRSTZ", "DR1_300", "DRDINT", "WTDRD1PP", "WTDRD1",
               "WTDR2DPP", "WTDR2D"],          # + day-1 nutrients, added below
    "DR2TOT": ["DR2DRSTZ", "DR2_300"],          # + day-2 nutrients, added below
    "PAQ": ["PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675",   # 2017-2020
            "PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820",  # 2021-2023
            "PAD680"],
    "SLQ": ["SLD012", "SLD013"],
    "SMQ": ["SMQ020", "SMQ040"],
    "ALQ": ["ALQ111", "ALQ121", "ALQ130"],
    "BPQ": ["BPQ020", "BPQ080"],
    "MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160L"],
    "DIQ": ["DIQ010", "DIQ050", "DIQ070"],
    "GLU": ["LBXGLU"],
    "INS": ["LBXIN"],
}

# Core nutrient set (dictionary rows with keep = derive). Code suffix -> name.
NUTRIENTS = {
    "TKCAL": "energy_kcal",
    "TCARB": "carbohydrate",
    "TSUGR": "total_sugars",
    "TFIBE": "fibre",
    "TTFAT": "total_fat",
    "TSFAT": "sat_fat",
    "TMFAT": "mufa",
    "TPFAT": "pufa",
    "TPROT": "protein",
    "TCHOL": "diet_cholesterol",
    "TSODI": "sodium",
    "TPOTA": "potassium",
    "TCALC": "calcium",
    "TMAGN": "magnesium",
    "TCAFF": "caffeine",
    "TALCO": "alcohol_g",
}
COLUMNS["DR1TOT"] += [f"DR1{c}" for c in NUTRIENTS]
COLUMNS["DR2TOT"] += [f"DR2{c}" for c in NUTRIENTS]

# --------------------------------------------------------------------------
# Derivation constants
# --------------------------------------------------------------------------
HBA1C_THRESHOLD = 6.5        # ADA: diabetes if HbA1c >= 6.5 %
MIN_AGE = 20

# DECISION: vigorous minutes count double (MET-weighted MVPA, as in WHO
# guidance: 1 min vigorous ~ 2 min moderate). Set to 1 for a plain sum.
VIGOROUS_WEIGHT = 2

# DECISION: plausible energy range (kcal/day), Banna et al. 2017 / Willett.
# Used only to FLAG reports in T03; whether to exclude is decided in EDA (T04).
ENERGY_LIMITS = {"female": (500, 3500), "male": (800, 4000)}

# DECISION: approximate drinking days per year for ALQ121 codes
# (same coding in both cycles; check the codebooks before the final run).
ALQ121_DAYS_PER_YEAR = {
    0: 0, 1: 365, 2: 286, 3: 182, 4: 104, 5: 52,
    6: 30, 7: 12, 8: 9, 9: 4.5, 10: 1.5,
}

# Unit of the 2021-2023 activity frequency -> multiplier to "times per week".
PA_UNIT_TO_WEEK = {"D": 7.0, "W": 1.0, "M": 7 / 30.4375, "Y": 7 / 365.25}

LEAKAGE_COLUMNS = ["hba1c", "chk_diagnosed_diabetes", "chk_insulin",
                   "chk_diabetes_pills", "chk_treated_diabetes",
                   "chk_fasting_glucose", "chk_fasting_insulin"]

