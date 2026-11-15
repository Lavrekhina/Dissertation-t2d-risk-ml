"""T02: fill keep / derived_feature / notes for every row of the variable dictionary.

Run after 01_data_audit.py, from the repo root:  python scripts/02_freeze_dictionary.py
Reads reports/variable_dictionary.csv, writes reports/variable_dictionary_frozen.csv.
Every decision and its reason is written here, so the frozen list is reproducible.
"""
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
d = pd.read_csv(REPORTS / "variable_dictionary.csv")

# (keep, derived_feature, notes)
R = {}

def s(vars_, keep, feat="", note=""):
    for v in vars_.split():
        R[v] = (keep, feat, note)

# ---- IDs, design, weights -------------------------------------------------
s("SEQN", "id", "", "Participant ID; merge key")
s("SDMVPSU SDMVSTRA", "design", "", "Survey design variables for the weighted sensitivity analysis")
s("WTMECPRP WTMEC2YR", "weight", "wt_mec",
  "MEC exam weight; renamed between cycles (PRP = 2017-2020, 2YR = 2021-2023). Harmonise to one column in T03")
s("WTINTPRP WTINT2YR", "no", "", "Interview weight; not needed because the cohort is examined adults")
s("WTPH2YR", "weight", "wt_lab",
  "2021-2023 phlebotomy weight; CDC recommends it for lab analytes (HbA1c) in that cycle. 2017-2020 uses WTMECPRP")
s("WTDRD1PP WTDRD1", "weight", "wt_diet1", "Dietary day-1 weight; renamed between cycles. Only needed for weighted diet summaries")
s("WTDR2DPP WTDR2D", "weight", "wt_diet2", "Dietary two-day weight; renamed between cycles")
s("WTSAFPRP WTSAF2YR", "no", "", "Fasting-subsample weight; fasting variables are not used as features")

# ---- Demographic ----------------------------------------------------------
s("RIDAGEYR", "yes", "age", "Also used for the adult (20+) filter")
s("RIAGENDR", "yes", "sex", "")
s("RIDRETH3", "yes", "race_ethnicity", "Chosen over RIDRETH1 because it separates non-Hispanic Asian")
s("RIDRETH1", "no", "", "Duplicate of RIDRETH3 with fewer categories")
s("DMDEDUC2", "yes", "education", "Recode 7/9 (refused / don't know) to missing")
s("DMDMARTZ", "yes", "marital_status", "Recode 77/99 to missing")
s("INDFMPIR", "yes", "income_poverty_ratio", "15-17% missing; impute in-fold")
s("DMDBORN4", "yes", "born_in_us", "Recode 77/99 to missing")
s("RIDSTATR", "filter", "", "Keep RIDSTATR = 2 (interviewed and examined)")
s("RIDEXPRG", "filter", "",
  "Exclude pregnant women (RIDEXPRG = 1): pregnancy changes HbA1c. Missing for men/older women by design")
s("SDDSRVYR", "no", "", "Cycle code; the pipeline adds its own 'cycle' column")
s("RIDEXMON", "no", "", "Season of exam; survey logistics, not a risk factor")
s("RIDAGEMN RIDEXAGM DMDHRAGZ DMDHREDZ DMDHRGND DMDHRMAZ DMDHSEDZ", "no", "", "Child-only or household-reference-person items")
s("DMDHHSIZ DMQMILIZ", "no", "", "Only in 2021-2023")
s("DMDYRUSZ DMDYRUSR", "no", "", "Renamed between cycles and >70% missing (asked only of people born outside the US)")
s("AIALANGA FIAINTRP FIALANG FIAPROXY MIAINTRP MIALANG MIAPROXY SIAINTRP SIALANG SIAPROXY",
  "no", "", "Interview administration; only in 2017-2020")

# ---- Body measures --------------------------------------------------------
s("BMXBMI", "yes", "bmi", "")
s("BMXWAIST", "yes", "waist_cm", "")
s("BMXHT", "yes", "height_cm", "")
s("BMXHIP", "derive", "waist_hip_ratio", "Waist / hip; check VIF against waist and BMI in EDA")
s("BMXWT", "no", "", "Redundant: BMI and height together already encode weight")
s("BMXARMC BMXARML BMXLEG", "no", "", "Limb measures; outside the planned clinical set")
s("BMXHEAD BMXRECUM BMDBMIC", "no", "", "Infant/child measures; 100% missing in adults")
s("BMDSTATS BMIARMC BMIARML BMIHEAD BMIHIP BMIHT BMILEG BMIRECUM BMIWAIST BMIWT", "no", "", "Status/comment codes")

# ---- Blood pressure -------------------------------------------------------
s("BPXOSY1 BPXOSY2 BPXOSY3", "derive", "sbp_mean", "Mean of available systolic readings")
s("BPXODI1 BPXODI2 BPXODI3", "derive", "dbp_mean", "Mean of available diastolic readings")
s("BPXOPLS1 BPXOPLS2 BPXOPLS3", "derive", "pulse_mean", "Mean resting pulse")
s("BPAOARM BPAOCSZ", "no", "", "Measurement details")

# ---- Lipids ---------------------------------------------------------------
s("LBDHDD", "yes", "hdl_mg_dl", "")
s("LBDHDDSI", "no", "", "Same as LBDHDD in mmol/L")
s("LBXTC", "yes", "total_chol_mg_dl", "Full MEC sample (replaces fasting-only TRIGLY/LDL)")
s("LBDTCSI", "no", "", "Same as LBXTC in mmol/L")
s("LBDLDL LBDLDLM LBDLDLN LBDLDLSI LBDLDMSI LBDLDNSI LBDTRSI LBXTR LBXTLG", "no", "",
  "Fasting subsample only (~58-61% of adults missing). Triglyceride also renamed (LBXTR -> LBXTLG). "
  "Replaced by total cholesterol (TCHOL file)")

# ---- Diet: day 1 and day 2 ------------------------------------------------
core = {"KCAL": "energy_kcal", "PROT": "protein", "CARB": "carbohydrate", "SUGR": "total_sugars",
        "FIBE": "fibre", "TFAT": "total_fat", "SFAT": "sat_fat", "MFAT": "mufa", "PFAT": "pufa",
        "CHOL": "diet_cholesterol", "SODI": "sodium", "POTA": "potassium", "MAGN": "magnesium",
        "CALC": "calcium", "CAFF": "caffeine", "ALCO": "alcohol_g"}
for k, f in core.items():
    note = "Absolute energy" if k == "KCAL" else "Per 1,000 kcal"
    R[f"DR1T{k}"] = ("derive", f, f"{note}; day 1, averaged with day 2 where available")
    R[f"DR2T{k}"] = ("derive", f, f"{note}; day 2, averaged with day 1")
for day in "12":
    s(f"DR{day}DRSTZ", "filter", "", f"Day-{day} recall valid only if = 1 (reliable). Day 1 is required for the diet features")
    s(f"DR{day}_300", "filter", "", "Intake vs usual (1 much more, 2 usual, 3 much less): flag for under/over-reporting check")
s("DRDINT", "filter", "", "Number of valid recall days (1 or 2); decides whether to average")
s("DRQSDIET DRQSDT1 DRQSDT2 DRQSDT3 DRQSDT4 DRQSDT5 DRQSDT6 DRQSDT8 DRQSDT9 DRQSDT10 DRQSDT11 DRQSDT12 DRQSDT91",
  "no", "", "Special-diet items: can reflect a diagnosis (leakage risk) and are mostly empty")
s("DRQSDT7", "no", "", "LEAKAGE: 'diabetic diet' reveals known diabetes")

# ---- Physical activity ----------------------------------------------------
s("PAD680", "yes", "sedentary_min_day", "Same question in both cycles; recode 7777/9999 to missing")
s("PAQ650 PAQ655 PAD660 PAQ665 PAQ670 PAD675", "derive", "leisure_mvpa_min_week",
  "2017-2020: days x minutes for vigorous + moderate recreation; 0 if PAQ650/PAQ665 = No. "
  "Count vigorous minutes x2 if using MET-weighting (state choice)")
s("PAD790Q PAD790U PAD800 PAD810Q PAD810U PAD820", "derive", "leisure_mvpa_min_week",
  "2021-2023 redesigned questions: convert frequency + unit to times/week, x minutes; 0 if frequency = 0. "
  "Different wording from 2017-2020: state as a limitation of the temporal check")
s("PAQ605 PAQ610 PAD615 PAQ620 PAQ625 PAD630 PAQ635 PAQ640 PAD645", "no", "",
  "Work and transport activity; not asked in 2021-2023")

# ---- Sleep ----------------------------------------------------------------
s("SLD012 SLD013", "derive", "sleep_hours",
  "Weekly average = (5 x weekday + 2 x weekend) / 7")
s("SLQ300 SLQ310 SLQ320 SLQ330", "no", "", "Clock times; sleep duration already captured")
s("SLQ030 SLQ040 SLQ050 SLQ120", "no", "", "Only in 2017-2020")

# ---- Smoking --------------------------------------------------------------
s("SMQ020 SMQ040", "derive", "smoking_status",
  "never (SMQ020 = 2) / former (SMQ040 = 3) / current (SMQ040 = 1 or 2). SMQ040 'missing' is skip logic, not data loss")
s("SMAQUEX2", "no", "", "Questionnaire mode flag")

# ---- Alcohol --------------------------------------------------------------
s("ALQ111 ALQ121", "derive", "drinking_freq",
  "Past-12-month frequency; 0 if never drank (ALQ111 = 2). Labels reworded only; check codes match in the docs")
s("ALQ130", "derive", "drinks_per_day",
  "Drinks per drinking day; 0 for non-drinkers. Asked in MEC, so it covers examined adults only")
s("ALQ142 ALQ151 ALQ170 ALQ270 ALQ280 ALQ290", "no", "", "Binge-drinking detail; skip logic, high missingness")

# ---- Blood pressure / cholesterol history ---------------------------------
s("BPQ020", "yes", "told_high_bp", "")
s("BPQ080", "yes", "told_high_cholesterol", "")
s("BPD035 BPQ030 BPQ040A BPQ050A BPQ150 BPQ060 BPQ070 BPQ090D BPQ100D BPQ101D", "no", "",
  "Follow-up and medication items; renamed or re-routed between cycles")

# ---- Medical history ------------------------------------------------------
s("MCQ160B MCQ160C MCQ160D MCQ160E MCQ160F", "derive", "history_cvd",
  "Any of heart failure, CHD, angina, heart attack, stroke")
s("MCQ160L", "yes", "told_liver_condition", "Fatty liver is common with diabetes; watch in the leakage screen")
s("MCQ300C", "no", "",
  "Family history of diabetes: strong predictor but not in 2021-2023. Option: keep for the main analysis only "
  "and run the temporal check without it")
s("MCQ366A MCQ366B MCQ366C MCQ366D MCQ371A MCQ371B MCQ371C MCQ371D", "no", "",
  "LEAKAGE risk (doctor advice / behaviour change can follow a diagnosis) and not in 2021-2023")
s("MCQ080", "no", "", "Not in 2021-2023; overlaps with BMI")

# ---- Target ---------------------------------------------------------------
s("LBXGH", "target", "diabetes", "diabetes = LBXGH >= 6.5; never a feature")

# ---- Leakage checks (never features) -------------------------------------
s("DIQ010", "leakage_check", "", "Diagnosed diabetes: compare with the HbA1c outcome; never a feature")
s("DIQ050 DIQ070", "leakage_check", "",
  "Insulin / diabetes pills, both cycles: used to identify treated diabetes (outcome limitation)")
s("DIQ160 DIQ180", "no", "", "LEAKAGE: prediabetes diagnosis / recent blood test track the outcome")
s("RXDDRUG RXDDRGID", "leakage_check", "",
  "Drug names, 2017-2020 only (2021-2023 file has no drug names): optional check for diabetes drugs")
s("LBXGLU LBXIN", "leakage_check", "", "Fasting subsample; used only in the single-feature AUC leakage screen")
s("LBDGLUSI LBDINSI LBDINLC", "no", "", "Unit duplicates / comment code")

# ---- default --------------------------------------------------------------
def decide(row):
    if row.variable in R:
        return R[row.variable]
    if row.file in ("DR1TOT", "DR2TOT"):
        return ("no", "", "Not in the core nutrient set (detail, correlated with totals, or recall metadata)")
    if row.role == "leakage_check":
        return ("no", "", "Diabetes follow-up / medication detail; not needed")
    return ("no", "", "Not needed for the research questions or not usable in both cycles")

d[["keep", "derived_feature", "notes"]] = d.apply(lambda r: pd.Series(decide(r)), axis=1)
cols = list(d.columns)
cols.insert(cols.index("notes"), cols.pop(cols.index("derived_feature")))
d = d[cols]
d.to_csv(REPORTS / "variable_dictionary_frozen.csv", index=False)

# Summary
print(d["keep"].value_counts().to_string())
feats = d[d["keep"].isin(["yes", "derive"])].drop_duplicates("derived_feature")
print("\nFinal features:", len(feats))
for g, sub in feats.groupby("group", sort=False):
    print(f"  {g}: {', '.join(sub['derived_feature'])}")
print(f"\nSaved {REPORTS / 'variable_dictionary_frozen.csv'}")
