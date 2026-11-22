"""Merge NHANES components, derive features and build the analytic cohort (T03).

Pipeline
--------
1. ``load_cycle``      read each component .xpt and left-join it onto DEMO by SEQN
2. ``derive``          harmonise renamed variables across cycles and build the
                       derived features listed in the frozen variable dictionary
3. ``apply_exclusions`` apply the cohort criteria in order, recording the
                       number of participants removed at each step (flow diagram)

Nothing here is fitted on the data (no imputation, scaling or outlier removal),
so the output can be split into train/test afterwards without leakage. Those
steps belong in the in-fold pipeline (T06).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import cohort_settings as S
from . import config
from .load import find_file

log = logging.getLogger(__name__)

Reader = Callable[[Path], pd.DataFrame]


# --------------------------------------------------------------------------
# 1. Load and merge
# --------------------------------------------------------------------------
def read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    # SAS transport stores exact zero as 5.4e-79; turn those back into 0.
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0.0)
    return df


def load_cycle(cycle: str, raw_dir: Path = config.RAW,
               reader: Reader = read_xpt) -> pd.DataFrame:
    """Return one row per DEMO participant with all component columns joined."""
    merged = None
    for comp in S.FILES:
        path = find_file(cycle, comp, raw_dir)   # accepts .xpt and .xpt.txt
        df = reader(path)
        if df["SEQN"].duplicated().any():
            raise ValueError(f"{path.name}: SEQN is not unique (expected one row per person)")
        wanted = S.COLUMNS[comp]
        present = [c for c in wanted if c in df.columns]
        missing = sorted(set(wanted) - set(present))
        if missing:
            log.info("%s %s: not in file (expected for renamed/redesigned items): %s",
                     cycle, comp, ", ".join(missing))
        df = df[["SEQN"] + present]
        if merged is None:
            merged = df
        else:
            clash = set(present) & set(merged.columns)
            if clash:
                raise ValueError(f"{comp}: columns already merged from another file: {clash}")
            merged = merged.merge(df, on="SEQN", how="left", validate="one_to_one")
    merged.insert(1, "cycle", cycle)
    return merged


def load_all(raw_dir: Path = config.RAW, reader: Reader = read_xpt) -> pd.DataFrame:
    frames = [load_cycle(c, raw_dir, reader) for c in config.CYCLES]
    out = pd.concat(frames, ignore_index=True, sort=False)
    if out["SEQN"].duplicated().any():
        raise ValueError("SEQN repeats across cycles")
    return out


# --------------------------------------------------------------------------
# 2. Harmonise and derive
# --------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column as float, or all-NaN if absent in this cycle."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _na(s: pd.Series, codes) -> pd.Series:
    """Set refused / don't-know codes to missing."""
    return s.mask(s.isin(codes))


def _yes_no(df: pd.DataFrame, name: str) -> pd.Series:
    """NHANES 1 = yes, 2 = no, 7/9 = refused/don't know -> 1/0/NaN."""
    s = _na(_col(df, name), [7, 9])
    return s.map({1.0: 1.0, 2.0: 0.0})


def _coalesce(df: pd.DataFrame, *names: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for n in names:
        out = out.fillna(_col(df, n))
    return out


def _row_mean(df: pd.DataFrame, names: list[str]) -> pd.Series:
    return pd.concat([_col(df, n) for n in names], axis=1).mean(axis=1, skipna=True)


RACE = {1: "mexican_american", 2: "other_hispanic", 3: "nh_white",
        4: "nh_black", 6: "nh_asian", 7: "other_multi"}
MARITAL = {1: "married_partner", 2: "widowed_divorced_separated", 3: "never_married"}


def _leisure_mvpa(df: pd.DataFrame) -> pd.Series:
    """Leisure-time MVPA, minutes/week, from both questionnaire versions."""
    vw = S.VIGOROUS_WEIGHT
    # 2017-2020: yes/no, days per week, minutes per day
    def old(flag, days, mins):
        f = _yes_no(df, flag)
        d = _na(_col(df, days), [77, 99])
        m = _na(_col(df, mins), [7777, 9999])
        return (d * m).where(f == 1, np.where(f == 0, 0.0, np.nan))
    old_total = old("PAQ665", "PAQ670", "PAD675") + vw * old("PAQ650", "PAQ655", "PAD660")

    # 2021-2023: frequency + unit (D/W/M/Y), minutes per session
    def new(freq, unit, mins):
        q = _na(_col(df, freq), [7777, 9999])
        if unit in df.columns:
            u = df[unit].astype("string").str.strip().str.upper()
            factor = u.map(S.PA_UNIT_TO_WEEK).astype(float)
        else:
            factor = pd.Series(np.nan, index=df.index)
        m = _na(_col(df, mins), [7777, 9999])
        val = q * factor * m
        return val.mask(q == 0, 0.0)
    new_total = new("PAD790Q", "PAD790U", "PAD800") + vw * new("PAD810Q", "PAD810U", "PAD820")

    return old_total.where(df["cycle"] == "2017-2020", new_total)


def _smoking(df: pd.DataFrame) -> pd.Series:
    ever = _yes_no(df, "SMQ020")
    now = _na(_col(df, "SMQ040"), [7, 9])
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out[ever == 0] = "never"
    out[(ever == 1) & (now == 3)] = "former"
    out[(ever == 1) & now.isin([1, 2])] = "current"
    return out


def _diet(df: pd.DataFrame) -> pd.DataFrame:
    """Average valid recall days; express nutrients per 1,000 kcal."""
    d1_ok = _col(df, "DR1DRSTZ") == 1
    d2_ok = (_col(df, "DR2DRSTZ") == 1) & d1_ok
    out = pd.DataFrame(index=df.index)
    out["diet_recall_days"] = np.select([d2_ok, d1_ok], [2, 1], 0)
    for code, name in S.NUTRIENTS.items():
        d1 = _col(df, f"DR1{code}")
        d2 = _col(df, f"DR2{code}")
        avg = pd.Series(np.where(d2_ok, (d1 + d2) / 2, np.where(d1_ok, d1, np.nan)),
                        index=df.index)
        out[name] = avg
    kcal = out["energy_kcal"].where(out["energy_kcal"] > 0)
    for name in S.NUTRIENTS.values():
        if name != "energy_kcal":
            out[name] = out[name] / kcal * 1000
    return out


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the analytic columns from the merged raw table."""
    df = raw
    is_old = df["cycle"] == "2017-2020"
    out = pd.DataFrame({"SEQN": df["SEQN"].astype("int64"), "cycle": df["cycle"]})

    # ---- target ---------------------------------------------------------
    out["hba1c"] = _col(df, "LBXGH")
    out["diabetes"] = (out["hba1c"] >= config.HBA1C_THRESHOLD).astype("Int8").where(
        out["hba1c"].notna())

    # ---- demographic ----------------------------------------------------
    out["age"] = _col(df, "RIDAGEYR")
    out["sex"] = _col(df, "RIAGENDR").map({1.0: "male", 2.0: "female"}).astype("string")
    out["race_ethnicity"] = _col(df, "RIDRETH3").map(RACE).astype("string")
    out["education"] = _na(_col(df, "DMDEDUC2"), [7, 9])          # ordinal 1-5
    out["marital_status"] = _na(_col(df, "DMDMARTZ"), [77, 99]).map(MARITAL).astype("string")
    out["income_poverty_ratio"] = _col(df, "INDFMPIR")
    out["born_in_us"] = _na(_col(df, "DMDBORN4"), [77, 99]).map({1.0: 1.0, 2.0: 0.0})

    # ---- clinical -------------------------------------------------------
    out["bmi"] = _col(df, "BMXBMI")
    out["height_cm"] = _col(df, "BMXHT")
    out["waist_cm"] = _col(df, "BMXWAIST")
    out["waist_hip_ratio"] = out["waist_cm"] / _col(df, "BMXHIP")
    out["sbp_mean"] = _row_mean(df, ["BPXOSY1", "BPXOSY2", "BPXOSY3"])
    out["dbp_mean"] = _row_mean(df, ["BPXODI1", "BPXODI2", "BPXODI3"])
    out["pulse_mean"] = _row_mean(df, ["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"])
    out["hdl_mg_dl"] = _col(df, "LBDHDD")
    out["total_chol_mg_dl"] = _col(df, "LBXTC")
    out["told_high_bp"] = _yes_no(df, "BPQ020")
    out["told_high_cholesterol"] = _yes_no(df, "BPQ080")
    cvd = pd.concat([_yes_no(df, f"MCQ160{x}") for x in "BCDEF"], axis=1)
    out["history_cvd"] = cvd.max(axis=1, skipna=True)   # 1 if any yes; NaN if all missing
    out["told_liver_condition"] = _yes_no(df, "MCQ160L")

    # ---- lifestyle ------------------------------------------------------
    out["smoking_status"] = _smoking(df)
    never_drank = _yes_no(df, "ALQ111") == 0
    freq = _na(_col(df, "ALQ121"), [77, 99])
    out["drinking_freq"] = freq.map(S.ALQ121_DAYS_PER_YEAR).mask(never_drank, 0.0)
    out["drinks_per_day"] = _na(_col(df, "ALQ130"), [777, 999]).mask(
        never_drank | (freq == 0), 0.0)
    out["leisure_mvpa_min_week"] = _leisure_mvpa(df)
    out["sedentary_min_day"] = _na(_col(df, "PAD680"), [7777, 9999])
    out["sleep_hours"] = (5 * _col(df, "SLD012") + 2 * _col(df, "SLD013")) / 7

    # ---- diet -----------------------------------------------------------
    out = out.join(_diet(df))
    lo = out["sex"].map({k: v[0] for k, v in S.ENERGY_LIMITS.items()}).astype(float)
    hi = out["sex"].map({k: v[1] for k, v in S.ENERGY_LIMITS.items()}).astype(float)
    out["qc_energy_implausible"] = ((out["energy_kcal"] < lo) | (out["energy_kcal"] > hi)
                                    ).astype("Int8").where(out["energy_kcal"].notna())
    usual = _na(_col(df, "DR1_300"), [7, 9])
    out["qc_day1_not_usual"] = usual.isin([1, 3]).astype("Int8").where(usual.notna())

    # ---- leakage-check columns (never features) -------------------------
    out["chk_diagnosed_diabetes"] = _na(_col(df, "DIQ010"), [7, 9]).map(
        {1.0: 1.0, 2.0: 0.0, 3.0: 0.0})                      # 3 = borderline -> not diagnosed
    out["chk_insulin"] = _yes_no(df, "DIQ050")
    out["chk_diabetes_pills"] = _yes_no(df, "DIQ070")
    out["chk_treated_diabetes"] = pd.concat(
        [out["chk_insulin"], out["chk_diabetes_pills"]], axis=1).max(axis=1)
    out["chk_fasting_glucose"] = _col(df, "LBXGLU")
    out["chk_fasting_insulin"] = _col(df, "LBXIN")

    # ---- survey design and weights --------------------------------------
    out["psu"] = _col(df, "SDMVPSU")
    # strata codes are made unique per cycle so pooled variance estimates work
    out["strata"] = (df["cycle"].astype(str) + "_"
                     + _col(df, "SDMVSTRA").astype("Int64").astype(str))
    out["wt_mec"] = _coalesce(df, "WTMECPRP", "WTMEC2YR")
    # CDC: lab analytes in 2021-2023 use the phlebotomy weight; 2017-2020 uses MEC
    out["wt_lab"] = _col(df, "WTMECPRP").where(is_old, _col(df, "WTPH2YR"))
    out["wt_diet1"] = _coalesce(df, "WTDRD1PP", "WTDRD1")
    out["wt_diet2"] = _coalesce(df, "WTDR2DPP", "WTDR2D")
    # Pooled weights: scale each cycle by its share of the total years covered
    # (3.2 / 5.2 and 2.0 / 5.2). DECISION: replaces "divide by the number of
    # cycles" in proposal 8.2, because the two cycles cover unequal periods.
    total_years = sum(S.CYCLE_YEARS.values())
    share = df["cycle"].map({k: v / total_years for k, v in S.CYCLE_YEARS.items()})
    for w in ["wt_mec", "wt_lab", "wt_diet1", "wt_diet2"]:
        out[f"{w}_pooled"] = out[w] * share

    # keep the raw filter variables for the exclusion step
    out["_examined"] = _col(df, "RIDSTATR") == 2
    out["_pregnant"] = _col(df, "RIDEXPRG") == 1
    out["_day1_valid"] = _col(df, "DR1DRSTZ") == 1
    return out


# --------------------------------------------------------------------------
# 3. Exclusions
# --------------------------------------------------------------------------
EXCLUSIONS = [
    # (label shown in the flow diagram, function returning rows to KEEP)
    ("Aged under 20", lambda d: d["age"] >= config.MIN_AGE),
    ("Interviewed only, not examined", lambda d: d["_examined"]),
    ("Pregnant at examination", lambda d: ~d["_pregnant"]),
    ("No valid HbA1c result", lambda d: d["hba1c"].notna()),
    ("No reliable day-1 dietary recall", lambda d: d["_day1_valid"]),
]


def _counts(d: pd.DataFrame) -> dict:
    c = {"n": int(len(d))}
    for cyc in config.CYCLES:
        c[cyc] = int((d["cycle"] == cyc).sum())
    return c


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    flow = [{"step": "All NHANES participants", **_counts(df), "excluded": None}]
    cur = df
    for label, keep in EXCLUSIONS:
        mask = keep(cur).fillna(False).astype(bool)
        dropped = cur[~mask]
        cur = cur[mask]
        flow.append({"step": f"Excluded: {label}", "excluded": _counts(dropped),
                     **_counts(cur)})
    final = cur.drop(columns=[c for c in cur.columns if c.startswith("_")]).reset_index(drop=True)
    flow.append({"step": "Analytic cohort", **_counts(final), "excluded": None,
                 "diabetes_cases": int(final["diabetes"].sum()),
                 "prevalence_unweighted": round(float(final["diabetes"].mean()), 4)})
    return final, flow


def check_cohort(df: pd.DataFrame) -> None:
    """Sanity checks that must hold for every build."""
    assert df["SEQN"].is_unique, "duplicate participants"
    assert (df["age"] >= config.MIN_AGE).all()
    assert df["hba1c"].notna().all() and df["diabetes"].notna().all()
    assert set(df["diabetes"].unique()) <= {0, 1}
    features = sum(config.FEATURES.values(), [])
    missing = [f for f in features if f not in df.columns]
    assert not missing, f"feature columns not built: {missing}"
    leaked = set(features) & set(S.LEAKAGE_COLUMNS + ["diabetes"])
    assert not leaked, f"leakage columns listed as features: {leaked}"


def build(raw_dir: Path = config.RAW, reader: Reader = read_xpt
          ) -> tuple[pd.DataFrame, list[dict]]:
    raw = load_all(raw_dir, reader)
    cohort, flow = apply_exclusions(derive(raw))
    check_cohort(cohort)
    return cohort, flow
