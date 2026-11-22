"""Tests for the T03 cohort build, using synthetic NHANES-shaped files.

No real data are needed: a fake reader returns small DataFrames with the same
column names and codes as the CDC files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2d import cohort_settings as S, config  # noqa: E402
from t2d.cohort import apply_exclusions, build, derive  # noqa: E402

N = 300


def _synthetic(cycle: str, comp: str, rng: np.random.Generator) -> pd.DataFrame:
    start = 109263 if cycle == "2017-2020" else 130378
    seqn = np.arange(start, start + N)
    old = cycle == "2017-2020"
    d = {"SEQN": seqn}
    r = lambda *a: rng.choice(a, N)  # noqa: E731
    if comp == "DEMO":
        d |= {"RIDSTATR": r(1, 2, 2, 2, 2), "RIAGENDR": r(1, 2), "RIDAGEYR": rng.integers(0, 81, N),
              "RIDRETH3": r(1, 2, 3, 4, 6, 7), "DMDBORN4": r(1, 2, 77), "DMDEDUC2": r(1, 2, 3, 4, 5, 9),
              "DMDMARTZ": r(1, 2, 3, 77), "INDFMPIR": rng.uniform(0, 5, N),
              "RIDEXPRG": r(np.nan, np.nan, 1, 2), "SDMVPSU": r(1, 2), "SDMVSTRA": r(173, 174, 175),
              ("WTMECPRP" if old else "WTMEC2YR"): rng.uniform(1e3, 5e4, N)}
    elif comp == "BMX":
        d |= {"BMXBMI": rng.normal(29, 6, N), "BMXHT": rng.normal(168, 10, N),
              "BMXWAIST": rng.normal(98, 15, N), "BMXHIP": rng.normal(105, 12, N)}
    elif comp == "BPXO":
        for k in "123":
            d |= {f"BPXOSY{k}": rng.normal(122, 15, N), f"BPXODI{k}": rng.normal(74, 10, N),
                  f"BPXOPLS{k}": rng.normal(72, 10, N)}
    elif comp == "HDL":
        d |= {"LBDHDD": rng.normal(53, 14, N)}
    elif comp == "TCHOL":
        d |= {"LBXTC": rng.normal(190, 40, N)}
    elif comp == "GHB":
        hb = rng.normal(5.7, 0.9, N)
        hb[rng.random(N) < 0.1] = np.nan
        d |= {"LBXGH": hb}
        if not old:
            d["WTPH2YR"] = rng.uniform(1e3, 5e4, N)
    elif comp == "DR1TOT":
        d |= {"DR1DRSTZ": r(1, 1, 1, 2, 5), "DR1_300": r(1, 2, 2, 3), "DRDINT": r(1, 2),
              ("WTDRD1PP" if old else "WTDRD1"): rng.uniform(1e3, 5e4, N),
              ("WTDR2DPP" if old else "WTDR2D"): rng.uniform(1e3, 5e4, N)}
        for c in S.NUTRIENTS:
            d[f"DR1{c}"] = rng.uniform(1, 2500, N)
    elif comp == "DR2TOT":
        d |= {"DR2DRSTZ": r(1, 1, 2), "DR2_300": r(1, 2, 3)}
        for c in S.NUTRIENTS:
            d[f"DR2{c}"] = rng.uniform(1, 2500, N)
    elif comp == "PAQ":
        d["PAD680"] = r(120, 300, 480, 9999)
        if old:
            d |= {"PAQ650": r(1, 2), "PAQ655": r(1, 3, 5), "PAD660": r(30, 60),
                  "PAQ665": r(1, 2), "PAQ670": r(2, 4), "PAD675": r(20, 45)}
        else:
            d |= {"PAD790Q": r(0, 1, 3), "PAD790U": r("D", "W", "M"), "PAD800": r(30, 60),
                  "PAD810Q": r(0, 2), "PAD810U": r("W", "Y"), "PAD820": r(20, 40)}
    elif comp == "SLQ":
        d |= {"SLD012": rng.uniform(4, 10, N), "SLD013": rng.uniform(4, 11, N)}
    elif comp == "SMQ":
        d |= {"SMQ020": r(1, 2, 9), "SMQ040": r(1, 2, 3)}
    elif comp == "ALQ":
        d |= {"ALQ111": r(1, 2), "ALQ121": r(0, 1, 5, 10, 99), "ALQ130": r(1, 2, 4, 999)}
    elif comp == "BPQ":
        d |= {"BPQ020": r(1, 2, 9), "BPQ080": r(1, 2)}
    elif comp == "MCQ":
        for x in "BCDEFL":
            d[f"MCQ160{x}"] = r(1, 2, 2, 2, 9)
    elif comp == "DIQ":
        d |= {"DIQ010": r(1, 2, 3), "DIQ050": r(1, 2), "DIQ070": r(1, 2)}
    elif comp == "GLU":
        d |= {"LBXGLU": rng.normal(105, 25, N)}
    elif comp == "INS":
        d |= {"LBXIN": rng.normal(12, 6, N)}
    return pd.DataFrame(d)


@pytest.fixture
def raw_dir(tmp_path):
    for cyc in config.CYCLES:
        for comp in S.FILES:
            cfg = config.CYCLES[cyc]
            p = tmp_path / cfg["folder"] / f"{cfg['name'](comp)}.xpt.txt"   # browser-style name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
    return tmp_path


@pytest.fixture
def reader():
    rng = np.random.default_rng(42)
    lookup = {}
    for cyc in config.CYCLES:
        for comp in S.FILES:
            lookup[config.CYCLES[cyc]["name"](comp)] = _synthetic(cyc, comp, rng)
    return lambda path: lookup[Path(path).name.split(".")[0]].copy()


def test_build_runs_and_flow_adds_up(raw_dir, reader):
    cohort, flow = build(raw_dir, reader)
    assert flow[0]["n"] == 2 * N
    for prev, step in zip(flow[:-2], flow[1:-1]):
        assert prev["n"] - step["excluded"]["n"] == step["n"]
        for cyc in config.CYCLES:
            assert prev[cyc] - step["excluded"][cyc] == step[cyc]
    assert flow[-1]["n"] == len(cohort) > 0
    assert set(cohort["cycle"]) == set(config.CYCLES)


def test_cohort_criteria(raw_dir, reader):
    cohort, _ = build(raw_dir, reader)
    assert (cohort["age"] >= 20).all()
    assert cohort["hba1c"].notna().all()
    assert ((cohort["hba1c"] >= 6.5) == (cohort["diabetes"] == 1)).all()
    assert not any(c.startswith("_") for c in cohort.columns)


def test_weights_harmonised(raw_dir, reader):
    cohort, _ = build(raw_dir, reader)
    for w in ["wt_mec", "wt_lab", "wt_diet1"]:
        assert cohort[w].notna().all(), w
    old = cohort["cycle"] == "2017-2020"
    assert (cohort.loc[old, "wt_lab"] == cohort.loc[old, "wt_mec"]).all()
    assert cohort["strata"].str.startswith(("2017-2020_", "2021-2023_")).all()


def _one(cycle="2017-2020", **cols):
    return pd.DataFrame({"SEQN": [1], "cycle": [cycle], **{k: [v] for k, v in cols.items()}})


def test_smoking_and_alcohol_codes():
    out = derive(pd.concat([
        _one(SMQ020=2, ALQ111=2, ALQ121=np.nan, ALQ130=np.nan),
        _one(SMQ020=1, SMQ040=3, ALQ111=1, ALQ121=5, ALQ130=2),
        _one(SMQ020=1, SMQ040=1, ALQ111=1, ALQ121=0, ALQ130=np.nan),
        _one(SMQ020=9, ALQ111=1, ALQ121=99, ALQ130=999),
    ], ignore_index=True))
    assert list(out["smoking_status"].fillna("NA")) == ["never", "former", "current", "NA"]
    assert out["drinking_freq"].tolist()[:3] == [0, 52, 0]
    assert np.isnan(out["drinking_freq"].iloc[3])
    assert out["drinks_per_day"].tolist()[:3] == [0, 2, 0]


def test_physical_activity_both_versions():
    old = _one(PAQ650=1, PAQ655=2, PAD660=30, PAQ665=1, PAQ670=3, PAD675=40)
    new = _one("2021-2023", PAD790Q=3, PAD790U="W", PAD800=40,
               PAD810Q=0, PAD810U=None, PAD820=np.nan)
    out = derive(pd.concat([old, new], ignore_index=True))
    vw = S.VIGOROUS_WEIGHT
    assert out["leisure_mvpa_min_week"].tolist() == [3 * 40 + vw * 2 * 30, 3 * 40]


def test_diet_averaging_and_density():
    row = _one(DR1DRSTZ=1, DR2DRSTZ=1, DR1TKCAL=2000, DR2TKCAL=1000,
               DR1TCARB=250, DR2TCARB=125)
    out = derive(row)
    assert out.loc[0, "energy_kcal"] == 1500
    assert out.loc[0, "carbohydrate"] == pytest.approx(125.0)   # 187.5 g / 1500 kcal * 1000
    assert out.loc[0, "diet_recall_days"] == 2
    one_day = derive(_one(DR1DRSTZ=1, DR2DRSTZ=2, DR1TKCAL=2000, DR2TKCAL=500))
    assert one_day.loc[0, "energy_kcal"] == 2000 and one_day.loc[0, "diet_recall_days"] == 1


def test_no_leakage_columns_in_feature_groups():
    feats = set(sum(config.FEATURES.values(), []))
    assert not feats & set(S.LEAKAGE_COLUMNS + ["diabetes", "hba1c"])
