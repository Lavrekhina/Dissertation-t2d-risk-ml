"""Build the T03 analytic dataset.

Run from the repo root:  python scripts/03_build_cohort.py
Reads the NHANES files already in data/raw/<cycle>/ (.xpt or .xpt.txt).

Outputs
    data/processed/analytic_cohort.parquet (or .csv if pyarrow is missing)
    reports/cohort_flow.json  and  reports/figures/cohort_flow.png/.svg
    reports/cohort_columns.csv   (column, group, dtype, % missing)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from t2d import cohort_settings as S, config, flow_diagram  # noqa: E402
from t2d.cohort import build  # noqa: E402


def column_report(df: pd.DataFrame) -> pd.DataFrame:
    group_of = {c: g for g, cols in config.FEATURES.items() for c in cols}
    for c in S.LEAKAGE_COLUMNS:
        group_of[c] = "LEAKAGE CHECK - never a feature"
    group_of["diabetes"] = "target"
    rows = []
    for c in df.columns:
        g = group_of.get(c)
        if g is None:
            g = ("weight/design" if c.startswith(("wt_", "psu", "strata"))
                 else "qc flag" if c.startswith("qc_") else "id/meta")
        rows.append({"column": c, "group": g, "dtype": str(df[c].dtype),
                     "pct_missing": round(100 * df[c].isna().mean(), 1),
                     **{f"pct_missing_{k}": round(100 * df.loc[df.cycle == k, c].isna().mean(), 1)
                        for k in config.CYCLES}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW)
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED)
    ap.add_argument("--reports-dir", type=Path, default=config.ROOT / "reports")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cohort, flow = build(args.raw_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = args.out_dir / "analytic_cohort.parquet"
        cohort.to_parquet(out, index=False)
    except ImportError:
        out = args.out_dir / "analytic_cohort.csv"
        cohort.to_csv(out, index=False)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "cohort_flow.json").write_text(json.dumps(flow, indent=2))
    column_report(cohort).to_csv(args.reports_dir / "cohort_columns.csv", index=False)
    figs = flow_diagram.draw(flow, args.reports_dir / "figures" / "cohort_flow")

    final = flow[-1]
    print(f"\nAnalytic cohort: {final['n']:,} adults "
          + ", ".join(f"{k}: {final[k]:,}" for k in config.CYCLES))
    print(f"Diabetes (HbA1c >= {config.HBA1C_THRESHOLD}%): {final['diabetes_cases']:,} "
          f"({final['prevalence_unweighted']:.1%} unweighted)")
    print(f"Saved {out}\n      {figs[0]}")


if __name__ == "__main__":
    main()
