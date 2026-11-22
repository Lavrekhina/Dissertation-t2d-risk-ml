"""T04: exploratory data analysis on the training split.

Run from the repo root:  python scripts/04_eda.py
Creates data/processed/split.csv the first time (fixed 80/20 split, seed in config),
then writes tables to reports/eda/ and figures to reports/figures/eda_*.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from t2d import config, eda  # noqa: E402
from t2d.data import load_cohort, load_split  # noqa: E402

OUT = config.ROOT / "reports" / "eda"
FIG = config.ROOT / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    split = load_split(cohort)
    cohort["split"] = cohort["SEQN"].map(split)
    tr = cohort[cohort["split"] == "train"].reset_index(drop=True)

    split_tab = cohort.groupby(["split", "cycle"])["diabetes"].agg(["size", "sum", "mean"])
    split_tab.to_csv(OUT / "split_summary.csv")
    print("Split (rows, cases, prevalence):\n", split_tab, "\n")

    tables = {
        "class_balance": eda.class_balance(tr),
        "univariate": eda.univariate(tr, FIG / "eda_distributions.png"),
        "missingness": eda.missingness(tr, FIG / "eda_missingness.png"),
        "missing_vs_observed": eda.missing_vs_observed(tr),
        "bivariate": eda.bivariate_tests(tr),
        "diet_quintiles": eda.diet_quintiles(tr, FIG / "eda_diet_quintiles.png"),
        "diet_quality": eda.diet_quality(tr),
        "implausible_energy_effect": eda.implausible_energy_effect(tr),
        "table_one": eda.table_one(tr),
        "diagnosed_vs_undiagnosed": eda.diagnosed_vs_undiagnosed(tr),
    }
    eda.plot_associations(tables["bivariate"], FIG / "eda_associations.png")
    prof = eda.prevalence_profiles(tr, FIG / "eda_prevalence_profiles.png")
    tables["prevalence_profiles"] = pd.concat(prof, names=["by", "level"]).reset_index()
    rho, pairs, vif = eda.collinearity(tr, FIG / "eda_correlation.png")
    tables |= {"correlated_pairs": pairs, "vif": vif}
    rho.to_csv(OUT / "spearman_matrix.csv")

    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
    print(f"Saved {len(tables) + 1} tables to {OUT} and 6 figures to {FIG}")


if __name__ == "__main__":
    main()
