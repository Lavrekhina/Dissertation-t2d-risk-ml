"""T05: leakage screen on the training split.

Run from the repo root:  python scripts/05_leakage_screen.py
Writes reports/leakage/*.csv and reports/figures/leakage_single_feature_auc.png.
Exits with an error if any model feature reaches the leakage threshold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2d import config, leakage  # noqa: E402
from t2d.data import feature_list, train_frame  # noqa: E402

OUT = config.ROOT / "reports" / "leakage"
FIG = config.ROOT / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = train_frame()
    feats = feature_list()

    auc = leakage.single_feature_auc(tr, feats + list(leakage.POSITIVE_CONTROLS))
    auc["is_control"] = auc["feature"].isin(leakage.POSITIVE_CONTROLS)
    review = leakage.review_table(auc[~auc["is_control"]])
    model = leakage.model_check(tr)
    crosstab = leakage.outcome_vs_diagnosis(tr)

    auc.to_csv(OUT / "single_feature_auc.csv", index=False)
    review.to_csv(OUT / "feature_review.csv", index=False)
    model.to_csv(OUT / "model_check.csv", index=False)
    crosstab.to_csv(OUT / "outcome_vs_diagnosis.csv", index=False)
    leakage.plot_auc(auc, FIG / "leakage_single_feature_auc.png")

    print(auc.head(12).round(3).to_string(index=False))
    print("\n", model.round(3).to_string(index=False))
    leaks = review[review["decision"] == "remove"]
    if len(leaks):
        sys.exit(f"LEAKAGE: remove {list(leaks['feature'])} before modelling")
    print(f"\nNo model feature reaches AUC {leakage.FLAG_AUC}. "
          f"{(review['auc_flag'] == 'review').sum()} feature(s) to review by hand. "
          f"Saved to {OUT}")


if __name__ == "__main__":
    main()
