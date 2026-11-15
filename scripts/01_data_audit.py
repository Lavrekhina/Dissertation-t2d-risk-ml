"""T02: NHANES data audit. Run from the repo root:  python scripts/01_data_audit.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2d.audit import run_audit  # noqa: E402
from t2d.config import ROOT  # noqa: E402

out = ROOT / "reports"
dictionary, files = run_audit(out_dir=out)

print("\nFiles found per cycle:")
print(files.pivot(index="file", columns="cycle", values="present").fillna(False))
print(f"\n{len(dictionary)} variables, {dictionary['in_all_cycles'].sum()} in every cycle")
flagged = dictionary[dictionary["flags"] != ""]
print(f"{len(flagged)} variables have flags to review:")
print(flagged[["variable", "file", "label", "flags"]].to_string(index=False))
print(f"\nSaved {out / 'variable_dictionary.csv'} and {out / 'file_availability.csv'}")