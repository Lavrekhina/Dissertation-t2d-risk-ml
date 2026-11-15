"""NHANES data audit (task T02): which variables exist in which cycle, and how complete they are.

For every component file in config.py and every cycle, the audit records each variable's
SAS label, type, and the share of adults (age >= MIN_AGE) with no value. It then builds
one row per variable with a flag column that points at anything to decide before the
variable list is frozen: a variable present in only one cycle, a type or label that
changed between cycles, or a high share of missing values.

Note: "missing" means an empty value in the file. NHANES codes such as 7/9/77/99
(refused / don't know) are NOT counted here; they are recoded during cleaning (T03).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (CYCLES, FEATURE_FILES, FILE_GROUPS, HIGH_MISSING, LEAKAGE_FILES,
                     MIN_AGE, RAW, TARGET_FILE)
from .load import find_file

AUDIT_FILES = FEATURE_FILES + [TARGET_FILE] + LEAKAGE_FILES
DESIGN_VARS = {"SDMVPSU", "SDMVSTRA"}


def read_with_labels(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read an .XPT file and return the data and {variable: SAS label}."""
    with pd.read_sas(path, format="xport", iterator=True, encoding="latin-1") as reader:
        labels = {}
        for f in getattr(reader, "fields", []):
            name, label = f.get("name", b""), f.get("label", b"")
            if isinstance(name, bytes):
                name = name.decode("latin-1")
            if isinstance(label, bytes):
                label = label.decode("latin-1")
            labels[name.strip()] = label.strip()
        df = reader.read()
    return df, labels


def variable_role(variable: str, base: str) -> str:
    if variable == "SEQN":
        return "id"
    if variable.startswith("WT"):
        return "weight"
    if variable in DESIGN_VARS:
        return "design"
    if base == TARGET_FILE:
        return "target"
    if base in LEAKAGE_FILES:
        return "leakage_check"
    return "feature_candidate"


def _not_missing(df: pd.DataFrame) -> pd.DataFrame:
    """True where a value is present (blank strings count as missing)."""
    present = df.notna()
    for col in df.columns[df.dtypes == object]:
        present[col] &= df[col].astype(str).str.strip() != ""
    return present


def audit_cycle(cycle: str, raw: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every file of one cycle. Returns (variable rows, file rows)."""
    demo, _ = read_with_labels(find_file(cycle, "DEMO", raw))
    adults = demo.loc[demo["RIDAGEYR"] >= MIN_AGE, "SEQN"]
    n_adults = len(adults)

    var_rows, file_rows = [], []
    for base in AUDIT_FILES:
        try:
            path = find_file(cycle, base, raw)
        except FileNotFoundError:
            file_rows.append({"cycle": cycle, "file": base, "present": False})
            continue
        df, labels = read_with_labels(path)
        # Some files (e.g. RXQ_RX) have several rows per person: a person counts as
        # "has a value" if any of their rows has one.
        present = _not_missing(df.drop(columns="SEQN")).groupby(df["SEQN"]).any()
        present = present.reindex(adults, fill_value=False)
        file_rows.append({
            "cycle": cycle, "file": base, "present": True, "file_name": path.name,
            "n_rows": len(df), "n_participants": df["SEQN"].nunique(),
            "n_adults_in_file": int(adults.isin(df["SEQN"]).sum()), "n_adults": n_adults,
        })
        for var in df.columns:
            if var == "SEQN" and base != "DEMO":
                continue  # the ID is listed once, under DEMO
            n_ok = n_adults if var == "SEQN" else int(present[var].sum())
            var_rows.append({
                "cycle": cycle, "file": base, "variable": var,
                "label": labels.get(var, ""),
                "type": "numeric" if pd.api.types.is_numeric_dtype(df[var]) else "text",
                "n_adults_with_value": n_ok,
                "pct_missing_adults": round(100 * (1 - n_ok / n_adults), 1) if n_adults else None,
            })
    return pd.DataFrame(var_rows), pd.DataFrame(file_rows)


def build_dictionary(variables: pd.DataFrame, cycles: list[str]) -> pd.DataFrame:
    """One row per variable, with per-cycle availability, missingness and flags."""
    rows = []
    for (base, var), g in variables.groupby(["file", "variable"], sort=False):
        by_cycle = g.set_index("cycle")
        labels = [lab for lab in by_cycle["label"] if lab]
        row = {
            "variable": var, "label": labels[0] if labels else "", "file": base,
            "group": FILE_GROUPS.get(base, ""), "role": variable_role(var, base),
        }
        flags = []
        for c in cycles:
            here = c in by_cycle.index
            row[f"in_{c}"] = here
            row[f"pct_missing_{c}"] = by_cycle.at[c, "pct_missing_adults"] if here else None
            if not here:
                flags.append(f"not in {c}")
            elif row[f"pct_missing_{c}"] is not None and \
                    row[f"pct_missing_{c}"] >= 100 * HIGH_MISSING:
                flags.append(f">= {int(100 * HIGH_MISSING)}% missing in {c}")
        if by_cycle["type"].nunique() > 1:
            flags.append("type differs between cycles")
        if len(set(labels)) > 1:
            flags.append("label differs between cycles: " + " | ".join(dict.fromkeys(labels)))
        row["in_all_cycles"] = all(row[f"in_{c}"] for c in cycles)
        row["flags"] = "; ".join(flags)
        row["keep"] = ""      # fill in by hand: yes / no
        row["notes"] = ""     # reason, recoding needed, etc.
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {f: i for i, f in enumerate(AUDIT_FILES)}
    return out.sort_values(["file", "variable"], key=lambda s: s.map(order) if s.name == "file"
                           else s).reset_index(drop=True)


def run_audit(raw: Path = RAW, out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = list(CYCLES)
    parts = [audit_cycle(c, raw) for c in cycles]
    variables = pd.concat([p[0] for p in parts], ignore_index=True)
    files = pd.concat([p[1] for p in parts], ignore_index=True)
    dictionary = build_dictionary(variables, cycles)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dictionary.to_csv(out_dir / "variable_dictionary.csv", index=False)
        files.to_csv(out_dir / "file_availability.csv", index=False)
    return dictionary, files