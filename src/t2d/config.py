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
FEATURE_FILES = ["DEMO", "BMX", "BPXO", "HDL", "TRIGLY", "DR1TOT", "DR2TOT",
                 "PAQ", "SLQ", "SMQ", "ALQ", "BPQ", "MCQ"]
TARGET_FILE = "GHB"                       # HbA1c -> target (>= 6.5 %)
LEAKAGE_FILES = ["DIQ", "RXQ_RX", "GLU", "INS"]  # checks only, never features

HBA1C_THRESHOLD = 6.5
MIN_AGE = 20
