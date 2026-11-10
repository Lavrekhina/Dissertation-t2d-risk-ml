"""Read NHANES SAS-transport (.XPT) files and merge components on SEQN."""
from pathlib import Path
import pandas as pd

from .config import CYCLES, RAW


def find_file(cycle: str, base: str, raw: Path = RAW) -> Path:
    """Locate a file such as P_GHB.xpt (also accepts the browser-added .xpt.txt)."""
    cfg = CYCLES[cycle]
    stem = cfg["name"](base)
    folder = raw / cfg["folder"]
    for suffix in (".xpt", ".XPT", ".xpt.txt"):
        p = folder / f"{stem}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem} not found in {folder}")


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")


def read_component(cycle: str, base: str, raw: Path = RAW) -> pd.DataFrame:
    return read_xpt(find_file(cycle, base, raw))


def merge_components(cycle: str, bases: list[str], raw: Path = RAW) -> pd.DataFrame:
    """Left-merge all components onto DEMO using the participant ID SEQN."""
    out = read_component(cycle, "DEMO", raw)
    for b in bases:
        if b == "DEMO":
            continue
        out = out.merge(read_component(cycle, b, raw), on="SEQN", how="left")
    out["cycle"] = cycle
    return out
