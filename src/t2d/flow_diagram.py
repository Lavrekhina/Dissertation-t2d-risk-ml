"""Participant flow diagram (TRIPOD+AI item 20a) from the exclusion counts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from . import config


def _fmt(c: dict) -> str:
    cycles = "  |  ".join(f"{k}: {c[k]:,}" for k in config.CYCLES)
    return f"n = {c['n']:,}\n{cycles}"


def draw(flow: list[dict], out_stem: Path) -> list[Path]:
    """Save ``<out_stem>.png`` and ``.svg``. Returns the paths."""
    steps = [flow[0]] + [s for s in flow[1:] if s["step"].startswith("Excluded")]
    n_rows = len(steps)
    fig, ax = plt.subplots(figsize=(10, 1.55 * n_rows))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_rows * 1.55)
    ax.axis("off")

    def box(x, y, w, h, text, fc, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec="#4a4a4a", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
                fontweight="bold" if bold else "normal", wrap=True)

    bw, bh, x_main = 4.6, 0.95, 0.3
    y = n_rows * 1.55 - 1.2
    box(x_main, y, bw, bh, f"{flow[0]['step']}\n{_fmt(flow[0])}", "#e8eef7", bold=True)
    for s in steps[1:]:
        y_next = y - 1.55
        ax.annotate("", xy=(x_main + bw / 2, y_next + bh), xytext=(x_main + bw / 2, y),
                    arrowprops=dict(arrowstyle="-|>", color="#4a4a4a"))
        ax.annotate("", xy=(5.25, y - 0.3), xytext=(x_main + bw / 2, y - 0.3),
                    arrowprops=dict(arrowstyle="-|>", color="#4a4a4a"))
        label = s["step"].replace("Excluded: ", "")
        box(5.25, y - 0.75, 4.45, 0.9, f"Excluded: {label}\n{_fmt(s['excluded'])}", "#fbe9e7")
        is_last = s is steps[-1]
        title = "Analytic cohort" if is_last else "Remaining"
        extra = ""
        if is_last:
            final = flow[-1]
            extra = (f"\nHbA1c ≥ {config.HBA1C_THRESHOLD}%: {final['diabetes_cases']:,} "
                     f"({final['prevalence_unweighted']:.1%}, unweighted)")
        grow = 0.3 if is_last else 0.0   # extra line for the outcome count
        box(x_main, y_next - grow, bw, bh + grow,
            f"{title}\n{_fmt(s)}{extra}", "#e6f4ea" if is_last else "#f5f5f5", bold=is_last)
        y = y_next
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
