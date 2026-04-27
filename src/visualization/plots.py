"""Figure generation for all paper figures.

Wraps src/plot_utils.py with the DataFrame-based interface used by
the team's experiment runners (experiments/run_*.py).

Figures:
    Fig 1 — R1 CNN double descent        (plot_double_descent)
    Fig 2 — R2 ResNet double descent     (plot_double_descent)
    Fig 3 — N1 test-error heatmap        (plot_error_heatmap)
    Fig 4 — N1 phase diagram (core)      (plot_phase_diagram)
    Fig 5 — N3 weight-decay ablation     (plot_weight_decay_curves)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import shared style + primitives from our plot_utils
import sys, os
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.plot_utils import set_style, PALETTE, _save  # noqa: E402

_PHASE_COLORS = {
    "benign":       PALETTE["benign"],
    "tempered":     PALETTE["tempered"],
    "catastrophic": PALETTE["catastrophic"],
}


# ── Internal save helper ──────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}.png / .pdf")


# ── Fig 1 & 2: Double-Descent Curves ─────────────────────────────────────────

def plot_double_descent(
    data: dict[str, pd.DataFrame],
    x_col: str,
    y_col: str          = "test_error",
    xlabel: str         = "Width multiplier $k$",
    ylabel: str         = "Error rate",
    title: str          = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot one or more double-descent curves on the same axes.

    Args:
        data: {label: DataFrame} — one entry per noise/condition.
              For Fig 1/2: a single entry {"η=15%": df}.
              For Fig 3 overlay: four entries, one per noise level.
        x_col: Column for x-axis (e.g. "k").
        y_col: Metric column (default "test_error").
        save_path: If given, save PNG + PDF (omit extension).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = list(PALETTE.values())[:len(data)]
    for (label, df), color in zip(data.items(), colors):
        df_sorted = df.sort_values(x_col)
        ax.plot(df_sorted[x_col], df_sorted[y_col], "o-", color=color, label=label)

    ax.set_xscale("log", base=2)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(data) > 1:
        ax.legend()

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ── Fig 3: Test-Error Heatmap ─────────────────────────────────────────────────

def plot_error_heatmap(
    phase_table: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of final test error in (k, η) space.

    Args:
        phase_table: DataFrame with columns [k, eta, test_error].
        save_path: If given, save PNG + PDF (omit extension).
    """
    set_style()
    pivot = phase_table.pivot(index="eta", columns="k", values="test_error")
    pivot = pivot.sort_index(ascending=False)  # η high at top

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot * 100,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Test Error (%)"},
    )
    ax.set_xlabel("Width Multiplier $k$")
    ax.set_ylabel("Label Noise Rate $\eta$")
    ax.set_title("Test Error Heatmap — CIFAR-10, CNN")
    ax.set_yticklabels([f"{float(l.get_text()):.0%}" for l in ax.get_yticklabels()])

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ── Fig 4: Phase Diagram ──────────────────────────────────────────────────────

def plot_phase_diagram(
    phase_table: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot partitioning (k, η) into three phases.

    Args:
        phase_table: DataFrame with columns [k, eta, phase].
                     phase ∈ {"benign", "tempered", "catastrophic"}.
        save_path: If given, save PNG + PDF (omit extension).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for phase, grp in phase_table.groupby("phase"):
        ax.scatter(
            grp["k"], grp["eta"],
            c=_PHASE_COLORS.get(phase, "#888888"),
            s=140, zorder=3,
            label=phase.capitalize(),
        )

    ax.set_xscale("log", base=2)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Width Multiplier $k$")
    ax.set_ylabel("Label Noise Rate $\eta$")
    ax.set_title("Overfitting Phase Diagram — CIFAR-10")
    ax.legend(title="Region", loc="upper right")

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ── Fig 5: Weight Decay Curves ────────────────────────────────────────────────

def plot_weight_decay_curves(
    data: dict[float, pd.DataFrame],
    x_col: str          = "k",
    y_col: str          = "test_error",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Double-descent curves for different weight-decay values.

    Args:
        data: {λ: DataFrame} — one entry per weight-decay value.
        save_path: If given, save PNG + PDF (omit extension).
    """
    import matplotlib.cm as cm
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    cmap   = cm.Blues
    wds    = sorted(data.keys())
    colors = [cmap(0.35 + 0.55 * i / max(len(wds) - 1, 1)) for i in range(len(wds))]

    for wd, color in zip(wds, colors):
        df = data[wd].sort_values(x_col)
        label = f"λ={wd:.0e}" if wd > 0 else "λ=0 (no regularisation)"
        ax.plot(df[x_col], df[y_col], "o-", color=color, label=label)

    ax.set_xscale("log", base=2)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Width Multiplier $k$")
    ax.set_ylabel("Test Error")
    ax.set_title("Effect of Weight Decay on Double Descent (η=15%)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ── Convenience alias (backward compat) ──────────────────────────────────────
plot_weight_decay = plot_weight_decay_curves
