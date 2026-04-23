"""Figure generation for all five paper figures.

Style conventions (apply to every plot):
    - Font: use matplotlib rcParams; set once in _apply_style()
    - Colors: seaborn colorblind palette
    - Save as both PNG (300 dpi) and PDF for paper inclusion
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Shared style ────────────────────────────────────────────────────────────

_PHASE_COLORS = {
    "benign": "#2ecc71",       # green
    "tempered": "#f39c12",     # orange
    "catastrophic": "#e74c3c", # red
}


def _apply_style() -> None:
    """Set global matplotlib style for all figures.

    TODO:
        Choose font sizes, line widths, and spine style consistent with the
        paper's LaTeX template (likely ACM or NeurIPS style).
        Example starting point:
            plt.rcParams.update({
                "font.size": 11,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.5,
            })
    """
    sns.set_theme(style="whitegrid", palette="colorblind")
    # TODO: adjust rcParams to match paper typography


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Figure 1 & 3: Double-Descent Curves ─────────────────────────────────────

def plot_double_descent(
    data: dict[str, pd.DataFrame],
    x_col: str,
    y_col: str = "test_error",
    xlabel: str = "Width multiplier k",
    ylabel: str = "Test error",
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot one or more double-descent curves on the same axes.

    Args:
        data: Mapping of legend label → DataFrame with columns [x_col, y_col].
              For Fig 1: one entry (e.g. {"η=15%": df}).
              For Fig 3: four entries, one per noise level.
        x_col: Column to use as x-axis (e.g. "k" or "width_mult").
        y_col: Metric column (default "test_error").
        save_path: If given, save PNG + PDF here (omit extension).

    Returns:
        matplotlib Figure.

    TODO:
        - Plot each series with ax.plot(df[x_col], df[y_col], label=label).
        - Use log scale for x-axis (ax.set_xscale("log")).
        - Mark the interpolation threshold (where train error first hits 0)
          with a vertical dashed line.
        - Add legend, axis labels, and optional title.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # TODO: implement plot body

    if save_path:
        _save(fig, save_path)
    return fig


# ── Figure 2: Test-Error Heatmap ─────────────────────────────────────────────

def plot_error_heatmap(
    phase_table: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of final test error in (k, η) space (Fig 2).

    Args:
        phase_table: DataFrame with columns [k, eta, test_error].
        save_path: If given, save PNG + PDF here (omit extension).

    Returns:
        matplotlib Figure.

    TODO:
        - Pivot to a matrix: rows=eta (sorted descending), cols=k (sorted ascending).
        - Use sns.heatmap with annot=True (show error values) and a diverging
          colormap ("RdYlGn_r" works well — red=high error, green=low).
        - Label axes clearly; use log-scale tick labels for k if possible.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    # TODO: implement heatmap body

    if save_path:
        _save(fig, save_path)
    return fig


# ── Figure 4: Phase Diagram ───────────────────────────────────────────────────

def plot_phase_diagram(
    phase_table: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter/region plot partitioning (k, η) into three phases (Fig 4, core).

    Args:
        phase_table: DataFrame with columns [k, eta, phase].
                     phase ∈ {"benign", "tempered", "catastrophic"}.
        save_path: If given, save PNG + PDF here (omit extension).

    Returns:
        matplotlib Figure.

    TODO:
        - Map each phase to its color in _PHASE_COLORS.
        - ax.scatter(k_vals, eta_vals, c=colors, s=120, zorder=3).
        - Optionally draw a hand-fitted or interpolated boundary between phases.
        - Use log x-axis; add a legend patch for each phase.
        - Title: "Empirical (k, η) Phase Diagram — CIFAR-10, ResNet-18".
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # TODO: implement phase diagram body

    if save_path:
        _save(fig, save_path)
    return fig


# ── Figure 5: Weight Decay Effect ────────────────────────────────────────────

def plot_weight_decay(
    data: dict[float, pd.DataFrame],
    x_col: str = "k",
    y_col: str = "test_error",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Double-descent curves for different weight decay values (Fig 5).

    Args:
        data: Mapping of λ (float) → DataFrame with columns [x_col, y_col].
        save_path: If given, save PNG + PDF here (omit extension).

    Returns:
        matplotlib Figure.

    TODO:
        - Same structure as plot_double_descent but label each curve by λ value.
        - Use a sequential colormap (e.g. Blues) so increasing λ maps to darker blue.
        - Key finding to highlight: the DD peak should shrink / disappear at high λ.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # TODO: implement weight decay plot body

    if save_path:
        _save(fig, save_path)
    return fig
