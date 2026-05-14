"""Publication-quality plotting utilities shared across all experiments.

Style conventions match a typical ML paper (serif font, clean axes, log-scale
x-axis for width sweeps).  All figures use the same colour palette so that
train/test/benign/tempered/catastrophic are visually consistent throughout
the paper.

Figures produced:
  plot_double_descent()  → R1 Fig 1, R2 Fig 2
  plot_phase_diagram()   → N1 Figs 3-5  (called from run_n1.py later)
  plot_wd_ablation()     → N2 Fig 6     (called from run_n2.py later)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    'train':        '#2166AC',
    'test':         '#D6604D',
    'benign':       '#4DAC26',
    'tempered':     '#F1A340',
    'catastrophic': '#D7191C',
}

# ── Global style ──────────────────────────────────────────────────────────────

def set_style() -> None:
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         11,
        'axes.titlesize':    12,
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.fontsize':   10,
        'figure.dpi':        150,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'lines.linewidth':   1.8,
        'lines.markersize':  6,
    })


# ── Figure 1 / Figure 2: Double Descent curve ─────────────────────────────────

def plot_double_descent(
    widths:       Sequence[int],
    train_errors: np.ndarray,
    test_errors:  np.ndarray,
    title:        str  = 'Double Descent',
    save_path:    str  = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot train/test error vs width multiplier (log scale).

    Args:
        widths:       1-D array of width multiplier values.
        train_errors: Shape (n_seeds, n_widths) or (n_widths,).
        test_errors:  Same shape as train_errors.
        title:        Figure title.
        save_path:    If given, save PNG to this path.

    Returns:
        (fig, ax) tuple.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    train_errors = np.atleast_2d(train_errors)  # → (n_seeds, n_widths)
    test_errors  = np.atleast_2d(test_errors)

    tr_mean, te_mean = train_errors.mean(0), test_errors.mean(0)

    ax.plot(widths, tr_mean, 'o-', color=PALETTE['train'], label='Train Error')
    ax.plot(widths, te_mean, 's-', color=PALETTE['test'],  label='Test Error')

    if train_errors.shape[0] > 1:
        ax.fill_between(widths, train_errors.min(0), train_errors.max(0),
                        alpha=0.15, color=PALETTE['train'])
        ax.fill_between(widths, test_errors.min(0),  test_errors.max(0),
                        alpha=0.15, color=PALETTE['test'])

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(widths)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Width Multiplier $k$')
    ax.set_ylabel('Error Rate')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[plot] saved → {save_path}")
    return fig, ax


# ── Figure 3-5: Phase diagram (stub — filled in by N1) ───────────────────────

def plot_phase_diagram(
    widths:     Sequence[int],
    noises:     Sequence[float],
    test_errors: np.ndarray,
    baseline_error: float,
    title:      str = 'Overfitting Phase Diagram',
    save_path:  str = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Heat-map and three-region phase diagram (used by N1).

    Args:
        widths:         Width multiplier values (x-axis).
        noises:         Noise-rate values (y-axis).
        test_errors:    Shape (n_noises, n_widths), mean over seeds.
        baseline_error: TestErr(w*, η=0) — used to compute Δ for region classification.
        title:          Figure title.
        save_path:      If given, save PNG.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Left: heat-map of raw test error ──────────────────────────────────
    ax = axes[0]
    im = ax.imshow(
        test_errors * 100, aspect='auto', origin='lower',
        extent=[0, len(widths), 0, len(noises)],
        cmap='RdYlGn_r',
    )
    ax.set_xticks(np.arange(len(widths)) + 0.5)
    ax.set_xticklabels([str(w) for w in widths], rotation=45)
    ax.set_yticks(np.arange(len(noises)) + 0.5)
    ax.set_yticklabels([f'{η:.0%}' for η in noises])
    ax.set_xlabel('Width Multiplier $k$')
    ax.set_ylabel('Label Noise Rate $\\eta$')
    ax.set_title('Test Error (%)')
    plt.colorbar(im, ax=ax)

    # ── Right: three-region classification ────────────────────────────────
    ax2 = axes[1]
    delta = test_errors - baseline_error
    region_colors = np.zeros((*delta.shape, 4))
    benign_rgb       = matplotlib.colors.to_rgba(PALETTE['benign'])
    tempered_rgb     = matplotlib.colors.to_rgba(PALETTE['tempered'])
    catastrophic_rgb = matplotlib.colors.to_rgba(PALETTE['catastrophic'])

    region_colors[delta < 0.05]                   = benign_rgb
    region_colors[(delta >= 0.05) & (delta < 0.15)] = tempered_rgb
    region_colors[delta >= 0.15]                  = catastrophic_rgb

    ax2.imshow(
        region_colors, aspect='auto', origin='lower',
        extent=[0, len(widths), 0, len(noises)],
    )
    ax2.set_xticks(np.arange(len(widths)) + 0.5)
    ax2.set_xticklabels([str(w) for w in widths], rotation=45)
    ax2.set_yticks(np.arange(len(noises)) + 0.5)
    ax2.set_yticklabels([f'{η:.0%}' for η in noises])
    ax2.set_xlabel('Width Multiplier $k$')
    ax2.set_title('Region (green=benign, orange=tempered, red=catastrophic)')

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE['benign'],       label='Benign  (Δ < 3%)'),
        Patch(facecolor=PALETTE['tempered'],     label='Tempered (3%–10%)'),
        Patch(facecolor=PALETTE['catastrophic'], label='Catastrophic (≥ 10%)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[plot] saved → {save_path}")
    return fig, axes
