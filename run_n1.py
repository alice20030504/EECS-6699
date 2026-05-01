"""
N1: Width × Noise 2-D Phase Diagram — core contribution.

Full grid (9 × 6 × 2 seeds = 108 runs):
  k    ∈ {1, 2, 3, 4, 6, 8, 16, 32, 64}
  η    ∈ {0%, 5%, 10%, 20%, 30%, 40%}

k=1           — underfitting regime anchor
k=3           — fills gap between k=2 and k=4
k=6           — interpolation threshold at η=15% (matches R1)
k=8 → k=64   — benign overfitting region

Phase thresholds (per-column baseline, i.e. same k at η=0%):
  Benign:       Δ < 5%
  Tempered:     5% ≤ Δ < 15%
  Catastrophic: Δ ≥ 15%

Usage
-----
# Supplement run: only the 3 new k values (36 runs, ~4h on T4)
python run_n1.py --widths 1 3 6

# Account A (noise split)
python run_n1.py --noise_rates 0.0 0.05

# Plot only after all results collected
python run_n1.py --plot_only

# With Google Drive
python run_n1.py --widths 1 3 6 --drive
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data       import get_cifar10_loaders
from src.models     import CNN5
from src.train      import train_one_run
from src.io_utils   import save_result, load_results, mount_drive_if_colab, get_result_dir
from src.plot_utils import set_style, PALETTE

# ── Experiment configuration ──────────────────────────────────────────────────

N1_CONFIG = {
    'experiment':    'N1',
    # Data
    'n_train':       5000,
    'data_seed':     42,
    'batch_size':    128,
    # Model
    'activation':    'relu',
    'n_classes':     10,
    # Full 9-point width sweep
    # k=1: underfitting anchor  k=3: fills k=2→4 gap  k=6: interpolation threshold
    'widths':      [1, 2, 3, 4, 6, 8, 16, 32, 64],
    'noise_rates': [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
    'seeds':       [42, 123],
    # Training
    'optimizer':     'adam',
    'lr':            1e-3,
    'weight_decay':  0.0,
    'epochs':        300,
    # Phase thresholds — per-column baseline (same k at η=0%)
    # Relaxed vs. original plan: wider benign/tempered bands fit empirical results
    'benign_thresh':     0.05,   # Δ < 5%
    'tempered_thresh':   0.15,   # 5% ≤ Δ < 15%
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_n1(cfg: dict, result_dir: str,
           noise_rates=None, widths=None,
           resume: bool = True) -> list[dict]:
    import torch
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_noise  = noise_rates if noise_rates is not None else cfg['noise_rates']
    active_widths = widths      if widths      is not None else cfg['widths']

    print(f"\n{'='*60}")
    print(f"Experiment N1: Width × Noise Phase Diagram   [device: {device_str}]")
    print(f"Widths:       {active_widths}")
    print(f"Noise rates:  {[f'{η:.0%}' for η in active_noise]}")
    print(f"Seeds:        {cfg['seeds']}")
    print(f"Total runs:   {len(active_widths) * len(active_noise) * len(cfg['seeds'])}")
    print(f"Result dir:   {result_dir}")
    print(f"{'='*60}\n")

    ckpt_dir    = str(Path(result_dir) / 'checkpoints')
    all_results = []

    for eta in active_noise:
        for k in active_widths:
            for seed in cfg['seeds']:
                eta_pct = round(eta * 100)
                run_id  = f"n1_k{k:03d}_eta{eta_pct:03d}_s{seed}"
                result_path = Path(result_dir) / f"{run_id}.json"

                if resume and result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                    print(
                        f"[skip] {run_id:25s} | "
                        f"train_err={result['train_error']:.3f}  "
                        f"test_err={result['test_error']:.3f}"
                    )
                    all_results.append(result)
                    continue

                print(f"\n[run ] {run_id} | k={k}, η={eta:.0%}, seed={seed}")
                train_loader, test_loader = get_cifar10_loaders(
                    n_train=cfg['n_train'],
                    noise_rate=eta,
                    data_seed=cfg['data_seed'],
                    batch_size=cfg['batch_size'],
                )
                model = CNN5(
                    width_multiplier=k,
                    n_classes=cfg['n_classes'],
                    activation=cfg['activation'],
                )
                print(f"       params = {model.count_params():,}")

                result = train_one_run(
                    model, train_loader, test_loader,
                    optimizer_name=cfg['optimizer'],
                    lr=cfg['lr'],
                    weight_decay=cfg['weight_decay'],
                    epochs=cfg['epochs'],
                    checkpoint_dir=ckpt_dir,
                    run_id=run_id,
                )
                result.update({
                    'experiment':       'N1',
                    'run_id':           run_id,
                    'width_multiplier': k,
                    'noise_rate':       eta,
                    'seed':             seed,
                    'config':           cfg,
                })
                save_result(result, str(result_path))
                all_results.append(result)

    print(f"\nN1 batch complete — {len(all_results)} runs in {result_dir}")
    return all_results


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _aggregate(results: list[dict]) -> tuple[list, list, np.ndarray]:
    """Return (widths, noises, mean_test_errors) averaged over seeds."""
    widths = sorted({r['width_multiplier'] for r in results})
    noises = sorted({r['noise_rate']       for r in results})

    # (n_noises, n_widths) matrix, mean over seeds
    matrix = np.full((len(noises), len(widths)), np.nan)
    for r in results:
        wi = widths.index(r['width_multiplier'])
        ni = noises.index(r['noise_rate'])
        prev = matrix[ni, wi]
        matrix[ni, wi] = r['test_error'] if np.isnan(prev) else (prev + r['test_error']) / 2

    return widths, noises, matrix


def _classify(delta: float, benign_t: float, tempered_t: float) -> str:
    if delta < benign_t:
        return 'benign'
    elif delta < tempered_t:
        return 'tempered'
    return 'catastrophic'


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_n1(result_dir: str, cfg: dict = None) -> None:
    """Load results from *result_dir* and generate Fig 3, 4, 5."""
    if cfg is None:
        cfg = N1_CONFIG

    results = load_results(result_dir, pattern='n1_*.json')
    if not results:
        print("[plot] No N1 results found — skipping.")
        return

    widths, noises, te_matrix = _aggregate(results)

    # ── Fig 3: test-error heatmap ─────────────────────────────────────────
    _plot_heatmap(widths, noises, te_matrix, result_dir)

    # ── Fig 4: DD curves, one per noise level ────────────────────────────
    _plot_dd_overlay(widths, noises, te_matrix, result_dir)

    # ── Fig 5: three-region phase diagram ────────────────────────────────
    baseline_per_k  = _get_baseline_per_k(te_matrix, noises)
    baseline_global = _get_baseline(te_matrix, noises)
    _plot_phase_diagram(widths, noises, te_matrix,
                        baseline_per_k, baseline_global, cfg, result_dir)

    # ── Benign boundary fit (uses global baseline) ───────────────────────
    _fit_benign_boundary(widths, noises, te_matrix, baseline_global, cfg, result_dir)


def _plot_heatmap(widths, noises, te_matrix, result_dir):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    im = ax.imshow(
        te_matrix * 100, aspect='auto', origin='lower',
        cmap='RdYlGn_r', vmin=0, vmax=100,
        extent=[-0.5, len(widths) - 0.5, -0.5, len(noises) - 0.5],
    )
    # Annotate cells
    for ni in range(len(noises)):
        for wi in range(len(widths)):
            val = te_matrix[ni, wi]
            if not np.isnan(val):
                ax.text(wi, ni, f'{val*100:.1f}', ha='center', va='center',
                        fontsize=8, color='black')

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([f'{η:.0%}' for η in noises])
    ax.set_xlabel('Width Multiplier $k$')
    ax.set_ylabel('Label Noise Rate $\\eta$')
    ax.set_title('Fig 3 — Test Error Heatmap (n=5000, CNN5)')
    plt.colorbar(im, ax=ax, label='Test Error (%)')

    plt.tight_layout()
    path = str(Path(result_dir) / 'fig3_n1_heatmap.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Fig 3 saved → {path}")


def _plot_dd_overlay(widths, noises, te_matrix, result_dir):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(noises) - 1, 1)) for i in range(len(noises))]

    for ni, (eta, color) in enumerate(zip(noises, colors)):
        row = te_matrix[ni]
        valid = ~np.isnan(row)
        if valid.any():
            ax.plot(
                np.array(widths)[valid], row[valid],
                'o-', color=color, label=f'η={eta:.0%}',
            )

    ax.set_xscale('log', base=2)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(widths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Width Multiplier $k$')
    ax.set_ylabel('Test Error')
    ax.set_title('Fig 4 — Double Descent Curves at Different Noise Levels')
    ax.legend(title='Noise Rate', loc='upper right')

    plt.tight_layout()
    path = str(Path(result_dir) / 'fig4_n1_dd_overlay.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Fig 4 saved → {path}")


def _get_baseline_per_k(te_matrix, noises) -> np.ndarray:
    """Per-column baseline: test error at η=0 for each k.

    Δ(k, η) = TestErr(k, η) − TestErr(k, η=0)
    This measures: 'how much does noise hurt THIS width?'
    rather than comparing against the global best.
    """
    if 0.0 in noises:
        return te_matrix[noises.index(0.0)].copy()
    # Fallback: column-wise minimum
    return np.nanmin(te_matrix, axis=0)


def _get_baseline(te_matrix, noises) -> float:
    """Global baseline (used only for title annotation)."""
    if 0.0 in noises:
        row = te_matrix[noises.index(0.0)]
        return float(np.nanmin(row))
    return float(np.nanmin(te_matrix))


def _plot_phase_diagram(widths, noises, te_matrix,
                        baseline_per_k, baseline_global, cfg, result_dir):
    """Phase diagram using per-column baseline: Δ(k,η) = TestErr(k,η) − TestErr(k,0)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bt = cfg['benign_thresh']
    tt = cfg['tempered_thresh']

    set_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    _COLORS = {
        'benign':       PALETTE['benign'],
        'tempered':     PALETTE['tempered'],
        'catastrophic': PALETTE['catastrophic'],
    }

    for ni, eta in enumerate(noises):
        if eta == 0.0:
            continue   # skip baseline row (Δ=0 by definition)
        for wi, k in enumerate(widths):
            val = te_matrix[ni, wi]
            bl  = baseline_per_k[wi]
            if np.isnan(val) or np.isnan(bl):
                continue
            delta  = val - bl
            region = _classify(delta, bt, tt)
            ax.scatter(k, eta, c=_COLORS[region], s=240, zorder=3,
                       marker='s', edgecolors='white', linewidths=0.5)
            ax.text(k, eta, f'{val*100:.0f}', ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold', zorder=4)

    ax.set_xscale('log', base=2)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(widths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Width Multiplier $k$')
    ax.set_ylabel('Label Noise Rate $\\eta$')
    ax.set_title(
        f'Fig 5 — Overfitting Phase Diagram\n'
        f'(Δ = TestErr(k,η) − TestErr(k,0),  '
        f'benign Δ<{bt*100:.0f}%, tempered Δ<{tt*100:.0f}%)'
    )

    legend_elements = [
        Patch(facecolor=_COLORS['benign'],       label=f'Benign  (Δ < {bt*100:.0f}%)'),
        Patch(facecolor=_COLORS['tempered'],     label=f'Tempered ({bt*100:.0f}%–{tt*100:.0f}%)'),
        Patch(facecolor=_COLORS['catastrophic'], label=f'Catastrophic (≥ {tt*100:.0f}%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = str(Path(result_dir) / 'fig5_n1_phase_diagram.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Fig 5 saved → {path}")


def _fit_benign_boundary(widths, noises, te_matrix, baseline_global, cfg, result_dir):
    """
    Benign boundary: for each η, find the smallest k such that
        TestErr(k, η) < baseline_global + benign_thresh
    where baseline_global = min TestErr at η=0 (best achievable performance).

    This answers: "how wide does the network need to be to perform
    near-optimally despite noise level η?"
    Fit w_benign(η) ≈ c · η^α via log-log linear regression.
    """
    import matplotlib.pyplot as plt

    bt = cfg['benign_thresh']
    absolute_threshold = baseline_global + bt   # e.g. 33.4% + 5% = 38.4%

    boundary = {}   # η → min k achieving near-optimal performance
    for ni, eta in enumerate(noises):
        if eta == 0.0:
            continue
        for wi, k in enumerate(widths):
            val = te_matrix[ni, wi]
            if np.isnan(val):
                continue
            if val < absolute_threshold:
                boundary[eta] = k
                break   # first (smallest) k that meets the criterion

    if len(boundary) < 2:
        print("[boundary] Not enough benign points for curve fitting.")
        return

    etas_b = np.array(sorted(boundary.keys()))
    ks_b   = np.array([boundary[η] for η in etas_b])

    # Log-log fit: log(k) = log(c) + α·log(η)
    log_eta = np.log(etas_b)
    log_k   = np.log(ks_b)
    alpha, log_c = np.polyfit(log_eta, log_k, 1)
    c = np.exp(log_c)

    print(f"\n[boundary] Empirical benign boundary fit:")
    print(f"  Criterion: TestErr(k,η) < {baseline_global*100:.1f}% + {bt*100:.0f}% = {absolute_threshold*100:.1f}%")
    print(f"  w_benign(η) ≈ {c:.2f} · η^{alpha:.3f}")
    print(f"  Data points: η={list(etas_b)}, k={list(ks_b)}")

    # Save fit result
    fit_result = {
        'formula':             f'w_benign = {c:.4f} * eta^{alpha:.4f}',
        'c':                   c,
        'alpha':               alpha,
        'baseline_global':     baseline_global,
        'absolute_threshold':  absolute_threshold,
        'data':                {f'{η:.2f}': int(k) for η, k in zip(etas_b, ks_b)},
    }
    fit_path = Path(result_dir) / 'benign_boundary_fit.json'
    with open(fit_path, 'w') as f:
        json.dump(fit_result, f, indent=2)
    print(f"  Saved → {fit_path}")

    # Plot boundary
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(etas_b, ks_b, color=PALETTE['benign'], s=80, zorder=3,
               label='Empirical boundary')

    eta_fine = np.linspace(etas_b.min(), etas_b.max(), 200)
    ax.plot(eta_fine, c * eta_fine ** alpha, '--',
            color=PALETTE['catastrophic'],
            label=f'Fit: $w_{{\\rm benign}} \\approx {c:.1f}\\,\\eta^{{{alpha:.2f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Label Noise Rate $\\eta$')
    ax.set_ylabel('Min Width $k$ for Near-Optimal Performance')
    ax.set_title(
        f'Fig 5b — Empirical Benign Boundary $w_{{\\rm benign}}(\\eta)$\n'
        f'(criterion: TestErr $<$ {baseline_global*100:.1f}% + {bt*100:.0f}% = {absolute_threshold*100:.1f}%)'
    )
    ax.legend()

    plt.tight_layout()
    path = str(Path(result_dir) / 'fig5b_benign_boundary.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Benign boundary saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run N1: Width × Noise phase diagram')
    parser.add_argument('--drive',       action='store_true', help='Save to Google Drive')
    parser.add_argument('--result_dir',  default=None,        help='Override result dir')
    parser.add_argument('--no_resume',   action='store_true', help='Re-run all')
    parser.add_argument('--plot_only',   action='store_true', help='Skip training, plot only')
    parser.add_argument('--noise_rates', nargs='+', type=float, default=None,
                        metavar='ETA',
                        help='Subset of noise rates, e.g. --noise_rates 0.0 0.05')
    parser.add_argument('--widths', nargs='+', type=int, default=None,
                        metavar='K',
                        help='Subset of widths to run, e.g. --widths 1 3 6')
    args = parser.parse_args()

    if args.drive:
        mount_drive_if_colab()

    result_dir = args.result_dir or get_result_dir('.', 'N1', use_drive=args.drive)

    if not args.plot_only:
        run_n1(N1_CONFIG, result_dir,
               noise_rates=args.noise_rates,
               widths=args.widths,
               resume=not args.no_resume)

    plot_n1(result_dir, N1_CONFIG)
