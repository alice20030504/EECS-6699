"""
N1: Width × Noise 2-D Phase Diagram — core contribution.

Sweeps a 6×6 grid:
  k    ∈ {2, 4, 8, 16, 32, 64}
  η    ∈ {0%, 5%, 10%, 20%, 30%, 40%}
  seeds = [42, 123]
→ 72 training runs total.

Each run saves one JSON to results/N1/.
After all runs complete, generates Fig 3 (heatmap), Fig 4 (DD overlay),
Fig 5 (three-region phase diagram), and fits the empirical benign boundary.

Parallelisation across 4 Colab accounts: pass --noise_rates to restrict which
η values this account handles.

Usage
-----
# Full sweep (single account)
python run_n1.py

# Account A
python run_n1.py --noise_rates 0.0 0.05

# Account B
python run_n1.py --noise_rates 0.10 0.20

# Account C
python run_n1.py --noise_rates 0.30 0.40

# Plot only (after merging all results into one dir)
python run_n1.py --plot_only --result_dir /path/to/merged/N1

# With Google Drive
python run_n1.py --noise_rates 0.0 0.05 --drive
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
    # Sweep
    'widths':      [2, 4, 8, 16, 32, 64],
    'noise_rates': [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
    'seeds':       [42, 123],
    # Training
    'optimizer':     'adam',
    'lr':            1e-3,
    'weight_decay':  0.0,
    'epochs':        300,
    # Phase classification thresholds (plan Sec 4)
    'benign_thresh':     0.03,   # Δ < 3%
    'tempered_thresh':   0.10,   # 3% ≤ Δ < 10%
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_n1(cfg: dict, result_dir: str, noise_rates=None, resume: bool = True) -> list[dict]:
    import torch
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_noise = noise_rates if noise_rates is not None else cfg['noise_rates']

    print(f"\n{'='*60}")
    print(f"Experiment N1: Width × Noise Phase Diagram   [device: {device_str}]")
    print(f"Widths:       {cfg['widths']}")
    print(f"Noise rates:  {[f'{η:.0%}' for η in active_noise]}")
    print(f"Seeds:        {cfg['seeds']}")
    print(f"Total runs:   {len(cfg['widths']) * len(active_noise) * len(cfg['seeds'])}")
    print(f"Result dir:   {result_dir}")
    print(f"{'='*60}\n")

    ckpt_dir    = str(Path(result_dir) / 'checkpoints')
    all_results = []

    for eta in active_noise:
        for k in cfg['widths']:
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
    baseline = _get_baseline(te_matrix, noises)
    _plot_phase_diagram(widths, noises, te_matrix, baseline, cfg, result_dir)

    # ── Benign boundary fit ───────────────────────────────────────────────
    _fit_benign_boundary(widths, noises, te_matrix, baseline, cfg, result_dir)


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


def _get_baseline(te_matrix, noises) -> float:
    """Baseline = test error at η=0, best width (minimum)."""
    if 0.0 in noises:
        eta0_row = te_matrix[noises.index(0.0)]
        valid = eta0_row[~np.isnan(eta0_row)]
        if len(valid):
            return float(valid.min())
    # Fallback: global min
    return float(np.nanmin(te_matrix))


def _plot_phase_diagram(widths, noises, te_matrix, baseline, cfg, result_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bt = cfg['benign_thresh']
    tt = cfg['tempered_thresh']

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    _COLORS = {
        'benign':       PALETTE['benign'],
        'tempered':     PALETTE['tempered'],
        'catastrophic': PALETTE['catastrophic'],
    }

    for ni, eta in enumerate(noises):
        for wi, k in enumerate(widths):
            val = te_matrix[ni, wi]
            if np.isnan(val):
                continue
            delta  = val - baseline
            region = _classify(delta, bt, tt)
            ax.scatter(k, eta, c=_COLORS[region], s=200, zorder=3,
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
        f'(baseline={baseline*100:.1f}%, '
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


def _fit_benign_boundary(widths, noises, te_matrix, baseline, cfg, result_dir):
    """
    For each η, find the smallest k that satisfies the benign condition.
    Fit w_benign(η) ≈ c · η^α via log-log linear regression.
    Save a summary table and an extra figure.
    """
    import matplotlib.pyplot as plt

    bt = cfg['benign_thresh']

    boundary = {}   # η → min benign k (or None)
    for ni, eta in enumerate(noises):
        if eta == 0.0:
            continue
        for wi, k in enumerate(widths):
            val = te_matrix[ni, wi]
            if np.isnan(val):
                continue
            if (val - baseline) < bt:
                boundary[eta] = k
                break   # first (smallest) benign k found

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
    print(f"  w_benign(η) ≈ {c:.2f} · η^{alpha:.3f}")
    print(f"  Data points: η={list(etas_b)}, k={list(ks_b)}")

    # Save fit result
    fit_result = {
        'formula': f'w_benign = {c:.4f} * eta^{alpha:.4f}',
        'c':       c,
        'alpha':   alpha,
        'data':    {f'{η:.2f}': int(k) for η, k in zip(etas_b, ks_b)},
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
    ax.set_ylabel('Min Benign Width $k$')
    ax.set_title('Empirical Benign Boundary $w_{\\rm benign}(\\eta)$')
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
                        help='Subset of noise rates to run, e.g. --noise_rates 0.0 0.05')
    args = parser.parse_args()

    if args.drive:
        mount_drive_if_colab()

    result_dir = args.result_dir or get_result_dir('.', 'N1', use_drive=args.drive)

    if not args.plot_only:
        run_n1(N1_CONFIG, result_dir,
               noise_rates=args.noise_rates,
               resume=not args.no_resume)

    plot_n1(result_dir, N1_CONFIG)
