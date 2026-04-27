"""
R1: 5-layer CNN Double Descent on CIFAR-10 (5 000-sample subset, 15% label noise).

Sweep: width multiplier k ∈ {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64}
       × 2 random seeds → 22 training runs.

Outputs
-------
results/R1/<run_id>.json   — one file per (k, seed), auto-skipped on re-run
results/R1/fig1_r1_dd.png  — Figure 1: double descent curve

Usage
-----
# Normal run (skips already-completed JSONs)
python run_r1.py

# Plot only — regenerate Fig 1 from existing JSONs, no training
python run_r1.py --plot_only

# Extend k=64 runs to 500 epochs (deletes their JSONs, reruns only those 2)
python run_r1.py --extend_k 64 --extend_epochs 500

# Colab with Google Drive
python run_r1.py --drive
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data     import get_cifar10_loaders
from src.models   import CNN5
from src.train    import train_one_run
from src.io_utils import save_result, load_results, mount_drive_if_colab, get_result_dir

# ── Experiment configuration ──────────────────────────────────────────────────

R1_CONFIG = {
    'experiment':   'R1',
    'n_train':      5000,
    'noise_rate':   0.15,
    'data_seed':    42,
    'batch_size':   128,
    'activation':   'relu',
    'n_classes':    10,
    'widths': [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
    'seeds':  [42, 123],
    'optimizer':    'adam',
    'lr':           1e-3,
    'weight_decay': 0.0,
    'epochs':       300,
    # k=64 gets extra epochs for better benign-region convergence
    'epochs_wide':  500,
    'wide_threshold': 48,   # k >= this value uses epochs_wide
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_r1(cfg: dict, result_dir: str, resume: bool = True,
           extend_k: int = None, extend_epochs: int = None) -> list[dict]:
    """
    Train all (k, seed) combinations.

    extend_k / extend_epochs: if set, delete JSONs for runs with
    width_multiplier == extend_k and rerun them with extend_epochs.
    All other runs are still skipped if their JSON exists.
    """
    import torch
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Experiment R1: CNN Double Descent   [device: {device_str}]")
    print(f"Result dir: {result_dir}")
    if extend_k:
        print(f"Extending k={extend_k} to {extend_epochs} epochs")
    print(f"{'='*60}\n")

    # Delete JSONs for runs being extended
    if extend_k is not None and extend_epochs is not None:
        for seed in cfg['seeds']:
            p = Path(result_dir) / f"r1_k{extend_k:03d}_s{seed}.json"
            if p.exists():
                p.unlink()
                print(f"[extend] deleted {p.name} → will rerun with {extend_epochs} epochs")

    ckpt_dir    = str(Path(result_dir) / 'checkpoints')
    all_results = []

    for k in cfg['widths']:
        # Determine epoch count for this width
        if extend_k is not None and k == extend_k:
            epochs = extend_epochs
        elif k >= cfg.get('wide_threshold', 999):
            epochs = cfg.get('epochs_wide', cfg['epochs'])
        else:
            epochs = cfg['epochs']

        for seed in cfg['seeds']:
            run_id      = f"r1_k{k:03d}_s{seed}"
            result_path = Path(result_dir) / f"{run_id}.json"

            # Skip if JSON exists and epoch count matches
            if resume and result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)
                if result.get('epochs', 0) >= epochs:
                    print(
                        f"[skip] {run_id:20s} | "
                        f"train_err={result['train_error']:.3f}  "
                        f"test_err={result['test_error']:.3f}  "
                        f"({result.get('epochs', '?')} ep)"
                    )
                    all_results.append(result)
                    continue
                else:
                    # Fewer epochs than target — delete and rerun
                    result_path.unlink()
                    print(f"[extend] {run_id}: {result['epochs']} ep → {epochs} ep")

            print(f"\n[run ] {run_id} | k={k}, seed={seed}, epochs={epochs}")
            train_loader, test_loader = get_cifar10_loaders(
                n_train=cfg['n_train'],
                noise_rate=cfg['noise_rate'],
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
                epochs=epochs,
                checkpoint_dir=ckpt_dir,
                run_id=run_id,
            )
            result.update({
                'experiment':       'R1',
                'run_id':           run_id,
                'width_multiplier': k,
                'seed':             seed,
                'noise_rate':       cfg['noise_rate'],
                'config':           cfg,
            })
            save_result(result, str(result_path))
            all_results.append(result)

    print(f"\nR1 complete — {len(all_results)} runs in {result_dir}")
    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_r1(result_dir: str) -> None:
    """Load results and save Figure 1 with enhancements:
    - Interpolation threshold marker
    - Secondary x-axis showing parameter count
    - Error bands across seeds
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from src.models import CNN5
    from src.plot_utils import set_style, PALETTE

    results = load_results(result_dir, pattern='r1_*.json')
    if not results:
        print("[plot] No R1 results found — skipping.")
        return

    widths = sorted({r['width_multiplier'] for r in results})
    seeds  = sorted({r['seed'] for r in results})

    train_errors = np.full((len(seeds), len(widths)), np.nan)
    test_errors  = np.full((len(seeds), len(widths)), np.nan)
    for r in results:
        si = seeds.index(r['seed'])
        wi = widths.index(r['width_multiplier'])
        train_errors[si, wi] = r['train_error']
        test_errors[si,  wi] = r['test_error']

    tr_mean = np.nanmean(train_errors, axis=0)
    te_mean = np.nanmean(test_errors,  axis=0)

    # ── Find interpolation threshold (first k where train_err ≈ 0) ────────
    interp_k = None
    for i, k in enumerate(widths):
        if tr_mean[i] < 0.02:   # train error < 2% → interpolating
            interp_k = k
            break

    # ── Parameter counts for secondary axis ───────────────────────────────
    param_counts = [CNN5(width_multiplier=k).count_params() for k in widths]

    set_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Main curves
    ax1.plot(widths, tr_mean, 'o-', color=PALETTE['train'], label='Train Error')
    ax1.plot(widths, te_mean, 's-', color=PALETTE['test'],  label='Test Error')

    # Error bands
    if len(seeds) > 1:
        ax1.fill_between(widths, np.nanmin(train_errors, 0), np.nanmax(train_errors, 0),
                         alpha=0.15, color=PALETTE['train'])
        ax1.fill_between(widths, np.nanmin(test_errors, 0),  np.nanmax(test_errors, 0),
                         alpha=0.15, color=PALETTE['test'])

    # Interpolation threshold marker
    if interp_k is not None:
        ax1.axvline(interp_k, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.text(interp_k * 1.08, ax1.get_ylim()[1] * 0.95,
                 f'Interpolation\nthreshold\n$k={interp_k}$',
                 fontsize=8, color='gray', va='top')

    ax1.set_xscale('log', base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_xticks(widths)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_xlabel('Width Multiplier $k$')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('R1 — CNN Double Descent (n=5000, η=15%)')
    ax1.legend(loc='upper right')

    # Secondary x-axis: parameter count
    ax2 = ax1.twiny()
    ax2.set_xscale('log', base=2)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(widths)
    ax2.set_xticklabels(
        [f'{p/1e3:.0f}K' if p >= 1000 else str(p) for p in param_counts],
        fontsize=7, rotation=30
    )
    ax2.set_xlabel('Parameter Count', fontsize=9)

    plt.tight_layout()
    save_path = str(Path(result_dir) / 'fig1_r1_dd.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Figure 1 saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run R1: CNN double descent')
    parser.add_argument('--drive',          action='store_true')
    parser.add_argument('--result_dir',     default=None)
    parser.add_argument('--no_resume',      action='store_true')
    parser.add_argument('--plot_only',      action='store_true')
    parser.add_argument('--extend_k',       type=int,   default=None,
                        help='Width multiplier to extend (e.g. 64)')
    parser.add_argument('--extend_epochs',  type=int,   default=500,
                        help='New epoch count for extended runs (default 500)')
    args = parser.parse_args()

    if args.drive:
        mount_drive_if_colab()

    result_dir = args.result_dir or get_result_dir('.', 'R1', use_drive=args.drive)

    if not args.plot_only:
        run_r1(R1_CONFIG, result_dir,
               resume=not args.no_resume,
               extend_k=args.extend_k,
               extend_epochs=args.extend_epochs)

    plot_r1(result_dir)
