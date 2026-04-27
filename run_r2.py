"""
R2: ResNet-18 (width-scalable) Double Descent on CIFAR-10.

Validates that the DD phenomenon observed in R1 (CNN) generalises to a deeper,
residual architecture.  Training uses SGD + cosine-decay (standard ResNet recipe).

Sweep: width multiplier k ∈ {1, 2, 4, 8, 16, 32}, 1 seed each → 6 training runs.

Channel progression: [k, 2k, 4k, 8k] across 4 ResNet stages.
  k=1 → ~3 K params   (under-parameterized for n=5 000)
  k=2 → ~12 K params  (near interpolation threshold)
  k=4 → ~50 K params  (over-parameterized)

Outputs
-------
results/R2/<run_id>.json   — one file per k
results/R2/fig2_r2_dd.png  — Figure 2: DD curve on ResNet-18

Usage
-----
python run_r2.py
python run_r2.py --drive
python run_r2.py --no_resume
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data       import get_cifar10_loaders
from src.models     import WideResNet18
from src.train      import train_one_run
from src.io_utils   import save_result, load_results, mount_drive_if_colab, get_result_dir
from src.plot_utils import plot_double_descent

# ── Experiment configuration ──────────────────────────────────────────────────

R2_CONFIG = {
    'experiment':   'R2',
    # Data — identical subset to R1 for fair comparison
    'n_train':      5000,
    'noise_rate':   0.15,
    'data_seed':    42,
    'batch_size':   128,
    # Model
    'n_classes':    10,
    # Sweep
    'widths': [1, 2, 4, 8, 16, 32],
    'seeds':  [42],       # single seed (validation role)
    # Training — SGD + cosine is the standard ResNet recipe
    'optimizer':    'sgd',
    'lr':           0.1,
    'weight_decay': 5e-4,
    'epochs':       200,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_r2(cfg: dict, result_dir: str, resume: bool = True) -> list[dict]:
    import torch
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Experiment R2: ResNet-18 Double Descent   [device: {device_str}]")
    print(f"Result dir: {result_dir}")
    print(f"{'='*60}\n")

    ckpt_dir    = str(Path(result_dir) / 'checkpoints')
    all_results = []

    for k in cfg['widths']:
        for seed in cfg['seeds']:
            run_id      = f"r2_k{k:03d}_s{seed}"
            result_path = Path(result_dir) / f"{run_id}.json"

            # ── Skip completed runs ───────────────────────────────────────
            if resume and result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)
                print(
                    f"[skip] {run_id:20s} | "
                    f"train_err={result['train_error']:.3f}  "
                    f"test_err={result['test_error']:.3f}"
                )
                all_results.append(result)
                continue

            # ── Run ───────────────────────────────────────────────────────
            print(f"\n[run ] {run_id} | k={k}, seed={seed}")
            torch.manual_seed(seed)

            train_loader, test_loader = get_cifar10_loaders(
                n_train=cfg['n_train'],
                noise_rate=cfg['noise_rate'],
                data_seed=cfg['data_seed'],
                batch_size=cfg['batch_size'],
            )
            model = WideResNet18(width_multiplier=k, n_classes=cfg['n_classes'])
            print(f"       params = {model.count_params():,}")

            result = train_one_run(
                model, train_loader, test_loader,
                optimizer_name=cfg['optimizer'],
                lr=cfg['lr'],
                weight_decay=cfg['weight_decay'],
                epochs=cfg['epochs'],
                checkpoint_dir=ckpt_dir,
                checkpoint_every=10,
                eval_every=10,
                run_id=run_id,
            )
            result.update({
                'experiment':       'R2',
                'run_id':           run_id,
                'width_multiplier': k,
                'seed':             seed,
                'noise_rate':       cfg['noise_rate'],
                'config':           cfg,
            })
            save_result(result, str(result_path))
            all_results.append(result)

    print(f"\nR2 complete — {len(all_results)} runs in {result_dir}")
    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_r2(result_dir: str) -> None:
    """Load results and save Figure 2 with interpolation threshold + param-count axis."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from src.models import WideResNet18
    from src.plot_utils import set_style, PALETTE

    results = load_results(result_dir, pattern='r2_*.json')
    if not results:
        print("[plot] No R2 results found — skipping.")
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

    # Interpolation threshold
    interp_k = None
    for i, k in enumerate(widths):
        if tr_mean[i] < 0.02:
            interp_k = k
            break

    # Parameter counts
    param_counts = [WideResNet18(width_multiplier=k).count_params() for k in widths]

    set_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(widths, tr_mean, 'o-', color=PALETTE['train'], label='Train Error')
    ax1.plot(widths, te_mean, 's-', color=PALETTE['test'],  label='Test Error')

    if len(seeds) > 1:
        ax1.fill_between(widths, np.nanmin(train_errors, 0), np.nanmax(train_errors, 0),
                         alpha=0.15, color=PALETTE['train'])
        ax1.fill_between(widths, np.nanmin(test_errors, 0),  np.nanmax(test_errors, 0),
                         alpha=0.15, color=PALETTE['test'])

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
    ax1.set_title('R2 — ResNet-18 Double Descent (n=5000, η=15%)')
    ax1.legend(loc='upper right')

    ax2 = ax1.twiny()
    ax2.set_xscale('log', base=2)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(widths)
    ax2.set_xticklabels(
        [f'{p/1e3:.0f}K' if p >= 1000 else str(p) for p in param_counts],
        fontsize=7, rotation=30,
    )
    ax2.set_xlabel('Parameter Count', fontsize=9)

    plt.tight_layout()
    save_path = str(Path(result_dir) / 'fig2_r2_dd.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot] Figure 2 saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run R2: ResNet-18 double descent')
    parser.add_argument('--drive',      action='store_true', help='Save results to Google Drive')
    parser.add_argument('--result_dir', default=None,        help='Override result directory')
    parser.add_argument('--no_resume',  action='store_true', help='Re-run all (ignore cached)')
    parser.add_argument('--plot_only',  action='store_true', help='Skip training, only plot')
    args = parser.parse_args()

    if args.drive:
        mount_drive_if_colab()

    result_dir = args.result_dir or get_result_dir('.', 'R2', use_drive=args.drive)

    if not args.plot_only:
        run_r2(R2_CONFIG, result_dir, resume=not args.no_resume)

    plot_r2(result_dir)
