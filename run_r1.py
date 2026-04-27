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
# Local
python run_r1.py

# Colab with Google Drive backup
python run_r1.py --drive

# Force re-run all (ignore existing results)
python run_r1.py --no_resume
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
from src.plot_utils import plot_double_descent

# ── Experiment configuration ──────────────────────────────────────────────────

R1_CONFIG = {
    'experiment':   'R1',
    # Data
    'n_train':      5000,
    'noise_rate':   0.15,
    'data_seed':    42,
    'batch_size':   128,
    # Model
    'activation':   'relu',
    'n_classes':    10,
    # Sweep
    'widths': [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
    'seeds':  [42, 123],
    # Training
    'optimizer':    'adam',
    'lr':           1e-3,
    'weight_decay': 0.0,
    'epochs':       300,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_r1(cfg: dict, result_dir: str, resume: bool = True) -> list[dict]:
    import torch
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Experiment R1: CNN Double Descent   [device: {device_str}]")
    print(f"Result dir: {result_dir}")
    print(f"{'='*60}\n")

    ckpt_dir    = str(Path(result_dir) / 'checkpoints')
    all_results = []

    for k in cfg['widths']:
        for seed in cfg['seeds']:
            run_id       = f"r1_k{k:03d}_s{seed}"
            result_path  = Path(result_dir) / f"{run_id}.json"

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
                epochs=cfg['epochs'],
                checkpoint_dir=ckpt_dir,
                run_id=run_id,
            )
            result.update({
                'experiment':      'R1',
                'run_id':          run_id,
                'width_multiplier': k,
                'seed':            seed,
                'noise_rate':      cfg['noise_rate'],
                'config':          cfg,
            })
            save_result(result, str(result_path))
            all_results.append(result)

    print(f"\nR1 complete — {len(all_results)} runs in {result_dir}")
    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_r1(result_dir: str) -> None:
    """Load results and save Figure 1."""
    results = load_results(result_dir, pattern='r1_*.json')
    if not results:
        print("[plot] No R1 results found — skipping.")
        return

    widths = sorted({r['width_multiplier'] for r in results})
    seeds  = sorted({r['seed'] for r in results})

    # Build (n_seeds, n_widths) arrays
    train_errors = np.full((len(seeds), len(widths)), np.nan)
    test_errors  = np.full((len(seeds), len(widths)), np.nan)

    for r in results:
        si = seeds.index(r['seed'])
        wi = widths.index(r['width_multiplier'])
        train_errors[si, wi] = r['train_error']
        test_errors[si,  wi] = r['test_error']

    save_path = str(Path(result_dir) / 'fig1_r1_dd.png')
    plot_double_descent(
        widths, train_errors, test_errors,
        title=f'R1 — CNN Double Descent (n=5000, η=15%)',
        save_path=save_path,
    )
    print(f"[plot] Figure 1 saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run R1: CNN double descent')
    parser.add_argument('--drive',     action='store_true', help='Save results to Google Drive')
    parser.add_argument('--result_dir', default=None,       help='Override result directory')
    parser.add_argument('--no_resume', action='store_true', help='Re-run all (ignore cached)')
    parser.add_argument('--plot_only', action='store_true', help='Skip training, only plot')
    args = parser.parse_args()

    if args.drive:
        mount_drive_if_colab()

    result_dir = args.result_dir or get_result_dir('.', 'R1', use_drive=args.drive)

    if not args.plot_only:
        run_r1(R1_CONFIG, result_dir, resume=not args.no_resume)

    plot_r1(result_dir)
