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
    """Load results and save Figure 2."""
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

    save_path = str(Path(result_dir) / 'fig2_r2_dd.png')
    plot_double_descent(
        widths, train_errors, test_errors,
        title='R2 — ResNet-18 Double Descent (n=5000, η=15%)',
        save_path=save_path,
    )
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
