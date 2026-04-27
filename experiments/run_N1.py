"""N1: (Width × Noise) 2D Phase Diagram — core contribution.

Sweeps a 5×4 grid: k ∈ {2,8,16,32,64} × η ∈ {0%,10%,20%,40%}.
Each (k, η) run saves to results/N1/k{k}_eta{eta_pct}.csv.

Run a single cell (for cluster parallelism):
    python experiments/run_N1.py --k 16 --eta 0.2
"""

import argparse
import yaml
from pathlib import Path

from src.models import build_resnet18
from src.data import get_cifar10_loaders
from src.training import Trainer
from src.utils import CSVLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/N1.yaml")
    p.add_argument("--k", type=int, default=None,
                   help="Single width multiplier to run.")
    p.add_argument("--eta", type=float, default=None,
                   help="Single noise level to run (e.g. 0.2 for 20%%).")
    return p.parse_args()


def eta_to_pct(eta: float) -> int:
    """Convert 0.20 → 20 for filename/display."""
    return round(eta * 100)


def run_single(k: int, eta: float, cfg: dict) -> None:
    out_path = (
        Path(cfg["logging"]["results_dir"])
        / f"k{k}_eta{eta_to_pct(eta)}.csv"
    )
    if out_path.exists():
        print(f"[N1] k={k} η={eta_to_pct(eta)}% already done, skipping.")
        return

    print(f"[N1] Starting k={k} η={eta_to_pct(eta)}%")
    train_loader, test_loader = get_cifar10_loaders(
        noise_rate=eta,
        n_train=cfg["data"]["subset_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    model = build_resnet18(width_mult=k, num_classes=10)

    with CSVLogger(out_path, overwrite=False) as logger:
        trainer = Trainer(model, train_loader, test_loader, cfg["training"])
        trainer.run(logger=logger)

    print(f"[N1] Done k={k} η={eta_to_pct(eta)}%  →  {out_path}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ks = [args.k] if args.k is not None else cfg["model"]["width_multipliers"]
    etas = [args.eta] if args.eta is not None else cfg["noise_levels"]

    for k in ks:
        for eta in etas:
            run_single(k, eta, cfg)


if __name__ == "__main__":
    main()
