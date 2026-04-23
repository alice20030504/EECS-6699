"""R2a: Model-wise Double Descent — reproduce Nakkiran (2020).

Sweeps width_mult k over cfg['model']['width_multipliers'] with fixed label noise.
Each (k) run saves to results/R2a/k{k}.csv.
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
    p.add_argument("--config", default="configs/R2a.yaml")
    p.add_argument("--k", type=int, default=None,
                   help="Run a single width multiplier (useful for parallelising on cluster).")
    return p.parse_args()


def run_single(k: int, cfg: dict) -> None:
    out_path = Path(cfg["logging"]["results_dir"]) / f"k{k}.csv"
    if out_path.exists():
        print(f"[R2a] k={k} already done, skipping.")
        return

    print(f"[R2a] Starting k={k}")
    train_loader, test_loader = get_cifar10_loaders(
        noise_eta=cfg["data"]["label_noise"],
        subset_size=None,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    model = build_resnet18(width_mult=k, num_classes=10)

    with CSVLogger(out_path, overwrite=False) as logger:
        trainer = Trainer(model, train_loader, test_loader, cfg["training"])
        trainer.run(logger=logger)

    print(f"[R2a] Done k={k}  →  {out_path}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ks = [args.k] if args.k is not None else cfg["model"]["width_multipliers"]
    for k in ks:
        run_single(k, cfg)


if __name__ == "__main__":
    main()
