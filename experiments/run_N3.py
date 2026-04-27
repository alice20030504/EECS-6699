"""N3: Weight Decay Ablation.

Sweeps k ∈ {4,8,16} × λ ∈ {0, 1e-4, 5e-4, 1e-3, 5e-3} with fixed η=15%.
Each (k, λ) run saves to results/N3/k{k}_wd{wd_str}.csv.
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
    p.add_argument("--config", default="configs/N3.yaml")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--wd", type=float, default=None,
                   help="Single weight decay value to run.")
    return p.parse_args()


def wd_to_str(wd: float) -> str:
    """Convert 5e-4 → '5e-4' for filenames."""
    if wd == 0.0:
        return "0"
    return f"{wd:.0e}"


def run_single(k: int, wd: float, cfg: dict) -> None:
    out_path = (
        Path(cfg["logging"]["results_dir"])
        / f"k{k}_wd{wd_to_str(wd)}.csv"
    )
    if out_path.exists():
        print(f"[N3] k={k} wd={wd} already done, skipping.")
        return

    print(f"[N3] Starting k={k} wd={wd}")
    train_loader, test_loader = get_cifar10_loaders(
        noise_rate=cfg["data"]["label_noise"],
        n_train=5000,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    model = build_resnet18(width_mult=k, num_classes=10)

    # Inject weight decay into training config for this run
    run_cfg = {**cfg["training"], "weight_decay": wd}

    with CSVLogger(out_path, overwrite=False) as logger:
        trainer = Trainer(model, train_loader, test_loader, run_cfg)
        trainer.run(logger=logger)

    print(f"[N3] Done k={k} wd={wd}  →  {out_path}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ks = [args.k] if args.k is not None else cfg["model"]["width_multipliers"]
    wds = [args.wd] if args.wd is not None else cfg["weight_decays"]

    for k in ks:
        for wd in wds:
            run_single(k, wd, cfg)


if __name__ == "__main__":
    main()
