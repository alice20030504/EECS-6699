"""Single-point runner for N2: N1-based weight decay ablation.

N2 is a follow-up to N1. It reuses the same CNN5/CIFAR-10 setup, fixes
eta=15%, and sweeps weight decay near the N1/R1 peak region.

This CSV runner is useful for Colab/cluster parallelism:

    python experiments/run_N2.py --k 8 --wd 0.001 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_cifar10_loaders
from src.models import CNN5
from src.training import Trainer
from src.utils import CSVLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/N2.yaml")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--wd", type=float, default=None, help="Single weight decay value.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def wd_to_str(wd: float) -> str:
    if wd == 0:
        return "0"
    return f"{wd:.0e}".replace("-", "m").replace("+", "p")


def run_single(k: int, wd: float, seed: int, cfg: dict) -> None:
    out_path = (
        ROOT
        / cfg["logging"]["results_dir"]
        / f"k{k}_wd{wd_to_str(wd)}_s{seed}.csv"
    )
    if out_path.exists():
        print(f"[N2] k={k} wd={wd:g} seed={seed} already done, skipping.")
        return

    print(f"[N2] Starting k={k} wd={wd:g} seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_cifar10_loaders(
        noise_rate=cfg["data"]["label_noise"],
        n_train=cfg["data"]["subset_size"],
        data_seed=cfg["data"].get("data_seed", 42),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    model = CNN5(
        width_multiplier=k,
        n_classes=10,
        activation=cfg["model"].get("activation", "relu"),
    )

    run_cfg = {**cfg["training"], "weight_decay": wd}
    with CSVLogger(out_path, overwrite=False) as logger:
        trainer = Trainer(model, train_loader, test_loader, run_cfg)
        trainer.run(logger=logger)

    print(f"[N2] Done k={k} wd={wd:g} seed={seed} -> {out_path}")


def main() -> None:
    args = parse_args()
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    ks = [args.k] if args.k is not None else cfg["model"]["width_multipliers"]
    wds = [args.wd] if args.wd is not None else cfg["weight_decays"]
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42])

    for wd in wds:
        for k in ks:
            for seed in seeds:
                run_single(k, wd, seed, cfg)


if __name__ == "__main__":
    main()
