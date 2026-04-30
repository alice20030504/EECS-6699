"""N3: Activation function comparison built on the N1/N2 phase-diagram setup.

This supplementary experiment asks whether smooth activations (GELU, Tanh) shift
the width/noise peak relative to the non-smooth ReLU. Under the Neural Tangent
Kernel (NTK) framework, smooth activations have different spectral properties
which can move the boundary of the benign overfitting region.

Setup mirrors N2: same CNN5/CIFAR-10 subset, fixed noise eta=15%, but now we
sweep activations (ReLU / GELU / Tanh) at widths spanning the interpolation
peak identified in N1.

Outputs
-------
results/N3/n3_*.json
results/N3/n3_summary.csv
results/N3/fig7_n3_activation.png

Usage
-----
python run_n3.py
python run_n3.py --plot_only
python run_n3.py --activations relu gelu
python run_n3.py --widths 4 8 16
python run_n3.py --drive
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_cifar10_loaders
from src.io_utils import get_result_dir, load_results, mount_drive_if_colab, save_result
from src.models import CNN5
from src.plot_utils import PALETTE, set_style
from src.train import train_one_run
from run_n1 import N1_CONFIG


N3_CONFIG = {
    "experiment": "N3",
    "parent_experiment": "N1",
    "rationale": (
        "N1/N2 follow-up: test whether smooth activations (GELU, Tanh) shift "
        "the double-descent peak vs non-smooth ReLU under label noise."
    ),
    "n_train": N1_CONFIG["n_train"],
    "noise_rate": 0.15,
    "data_seed": N1_CONFIG["data_seed"],
    "batch_size": N1_CONFIG["batch_size"],
    "n_classes": N1_CONFIG["n_classes"],
    "widths": [4, 8, 16, 32],
    "activations": ["relu", "gelu", "tanh"],
    "seeds": [42],
    "optimizer": N1_CONFIG["optimizer"],
    "lr": N1_CONFIG["lr"],
    "weight_decay": 0.0,
    "epochs": N1_CONFIG["epochs"],
    "checkpoint_every": N1_CONFIG["checkpoint_every"],
    "eval_every": N1_CONFIG["eval_every"],
}

# Colours and display names for each activation
_ACT_STYLE: dict[str, dict] = {
    "relu": {
        "color": PALETTE["catastrophic"],
        "label": "ReLU (non-smooth)",
        "linewidth": 2.6,
        "markersize": 6.5,
        "linestyle": "-",
        "alpha": 1.0,
        "zorder": 4,
    },
    "gelu": {
        "color": PALETTE["train"],
        "label": "GELU (smooth)",
        "linewidth": 2.6,
        "markersize": 6.5,
        "linestyle": "-",
        "alpha": 1.0,
        "zorder": 4,
    },
    "tanh": {
        "color": PALETTE["tempered"],
        "label": "Tanh (smooth)",
        "linewidth": 2.6,
        "markersize": 6.5,
        "linestyle": "--",
        "alpha": 1.0,
        "zorder": 3,
    },
}


def run_n3(
    cfg: dict,
    result_dir: str,
    widths: list[int] | None = None,
    activations: list[str] | None = None,
    resume: bool = True,
) -> list[dict]:
    import torch

    active_widths = widths if widths is not None else cfg["widths"]
    active_acts = activations if activations is not None else cfg["activations"]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 72)
    print(f"N3: Activation function comparison on N1 setup [device: {device_str}]")
    print(f"Parent setup:  {cfg.get('parent_experiment', 'N1')} (CNN5/CIFAR-10)")
    print(f"Fixed noise:   {cfg['noise_rate']:.0%}")
    print(f"Widths:        {active_widths}")
    print(f"Activations:   {active_acts}")
    print(f"Seeds:         {cfg['seeds']}")
    print(f"Total runs:    {len(active_widths) * len(active_acts) * len(cfg['seeds'])}")
    print(f"Result dir:    {result_dir}")
    print("=" * 72 + "\n")

    ckpt_dir = str(Path(result_dir) / "checkpoints")
    all_results: list[dict] = []

    for act in active_acts:
        for k in active_widths:
            for seed in cfg["seeds"]:
                run_id = f"n3_k{k:03d}_act{act}_s{seed}"
                result_path = Path(result_dir) / f"{run_id}.json"

                if resume and result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                    print(
                        f"[skip] {run_id:30s} | "
                        f"train_err={result['train_error']:.3f} "
                        f"test_err={result['test_error']:.3f}"
                    )
                    all_results.append(result)
                    continue

                print(f"\n[run ] {run_id} | k={k}, act={act}, seed={seed}")
                torch.manual_seed(seed)
                np.random.seed(seed)

                train_loader, test_loader = get_cifar10_loaders(
                    n_train=cfg["n_train"],
                    noise_rate=cfg["noise_rate"],
                    data_seed=cfg["data_seed"],
                    batch_size=cfg["batch_size"],
                    num_workers=cfg.get("num_workers", 2),
                )
                model = CNN5(
                    width_multiplier=k,
                    n_classes=cfg["n_classes"],
                    activation=act,
                )
                print(f"       params = {model.count_params():,}")

                result = train_one_run(
                    model,
                    train_loader,
                    test_loader,
                    optimizer_name=cfg["optimizer"],
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    epochs=cfg["epochs"],
                    checkpoint_dir=ckpt_dir,
                    checkpoint_every=cfg["checkpoint_every"],
                    eval_every=cfg["eval_every"],
                    run_id=run_id,
                )
                result.update(
                    {
                        "experiment": "N3",
                        "parent_experiment": cfg.get("parent_experiment", "N1"),
                        "run_id": run_id,
                        "width_multiplier": k,
                        "activation": act,
                        "seed": seed,
                        "noise_rate": cfg["noise_rate"],
                        "weight_decay": cfg["weight_decay"],
                        "config": cfg,
                    }
                )
                save_result(result, str(result_path))
                all_results.append(result)

    print(f"\nN3 complete: {len(all_results)} runs in {result_dir}")
    return all_results


def _aggregate(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in results:
        grouped[(str(r["activation"]), int(r["width_multiplier"]))].append(r)

    rows: list[dict] = []
    for (act, k), items in sorted(grouped.items()):
        rows.append(
            {
                "activation": act,
                "k": k,
                "n_runs": len(items),
                "train_error": float(np.mean([r["train_error"] for r in items])),
                "test_error": float(np.mean([r["test_error"] for r in items])),
                "train_error_min": float(np.min([r["train_error"] for r in items])),
                "test_error_min": float(np.min([r["test_error"] for r in items])),
                "test_error_max": float(np.max([r["test_error"] for r in items])),
            }
        )
    return rows


def _save_summary(rows: list[dict], result_dir: str) -> None:
    path = Path(result_dir) / "n3_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table] saved {path}")


def plot_n3(result_dir: str) -> None:
    results = load_results(result_dir, pattern="n3_*.json")
    if not results:
        print("[plot] No N3 results found; skipping.")
        return

    rows = _aggregate(results)
    _save_summary(rows, result_dir)
    _plot_activation_comparison(rows, result_dir)


def _plot_activation_comparison(rows: list[dict], result_dir: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    by_act: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_act[str(row["activation"])].append(row)

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    act_order = ["relu", "gelu", "tanh"]
    for act in act_order:
        if act not in by_act:
            continue
        act_rows = sorted(by_act[act], key=lambda r: r["k"])
        widths = [r["k"] for r in act_rows]
        test_errors = [r["test_error"] for r in act_rows]
        style = _ACT_STYLE.get(
            act,
            {
                "color": "0.55",
                "label": act,
                "linewidth": 1.6,
                "markersize": 5,
                "linestyle": "--",
                "alpha": 0.7,
                "zorder": 2,
            },
        )
        ax.plot(
            widths,
            test_errors,
            f"o{style['linestyle']}",
            color=style["color"],
            label=style["label"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    all_widths = sorted({r["k"] for r in rows})
    ax.set_xticks(all_widths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Width Multiplier k")
    ax.set_ylabel("Test Error")
    ax.set_title("Fig 7 - N3 Activation Comparison (eta=15%)")
    ax.legend(title="Activation", fontsize=9)
    plt.tight_layout()

    path = Path(result_dir) / "fig7_n3_activation.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N3: activation function comparison")
    parser.add_argument("--drive", action="store_true", help="Save results to Google Drive")
    parser.add_argument("--result_dir", default=None, help="Override result directory")
    parser.add_argument("--no_resume", action="store_true", help="Re-run all jobs")
    parser.add_argument("--plot_only", action="store_true", help="Skip training and regenerate plots")
    parser.add_argument("--widths", nargs="+", type=int, default=None)
    parser.add_argument("--activations", nargs="+", type=str, default=None,
                        choices=["relu", "gelu", "tanh"],
                        help="Subset of activations to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--n_train", type=int, default=None, help="Override subset size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader worker count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drive:
        mount_drive_if_colab()

    cfg = dict(N3_CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.n_train is not None:
        cfg["n_train"] = args.n_train
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    result_dir = args.result_dir or get_result_dir(".", "N3", use_drive=args.drive)
    if not args.plot_only:
        run_n3(
            cfg,
            result_dir,
            widths=args.widths,
            activations=args.activations,
            resume=not args.no_resume,
        )
    plot_n3(result_dir)


if __name__ == "__main__":
    main()
