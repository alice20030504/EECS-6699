"""N2: Weight decay ablation built on the N1 phase-diagram setup.

This supplementary experiment asks whether the double-descent peak is reduced
or removed by explicit L2 regularization. It is a follow-up to N1: we reuse the
same CNN5/CIFAR-10 subset/training setup, fix a representative noise level
near the double-descent setting (eta=15%), and sweep weight decay over widths
spanning the transition into the N1/R1 interpolation peak.

Outputs
-------
results/N2/n2_*.json
results/N2/n2_summary.csv
results/N2/fig6_n2_weight_decay.png

Usage
-----
python run_n2.py
python run_n2.py --plot_only
python run_n2.py --weight_decays 0 0.0001 0.001
python run_n2.py --drive
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


N2_CONFIG = {
    "experiment": "N2",
    "parent_experiment": "N1",
    "rationale": "N1 follow-up: test whether weight decay suppresses the width/noise peak.",
    "n_train": N1_CONFIG["n_train"],
    "noise_rate": 0.15,
    "data_seed": N1_CONFIG["data_seed"],
    "batch_size": N1_CONFIG["batch_size"],
    "activation": N1_CONFIG["activation"],
    "n_classes": N1_CONFIG["n_classes"],
    "widths": [2, 4, 6, 8, 16],
    "weight_decays": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    "seeds": [42],
    "optimizer": N1_CONFIG["optimizer"],
    "lr": N1_CONFIG["lr"],
    "epochs": N1_CONFIG["epochs"],
    "checkpoint_every": N1_CONFIG["checkpoint_every"],
    "eval_every": N1_CONFIG["eval_every"],
}


def _wd_tag(weight_decay: float) -> str:
    if weight_decay == 0:
        return "0"
    return f"{weight_decay:.0e}".replace("-", "m").replace("+", "p")


def run_n2(
    cfg: dict,
    result_dir: str,
    widths: list[int] | None = None,
    weight_decays: list[float] | None = None,
    resume: bool = True,
) -> list[dict]:
    import torch

    active_widths = widths if widths is not None else cfg["widths"]
    active_wds = weight_decays if weight_decays is not None else cfg["weight_decays"]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 72)
    print(f"N2: Weight decay ablation on N1 setup [device: {device_str}]")
    print(f"Parent setup:  {cfg.get('parent_experiment', 'N1')} (CNN5/CIFAR-10)")
    print(f"Fixed noise:   {cfg['noise_rate']:.0%}")
    print(f"Widths:        {active_widths}")
    print(f"Weight decay:  {active_wds}")
    print(f"Seeds:         {cfg['seeds']}")
    print(f"Total runs:    {len(active_widths) * len(active_wds) * len(cfg['seeds'])}")
    print(f"Result dir:    {result_dir}")
    print("=" * 72 + "\n")

    ckpt_dir = str(Path(result_dir) / "checkpoints")
    all_results: list[dict] = []

    for wd in active_wds:
        for k in active_widths:
            for seed in cfg["seeds"]:
                run_id = f"n2_k{k:03d}_wd{_wd_tag(wd)}_s{seed}"
                result_path = Path(result_dir) / f"{run_id}.json"

                if resume and result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                    print(
                        f"[skip] {run_id:24s} | "
                        f"train_err={result['train_error']:.3f} "
                        f"test_err={result['test_error']:.3f}"
                    )
                    all_results.append(result)
                    continue

                print(f"\n[run ] {run_id} | k={k}, wd={wd:g}, seed={seed}")
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
                    activation=cfg["activation"],
                )
                print(f"       params = {model.count_params():,}")

                result = train_one_run(
                    model,
                    train_loader,
                    test_loader,
                    optimizer_name=cfg["optimizer"],
                    lr=cfg["lr"],
                    weight_decay=wd,
                    epochs=cfg["epochs"],
                    checkpoint_dir=ckpt_dir,
                    checkpoint_every=cfg["checkpoint_every"],
                    eval_every=cfg["eval_every"],
                    run_id=run_id,
                )
                result.update(
                    {
                        "experiment": "N2",
                        "parent_experiment": cfg.get("parent_experiment", "N1"),
                        "run_id": run_id,
                        "width_multiplier": k,
                        "seed": seed,
                        "noise_rate": cfg["noise_rate"],
                        "weight_decay": wd,
                        "config": cfg,
                    }
                )
                save_result(result, str(result_path))
                all_results.append(result)

    print(f"\nN2 complete: {len(all_results)} runs in {result_dir}")
    return all_results


def _aggregate(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for r in results:
        grouped[(float(r["weight_decay"]), int(r["width_multiplier"]))].append(r)

    rows: list[dict] = []
    for (wd, k), items in sorted(grouped.items()):
        rows.append(
            {
                "weight_decay": wd,
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
    path = Path(result_dir) / "n2_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table] saved {path}")


def plot_n2(result_dir: str) -> None:
    results = load_results(result_dir, pattern="n2_*.json")
    if not results:
        print("[plot] No N2 results found; skipping.")
        return

    rows = _aggregate(results)
    _save_summary(rows, result_dir)
    _plot_weight_decay(rows, result_dir)


def _plot_weight_decay(rows: list[dict], result_dir: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    by_wd: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_wd[float(row["weight_decay"])].append(row)

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    wds = sorted(by_wd)
    emphasis = {
        0.0: {
            "color": PALETTE["catastrophic"],
            "label": "wd=0 (no regularization)",
            "linewidth": 2.8,
            "markersize": 6.5,
            "alpha": 1.0,
            "zorder": 4,
        },
        1e-2: {
            "color": PALETTE["train"],
            "label": "wd=1e-2 (moderate)",
            "linewidth": 2.8,
            "markersize": 6.5,
            "alpha": 1.0,
            "zorder": 4,
        },
        1e-1: {
            "color": PALETTE["tempered"],
            "label": "wd=1e-1 (strong)",
            "linewidth": 2.8,
            "markersize": 6.5,
            "alpha": 1.0,
            "zorder": 4,
        },
    }

    for wd in wds:
        wd_rows = sorted(by_wd[wd], key=lambda r: r["k"])
        widths = [r["k"] for r in wd_rows]
        test_errors = [r["test_error"] for r in wd_rows]
        style = emphasis.get(
            wd,
            {
                "color": "0.55",
                "label": f"wd={wd:.0e} (minor)",
                "linewidth": 1.4,
                "markersize": 4.5,
                "alpha": 0.45,
                "zorder": 2,
            },
        )
        ax.plot(
            widths,
            test_errors,
            "o--" if wd not in emphasis else "o-",
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
    ax.set_xticks(sorted({r["k"] for r in rows}))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Width Multiplier k")
    ax.set_ylabel("Test Error")
    ax.set_title("Fig 6 - N2 Weight Decay Follow-up on N1 (eta=15%)")
    ax.legend(title="Weight Decay", fontsize=8)
    plt.tight_layout()

    path = Path(result_dir) / "fig6_n2_weight_decay.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N2: weight decay ablation")
    parser.add_argument("--drive", action="store_true", help="Save results to Google Drive")
    parser.add_argument("--result_dir", default=None, help="Override result directory")
    parser.add_argument("--no_resume", action="store_true", help="Re-run all jobs")
    parser.add_argument("--plot_only", action="store_true", help="Skip training and regenerate plots")
    parser.add_argument("--widths", nargs="+", type=int, default=None)
    parser.add_argument("--weight_decays", nargs="+", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs, useful for smoke tests")
    parser.add_argument("--n_train", type=int, default=None, help="Override subset size, useful for smoke tests")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader worker count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drive:
        mount_drive_if_colab()

    cfg = dict(N2_CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.n_train is not None:
        cfg["n_train"] = args.n_train
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    result_dir = args.result_dir or get_result_dir(".", "N2", use_drive=args.drive)
    if not args.plot_only:
        run_n2(
            cfg,
            result_dir,
            widths=args.widths,
            weight_decays=args.weight_decays,
            resume=not args.no_resume,
        )
    plot_n2(result_dir)


if __name__ == "__main__":
    main()
