"""N1: Width x Noise phase diagram for benign overfitting.

This is the main new experiment in the project plan. It sweeps CNN width and
label-noise level on a fixed CIFAR-10 subset, then classifies each point as:

    benign, tempered, or catastrophic

The test-error gap follows the project plan:

    gap(k, eta) = TestErr(k, eta) - TestErr(k*, 0)

where k* is the best width under zero label noise.

Outputs
-------
results/N1/n1_*.json
results/N1/fig3_n1_heatmap.png
results/N1/fig4_n1_dd_overlay.png
results/N1/fig5_n1_phase_diagram.png
results/N1/benign_boundary_fit.json
results/N1/n1_phase_table.csv

Usage
-----
python run_n1.py
python run_n1.py --noise_rates 0.0 0.05
python run_n1.py --plot_only
python run_n1.py --drive
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


N1_CONFIG = {
    "experiment": "N1",
    "n_train": 5000,
    "data_seed": 42,
    "batch_size": 128,
    "activation": "relu",
    "n_classes": 10,
    "widths": [2, 4, 8, 16, 32, 64],
    "noise_rates": [0.0, 0.05, 0.10, 0.20, 0.40],
    "seeds": [42, 123],
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 300,
    "checkpoint_every": 10,
    "eval_every": 10,
    "benign_thresh": 0.03,
    "tempered_thresh": 0.10,
}


def _eta_tag(eta: float) -> str:
    return f"{round(eta * 100):03d}"


def run_n1(
    cfg: dict,
    result_dir: str,
    noise_rates: list[float] | None = None,
    widths: list[int] | None = None,
    resume: bool = True,
) -> list[dict]:
    import torch

    active_noises = noise_rates if noise_rates is not None else cfg["noise_rates"]
    active_widths = widths if widths is not None else cfg["widths"]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 72)
    print(f"N1: Width x Noise phase diagram [device: {device_str}]")
    print(f"Widths:      {active_widths}")
    print(f"Noise rates: {[f'{eta:.0%}' for eta in active_noises]}")
    print(f"Seeds:       {cfg['seeds']}")
    print(f"Total runs:  {len(active_widths) * len(active_noises) * len(cfg['seeds'])}")
    print(f"Result dir:  {result_dir}")
    print("=" * 72 + "\n")

    ckpt_dir = str(Path(result_dir) / "checkpoints")
    all_results: list[dict] = []

    for eta in active_noises:
        for k in active_widths:
            for seed in cfg["seeds"]:
                run_id = f"n1_k{k:03d}_eta{_eta_tag(eta)}_s{seed}"
                result_path = Path(result_dir) / f"{run_id}.json"

                if resume and result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                    print(
                        f"[skip] {run_id:25s} | "
                        f"train_err={result['train_error']:.3f} "
                        f"test_err={result['test_error']:.3f}"
                    )
                    all_results.append(result)
                    continue

                print(f"\n[run ] {run_id} | k={k}, eta={eta:.0%}, seed={seed}")
                torch.manual_seed(seed)
                np.random.seed(seed)

                train_loader, test_loader = get_cifar10_loaders(
                    n_train=cfg["n_train"],
                    noise_rate=eta,
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
                    weight_decay=cfg["weight_decay"],
                    epochs=cfg["epochs"],
                    checkpoint_dir=ckpt_dir,
                    checkpoint_every=cfg["checkpoint_every"],
                    eval_every=cfg["eval_every"],
                    run_id=run_id,
                )
                result.update(
                    {
                        "experiment": "N1",
                        "run_id": run_id,
                        "width_multiplier": k,
                        "noise_rate": eta,
                        "seed": seed,
                        "config": cfg,
                    }
                )
                save_result(result, str(result_path))
                all_results.append(result)

    print(f"\nN1 batch complete: {len(all_results)} runs in {result_dir}")
    return all_results


def _aggregate(results: list[dict]) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    widths = sorted({int(r["width_multiplier"]) for r in results})
    noises = sorted({float(r["noise_rate"]) for r in results})
    test_values: dict[tuple[float, int], list[float]] = defaultdict(list)
    train_values: dict[tuple[float, int], list[float]] = defaultdict(list)

    for r in results:
        key = (float(r["noise_rate"]), int(r["width_multiplier"]))
        test_values[key].append(float(r["test_error"]))
        train_values[key].append(float(r["train_error"]))

    test_matrix = np.full((len(noises), len(widths)), np.nan)
    train_matrix = np.full((len(noises), len(widths)), np.nan)
    for ni, eta in enumerate(noises):
        for wi, k in enumerate(widths):
            key = (eta, k)
            if test_values[key]:
                test_matrix[ni, wi] = float(np.mean(test_values[key]))
                train_matrix[ni, wi] = float(np.mean(train_values[key]))

    return widths, noises, train_matrix, test_matrix


def _phase_for_point(
    gap: float,
    benign_thresh: float,
    tempered_thresh: float,
) -> str:
    if np.isnan(gap):
        return "missing"
    if gap < benign_thresh:
        return "benign"
    if gap < tempered_thresh:
        return "tempered"
    return "catastrophic"


def _build_phase_table(
    widths: list[int],
    noises: list[float],
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    cfg: dict,
) -> list[dict]:
    if 0.0 not in noises:
        raise ValueError("N1 phase classification requires eta=0 baseline runs.")

    eta0_idx = noises.index(0.0)
    clean_row = test_matrix[eta0_idx]
    baseline_error = float(np.nanmin(clean_row))
    baseline_width = int(widths[int(np.nanargmin(clean_row))])

    rows: list[dict] = []
    for ni, eta in enumerate(noises):
        for wi, k in enumerate(widths):
            train_error = float(train_matrix[ni, wi])
            test_error = float(test_matrix[ni, wi])
            gap = test_error - baseline_error
            phase = _phase_for_point(
                gap,
                cfg["benign_thresh"],
                cfg["tempered_thresh"],
            )
            rows.append(
                {
                    "k": k,
                    "eta": eta,
                    "train_error": train_error,
                    "test_error": test_error,
                    "baseline_width": baseline_width,
                    "baseline_error": baseline_error,
                    "gap": float(gap),
                    "phase": phase,
                }
            )
    return rows


def _save_phase_table(rows: list[dict], result_dir: str) -> None:
    path = Path(result_dir) / "n1_phase_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table] saved {path}")


def plot_n1(result_dir: str, cfg: dict | None = None) -> None:
    cfg = cfg or N1_CONFIG
    results = load_results(result_dir, pattern="n1_*.json")
    if not results:
        print("[plot] No N1 results found; skipping.")
        return

    widths, noises, train_matrix, test_matrix = _aggregate(results)
    phase_rows = _build_phase_table(widths, noises, train_matrix, test_matrix, cfg)
    _save_phase_table(phase_rows, result_dir)

    _plot_heatmap(widths, noises, test_matrix, result_dir)
    _plot_dd_overlay(widths, noises, test_matrix, result_dir)
    _plot_phase_diagram(widths, noises, phase_rows, cfg, result_dir)
    _fit_benign_boundary(widths, noises, phase_rows, result_dir)


def _plot_heatmap(widths: list[int], noises: list[float], test_matrix: np.ndarray, result_dir: str) -> None:
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(
        test_matrix * 100,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn_r",
        extent=[-0.5, len(widths) - 0.5, -0.5, len(noises) - 0.5],
    )

    for ni in range(len(noises)):
        for wi in range(len(widths)):
            val = test_matrix[ni, wi]
            if not np.isnan(val):
                ax.text(wi, ni, f"{val * 100:.1f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([f"{eta:.0%}" for eta in noises])
    ax.set_xlabel("Width Multiplier k")
    ax.set_ylabel("Label Noise Rate eta")
    ax.set_title("Fig 3 - N1 Test Error Heatmap")
    plt.colorbar(im, ax=ax, label="Test Error (%)")
    plt.tight_layout()

    path = Path(result_dir) / "fig3_n1_heatmap.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path}")


def _plot_dd_overlay(widths: list[int], noises: list[float], test_matrix: np.ndarray, result_dir: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [plt.cm.plasma(i / max(len(noises) - 1, 1)) for i in range(len(noises))]

    for ni, (eta, color) in enumerate(zip(noises, colors)):
        row = test_matrix[ni]
        valid = ~np.isnan(row)
        if valid.any():
            ax.plot(np.array(widths)[valid], row[valid], "o-", color=color, label=f"eta={eta:.0%}")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(widths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Width Multiplier k")
    ax.set_ylabel("Test Error")
    ax.set_title("Fig 4 - Double Descent Curves by Noise Level")
    ax.legend(title="Noise Rate", loc="upper right")
    plt.tight_layout()

    path = Path(result_dir) / "fig4_n1_dd_overlay.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path}")


def _plot_phase_diagram(
    widths: list[int],
    noises: list[float],
    phase_rows: list[dict],
    cfg: dict,
    result_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    colors = {
        "benign": PALETTE["benign"],
        "tempered": PALETTE["tempered"],
        "catastrophic": PALETTE["catastrophic"],
        "missing": "#FFFFFF",
    }
    labels = {
        "benign": f"Benign (gap < {cfg['benign_thresh'] * 100:.0f}%)",
        "tempered": (
            f"Tempered ({cfg['benign_thresh'] * 100:.0f}%"
            f"-{cfg['tempered_thresh'] * 100:.0f}%)"
        ),
        "catastrophic": f"Catastrophic (gap >= {cfg['tempered_thresh'] * 100:.0f}%)",
    }

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in phase_rows:
        phase = row["phase"]
        if phase == "missing":
            continue
        ax.scatter(
            row["k"],
            row["eta"],
            c=colors[phase],
            s=220,
            marker="s",
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            row["k"],
            row["eta"],
            f"{row['test_error'] * 100:.0f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(widths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Width Multiplier k")
    ax.set_ylabel("Label Noise Rate eta")
    ax.set_title("Fig 5 - N1 Overfitting Phase Diagram")
    ax.legend(
        handles=[Patch(facecolor=colors[p], label=labels[p]) for p in labels],
        loc="upper right",
        fontsize=8,
    )
    plt.tight_layout()

    path = Path(result_dir) / "fig5_n1_phase_diagram.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path}")


def _fit_benign_boundary(widths: list[int], noises: list[float], phase_rows: list[dict], result_dir: str) -> None:
    by_eta: dict[float, list[int]] = defaultdict(list)
    for row in phase_rows:
        if row["eta"] > 0 and row["phase"] == "benign":
            by_eta[float(row["eta"])].append(int(row["k"]))

    boundary = {eta: min(ks) for eta, ks in by_eta.items() if ks}
    if len(boundary) < 2:
        print("[boundary] Not enough benign points for a power-law fit.")
        return

    etas = np.array(sorted(boundary.keys()))
    ks = np.array([boundary[eta] for eta in etas])
    alpha, log_c = np.polyfit(np.log(etas), np.log(ks), 1)
    c = float(np.exp(log_c))

    fit_result = {
        "formula": f"w_benign = {c:.4f} * eta^{alpha:.4f}",
        "c": c,
        "alpha": float(alpha),
        "data": {f"{eta:.2f}": int(k) for eta, k in zip(etas, ks)},
    }
    path = Path(result_dir) / "benign_boundary_fit.json"
    with open(path, "w") as f:
        json.dump(fit_result, f, indent=2)
    print(f"[boundary] saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N1: Width x Noise phase diagram")
    parser.add_argument("--drive", action="store_true", help="Save results to Google Drive")
    parser.add_argument("--result_dir", default=None, help="Override result directory")
    parser.add_argument("--no_resume", action="store_true", help="Re-run all jobs")
    parser.add_argument("--plot_only", action="store_true", help="Skip training and regenerate plots")
    parser.add_argument("--noise_rates", nargs="+", type=float, default=None)
    parser.add_argument("--widths", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs, useful for smoke tests")
    parser.add_argument("--n_train", type=int, default=None, help="Override subset size, useful for smoke tests")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader worker count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drive:
        mount_drive_if_colab()

    cfg = dict(N1_CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.n_train is not None:
        cfg["n_train"] = args.n_train
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    result_dir = args.result_dir or get_result_dir(".", "N1", use_drive=args.drive)
    if not args.plot_only:
        run_n1(
            cfg,
            result_dir,
            noise_rates=args.noise_rates,
            widths=args.widths,
            resume=not args.no_resume,
        )
    plot_n1(result_dir, cfg)


if __name__ == "__main__":
    main()
