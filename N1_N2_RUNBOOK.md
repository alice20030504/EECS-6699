# N1/N2 Runbook

This file documents the canonical N1 and N2 experiments used by the project
plan.

## N1: Width x Noise Phase Diagram

Goal: map when interpolation is benign, tempered, or catastrophic over the
plane of model width and label-noise level.

Canonical grid:

- Model: CNN5
- Dataset: CIFAR-10, fixed 5000-sample training subset
- Widths: `2, 4, 8, 16, 32, 64`
- Noise rates: `0, 0.05, 0.10, 0.20, 0.40`
- Seeds: `42, 123`
- Total: `6 x 5 x 2 = 60` training runs

Run the full sweep:

```bash
python run_n1.py
```

Run a subset, useful for parallel Colab accounts:

```bash
python run_n1.py --noise_rates 0.0 0.05
python run_n1.py --noise_rates 0.10 0.20
python run_n1.py --noise_rates 0.40
```

Regenerate plots from existing JSON results:

```bash
python run_n1.py --plot_only
```

Outputs:

- `results/N1/n1_*.json`
- `results/N1/n1_phase_table.csv`
- `results/N1/fig3_n1_heatmap.png`
- `results/N1/fig4_n1_dd_overlay.png`
- `results/N1/fig5_n1_phase_diagram.png`
- `results/N1/benign_boundary_fit.json` when enough benign points exist

Phase rule:

- Baseline: `gap(k, eta) = TestErr(k, eta) - TestErr(k*, 0)`
- `k*`: the width with the lowest test error at `eta = 0`
- `benign`: `gap < 0.03`
- `tempered`: `0.03 <= gap < 0.10`
- `catastrophic`: `gap >= 0.10`

## N2: Weight Decay Ablation on Top of N1

Goal: use N1's setup as the base experiment, then test whether explicit L2
regularization reduces the double-descent peak. N2 is therefore a mechanism
follow-up: N1 maps the phase diagram, and N2 asks whether regularization can
shift or flatten the peak region.

Canonical grid:

- Model: CNN5
- Dataset: CIFAR-10, fixed 5000-sample training subset
- Noise rate: `0.15`
- Widths: `4, 8, 16`
- Weight decays: `0, 1e-4, 1e-3, 1e-2, 1e-1`
- Seed: `42`
- Total: `3 x 5 = 15` training runs

Run the full sweep:

```bash
python run_n2.py
```

Run a subset:

```bash
python run_n2.py --weight_decays 0 0.0001 0.001
python run_n2.py --widths 8 16
```

Regenerate plots from existing JSON results:

```bash
python run_n2.py --plot_only
```

Outputs:

- `results/N2/n2_*.json`
- `results/N2/n2_summary.csv`
- `results/N2/fig6_n2_weight_decay.png`

## CSV single-point runners

Use these for one cell at a time on Colab/cluster:

```bash
python experiments/run_N1.py --k 16 --eta 0.2 --seed 42
python experiments/run_N2.py --k 8 --wd 0.001 --seed 42
```
