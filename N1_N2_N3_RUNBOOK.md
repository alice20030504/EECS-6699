# N1 / N2 / N3 Runbook

This file documents the canonical N1, N2, and N3 experiments used by the
project plan.

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
- Widths: `2, 4, 6, 8, 16`
- Weight decays: `0, 1e-4, 1e-3, 1e-2, 1e-1`
- Seed: `42`
- Total: `5 x 5 = 25` training runs

Run the full sweep:

```bash
python run_n2.py
```

Run a subset:

```bash
python run_n2.py --weight_decays 0 0.0001 0.001
python run_n2.py --widths 2
python run_n2.py --widths 6
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

Figure 6 keeps all five weight-decay curves for completeness, but visually
emphasizes the three most interpretable settings: `wd=0` as no explicit
regularization, `wd=1e-2` as the moderate setting that can flatten the peak,
and `wd=1e-1` as the over-regularized setting. The near-baseline settings
`1e-4` and `1e-3` are drawn as faded reference curves.

The `k=6` point is included because R1 identified it as the approximate
interpolation threshold at `eta=15%`, making it the most important location for
showing the double-descent peak and whether weight decay suppresses it.

## N3: Activation Function Comparison on Top of N1/N2

Goal: test whether smooth activations (GELU, Tanh) shift the position or height
of the interpolation peak relative to non-smooth ReLU. Under the NTK framework,
smooth activations have different spectral properties which may move the benign
overfitting boundary. N3 reuses the same CNN5/CIFAR-10 setup as N1/N2.

Canonical grid:

- Model: CNN5
- Dataset: CIFAR-10, fixed 5000-sample training subset
- Noise rate: `0.15` (same as N2)
- Widths: `4, 8, 16, 32`
- Activations: `relu`, `gelu`, `tanh`
- Seed: `42`
- Total: `4 x 3 = 12` training runs

Run the full sweep:

```bash
python run_n3.py
```

Run a subset:

```bash
python run_n3.py --activations relu gelu
python run_n3.py --widths 8 16
```

Regenerate plots from existing JSON results:

```bash
python run_n3.py --plot_only
```

Outputs:

- `results/N3/n3_*.json`
- `results/N3/n3_summary.csv`
- `results/N3/fig7_n3_activation.png`

Figure 7 plots three test-error-vs-width curves (one per activation) on a
log₂ x-axis. Key observations to record: peak position (which k), peak height
(maximum test error), and curve shape in the benign region (large k).
A null result (curves nearly identical) is also a valid finding.

## CSV single-point runners

Use these for one cell at a time on Colab/cluster:

```bash
python experiments/run_N1.py --k 16 --eta 0.2 --seed 42
python experiments/run_N2.py --k 8 --wd 0.001 --seed 42
python experiments/run_N3.py --k 8 --act gelu --seed 42
```
