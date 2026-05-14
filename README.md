# Benign Overfitting in Overparameterized Neural Networks
EECS 6699 — Spring 2026

We investigate the **benign overfitting** phenomenon in deep CNNs by reproducing model-wise double descent and mapping out a **(width × label-noise) phase diagram** on CIFAR-10.

---

## Team

| Member | Role | Experiments | Paper |
|--------|------|-------------|-------|
| **Yixuan Ye** | Project design & engineering | R1 · R2 · N1 (η = 0%, 5%) | Paper revision |
| **Shurong Zhang** | Main experiments & presentation | N1 (η = 10%, 20%) | PPT · Paper revision |
| **Yuxia Meng** | Theory & writing | Theory background · N2 | Sec 1–2 · Paper revision |
| **Rui Li** | Ablations & presentation | N3 · N1 (η = 30%, 40%) | PPT · Paper revision |

---

## Experiments

| ID | Description | Model | Runs | Script | Notebook |
|----|-------------|-------|------|--------|----------|
| **R1** | CNN5 double descent on CIFAR-10 (η=15%) | CNN5 | 22 | `run_r1.py` | `R1_CNN_DoubleDescent.ipynb` |
| **R2** | ResNet-18 double descent — validates R1 generalises across architectures | WideResNet18 | 6 | `run_r2.py` | `R2_ResNet_DoubleDescent.ipynb` |
| **N1** | (Width × Noise) 2D phase diagram — core contribution | CNN5 | 108 | `run_n1.py` | `N1_PhaseDiagram.ipynb` |
| **N2** | Weight-decay ablation — does L2 regularisation suppress the DD peak? | CNN5 | 25 | `run_n2.py` | — |
| **N3** | Activation comparison — ReLU vs GELU vs Tanh | CNN5 | 24 | `run_n3.py` | `N3_ActivationComparison.ipynb` |

### N1 Grid

k ∈ {1, 2, 3, 4, 6, 8, 16, 32, 64} × η ∈ {0%, 5%, 10%, 20%, 30%, 40%} × 2 seeds = **108 runs**

| Noise rates | Account | Owner |
|-------------|---------|-------|
| η = 0%, 5% | Account A | Ye |
| η = 10%, 20% | Account B | Alice |
| η = 30%, 40% | Account C | Lyric |

**Phase classification** — Test Error Gap Δ = TestErr(k, η) − TestErr(k, 0) (per-column baseline):

| Region | Threshold |
|--------|-----------|
| Benign | Δ < 5% |
| Tempered | 5% ≤ Δ < 15% |
| Catastrophic | Δ ≥ 15% |

**Benign boundary** — separate criterion: TestErr(k, η) < TestErr\_clean,\* + 15%.
After all runs complete, `run_n1.py --plot_only` fits **w\_benign(η) ≈ 320 · η^1.00** and saves `results/N1/benign_boundary_fit.json`.

---

## Repository Structure

```
EECS-6699/
├── src/
│   ├── data.py              # CIFAR-10 subset with symmetric label noise
│   ├── train.py             # Training loop (Adam/SGD, checkpointing, JSON logging)
│   ├── io_utils.py          # Result I/O + Google Drive helpers
│   ├── plot_utils.py        # Shared matplotlib style and colour palette
│   └── models/
│       ├── cnn.py           # CNN5 — 5-layer CNN, width multiplier k, swappable activation
│       └── resnet.py        # WideResNet18 — ResNet-18 with width multiplier k
├── run_r1.py                # R1 runner (22 runs, Adam, 300 ep)
├── run_r2.py                # R2 runner (6 runs, SGD+cosine, 200 ep)
├── run_n1.py                # N1 runner (108 runs, --noise_rates / --widths for parallelism)
├── run_n2.py                # N2 runner (25 runs, weight-decay sweep)
├── run_n3.py                # N3 runner (24 runs, activation comparison)
├── R1_CNN_DoubleDescent.ipynb
├── R2_ResNet_DoubleDescent.ipynb
├── N1_PhaseDiagram.ipynb
├── N3_ActivationComparison.ipynb
├── N1_N2_N3_RUNBOOK.md      # Detailed runbook for N1, N2, N3
├── results/                 # Auto-created; one JSON per run + figures
└── requirements.txt
```

---

## Quickstart

**Local**
```bash
pip install -r requirements.txt
python run_r1.py
python run_r2.py
python run_n1.py --noise_rates 0.10 0.20   # one noise split
python run_n2.py
python run_n3.py
```

**Colab (recommended)**

N1 is parallelised across 3 accounts:
1. Open `N1_PhaseDiagram.ipynb` → Runtime → T4 GPU
2. Set your GitHub PAT in cell 2
3. In cell 3, uncomment your `MY_NOISE_RATES` line
4. Run all cells — results stream to shared Google Drive folder and resume on disconnect

N3 runs on a single account (~2 h on T4):
1. Open `N3_ActivationComparison.ipynb` → Runtime → T4 GPU
2. Set PAT → run all cells

All runners skip completed JSON files automatically (`resume=True` default).

---

## Result Schema

Every experiment writes one JSON per run:

```json
{
  "experiment":       "N1",
  "run_id":           "n1_k016_eta020_s42",
  "width_multiplier": 16,
  "noise_rate":       0.20,
  "seed":             42,
  "train_error":      0.004,
  "test_error":       0.387,
  "n_params":         152834,
  "epochs":           300,
  "wall_time_s":      1820,
  "history":          [...]
}
```

Load with `src.io_utils.load_results(result_dir)`.
N2 adds a `weight_decay` field; N3 adds an `activation` field.

---

## Key Findings

| Experiment | Finding |
|-----------|---------|
| R1 | Textbook DD curve: interpolation threshold at k=6 (22K params), peak at k=4 |
| R2 | Same phenomenon in ResNet-18; threshold at k=2 due to faster parameter scaling |
| N1 | Phase diagram: benign region at low η + large k; catastrophic at high η + small k |
| N1 | Empirical boundary: w\_benign(η) ≈ 320·η^1.00 (linear scaling, η ∈ {5%,10%,20%}) |
| N2 | wd=1e-2 reduces peak at k=4 from 63% to 56.5%; wd=1e-1 overshoots and hurts small k |
| N3 | ReLU ≈ GELU throughout; Tanh has higher peak (64.5% vs 62–63%) due to saturation |

---

## References

- Nakkiran et al. (2020) *Deep Double Descent* — ICLR
- Bartlett et al. (2020) *Benign Overfitting in Linear Regression* — PNAS
- Mallinar et al. (2022) *Benign, Tempered, or Catastrophic* — NeurIPS
- Belkin et al. (2019) *Reconciling Modern ML and Bias-Variance* — PNAS
- Frei et al. (2022) *Benign Overfitting without Linearity* — COLT
