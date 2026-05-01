# Benign Overfitting in Overparameterized Neural Networks
EECS 6699 — Spring 2026

We investigate the **benign overfitting** phenomenon in deep CNNs by mapping out a **(width × label-noise)** phase diagram on CIFAR-10 and reproducing model-wise double descent from Nakkiran et al. (2020).

---

## Team

| Member | Role | Experiments | Paper |
|--------|------|-------------|-------|
| **Meng (A)** | Theory & Writing | Literature review | Sec 1, 2, 5 (~45%) |
| **Ye (B)** | Engineering & Reproduction | R1 · R2 · N1 Account A: η ∈ {0%, 5%} | Sec 3.1 (~15%) |
| **Alice (C)** | Main Experiments | N1 Account B: η ∈ {10%, 20%} | Sec 4.1 — Phase Diagram (~30%) |
| **Lyric (D)** | Visualization & Coordination | N2 · N3 · N1 Account C: η ∈ {30%, 40%} | Sec 4.2, 4.3, all figures (~10%) |

---

## Experiments

| ID | Description | Model | Runs | Script | Notebook |
|----|-------------|-------|------|--------|----------|
| **R1** | CNN5 model-wise double descent on CIFAR-10 (η=15%) | CNN5 | 22 | `run_r1.py` | `R1_CNN_DoubleDescent.ipynb` |
| **R2** | ResNet-18 double descent — validates R1 generalises to deeper arch | WideResNet18 | 6 | `run_r2.py` | `R2_ResNet_DoubleDescent.ipynb` |
| **N1** | (Width × Noise) 2D phase diagram — core contribution | CNN5 | 72 | `run_n1.py` | `N1_PhaseDiagram.ipynb` |
| **N2** | Weight-decay ablation — does L2 regularisation eliminate the DD peak? | CNN5 | 20 | `run_n2.py` | — |
| **N3** | Activation comparison — do smooth activations (GELU/Tanh) shift the peak? | CNN5 | 24 | `run_n3.py` | `N3_ActivationComparison.ipynb` |

### N1 Grid

k ∈ {2, 4, 8, 16, 32, 64} × η ∈ {0%, 5%, 10%, 20%, 30%, 40%} × 2 seeds = **72 runs**

| Noise rates | Colab account | Owner |
|-------------|---------------|-------|
| η = 0%, 5% | Account A | Ye |
| η = 10%, 20% | Account B | Alice |
| η = 30%, 40% | Account C | Lyric |

**Phase classification** — Test Error Gap Δ = TestErr(η, k) − TestErr(0, k):
benign Δ < 3% · tempered 3–10% · catastrophic ≥ 10%

After all 72 runs complete, `run_n1.py --plot_only` fits the empirical boundary
**w\_benign ≈ c · η^α** and writes `results/N1/benign_boundary_fit.json`.

---

## Repository

```
EECS-6699/
├── docs/                    # Project plan + course guidelines
├── src/
│   ├── data.py              # CIFAR-10 subset with symmetric label noise
│   ├── train.py             # Training loop (Adam/SGD, checkpointing, JSON logging)
│   ├── io_utils.py          # Result I/O + Google Drive helpers
│   ├── plot_utils.py        # Shared matplotlib style and colour palette
│   └── models/
│       ├── cnn.py           # CNN5 — 5-layer CNN, width multiplier k, activation-swappable
│       └── resnet.py        # WideResNet18 — ResNet-18, width multiplier k
├── run_r1.py                # R1 runner
├── R1_CNN_DoubleDescent.ipynb       # Colab notebook for R1
├── run_r2.py                # R2 runner
├── R2_ResNet_DoubleDescent.ipynb    # Colab notebook for R2
├── run_n1.py                # N1 runner (supports --noise_rates for parallelism)
├── N1_PhaseDiagram.ipynb            # Colab notebook for N1
├── run_n2.py                # N2 runner (weight-decay ablation)
├── run_n3.py                # N3 runner (activation comparison)
├── N3_ActivationComparison.ipynb    # Colab notebook for N3
├── N1_N2_N3_RUNBOOK.md      # Detailed runbook for N1, N2, N3
├── results/                 # Auto-created; JSON + figures per experiment
│   ├── R1/   R2/   N1/   N2/   N3/
└── requirements.txt
```

---

## Quickstart

**Local**
```bash
pip install -r requirements.txt
python run_r1.py
python run_r2.py
python run_n1.py --noise_rates 0.10 0.20   # Alice's portion
```

**Colab (recommended — GPU required)**

*N1 (parallelised across 3 accounts):*
1. Open `N1_PhaseDiagram.ipynb` → Runtime → GPU (T4)
2. Set your GitHub PAT in cell 2
3. In cell 3, uncomment your account's `MY_NOISE_RATES` line
4. Run all cells — results stream to Google Drive and resume automatically

*N3 (single account, ~2 h):*
1. Open `N3_ActivationComparison.ipynb` → Runtime → GPU (T4)
2. Set your GitHub PAT in cell 2
3. Run all cells — 12 runs complete sequentially, results saved to Drive

Results are written as JSON to `results/<EXP>/`. Completed runs are skipped on re-run.

---

## Result Schema

All experiments share a common JSON format:

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
N2 adds a `weight_decay` field. N3 adds an `activation` field.

---

## Paper Outline

| Section | Pages | Owner |
|---------|-------|-------|
| 1. Introduction | 1.5 | Meng |
| 2. Problem Description | 3.0 | Meng |
| 3. Reproduction (R1/R2) | 2.5 | Ye |
| 4.1 Phase Diagram (N1) | 3.5 | Alice |
| 4.2 Weight Decay (N2) | 1.5 | Lyric |
| 4.3 Activation Comparison (N3) | 2.0 | Lyric |
| 5. Discussion & Conclusion | 2.0 | Meng |
| References & Appendix | 0.5 | All |

---

## References

- Nakkiran et al. (2020) *Deep Double Descent* — ICLR
- Bartlett et al. (2020) *Benign Overfitting in Linear Regression* — PNAS
- Mallinar et al. (2022) *Benign, Tempered, or Catastrophic: Overparameterization in Regression* — NeurIPS
- Belkin et al. (2019) *Reconciling Modern Machine Learning and the Bias-Variance Trade-off* — PNAS
- Frei et al. (2023) *Benign Overfitting without Linearity* — ICML
