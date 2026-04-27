# Benign Overfitting in Overparameterized Neural Networks
EECS 6699 Project — Spring 2026

## Overview
We study the **benign overfitting** phenomenon in deep CNNs by systematically mapping out the **(width × label-noise)** phase diagram on CIFAR-10.

## Repository structure

```
project-6699/
├── src/
│   ├── data.py          # CIFAR-10 subset + configurable label noise
│   ├── train.py         # Training loop (Adam/SGD, checkpointing, JSON logging)
│   ├── io_utils.py      # save/load results, Google Drive helpers
│   ├── plot_utils.py    # Unified plotting style (all figures)
│   └── models/
│       ├── cnn.py       # CNN5 — 5-layer CNN with width multiplier k
│       └── resnet.py    # WideResNet18 — ResNet-18 with width multiplier k
├── run_r1.py            # R1: CNN double descent (22 runs)
├── run_r2.py            # R2: ResNet-18 double descent (6 runs)
├── notebooks/
│   ├── R1_CNN_DoubleDescent.ipynb
│   └── R2_ResNet_DoubleDescent.ipynb
├── results/             # Auto-created; JSON results + PNG figures
└── requirements.txt
```

## Experiments

| ID | Description | Model | Runs | Script |
|----|-------------|-------|------|--------|
| **R1** | CNN double descent (reproduce DD curve) | CNN5 | 22 | `run_r1.py` |
| **R2** | ResNet-18 validation | WideResNet18 | 6 | `run_r2.py` |
| N1 | Width × Noise 2-D phase diagram *(coming)* | CNN5 | 60 | — |
| N2 | Weight-decay ablation *(coming)* | CNN5 | 15 | — |
| N3 | Activation-function comparison *(coming)* | CNN5 | 12 | — |

## Quick start (local)

```bash
pip install -r requirements.txt
python run_r1.py          # runs all 22 R1 jobs, saves results/, plots Fig 1
python run_r2.py          # runs all 6  R2 jobs, saves results/, plots Fig 2
```

Results are saved as JSON in `results/R1/` and `results/R2/`.  
Re-running is safe — completed runs are automatically skipped (`resume=True`).

## Quick start (Colab)

1. Open `notebooks/R1_CNN_DoubleDescent.ipynb` in Google Colab.
2. Set `REPO_URL` to your GitHub repo URL in cell 2.
3. Set `USE_DRIVE = True` to persist results on Google Drive.
4. Runtime → Change runtime type → **GPU (T4)**.
5. Run all cells.

To parallelize across multiple Colab accounts, uncomment the width-split block
in cell 3 of each notebook.

## Shared pipeline contract

All experiments use the same JSON result schema:

```json
{
  "experiment":       "R1",
  "run_id":           "r1_k008_s042",
  "width_multiplier": 8,
  "seed":             42,
  "noise_rate":       0.15,
  "train_error":      0.012,
  "test_error":       0.341,
  "n_params":         47424,
  "epochs":           300,
  "wall_time_s":      480,
  "history":          [...]
}
```

N1/N2/N3 runners add extra keys (`weight_decay`, `activation`) but load with the
same `src.io_utils.load_results()` utility.

## Key design choices

| Choice | Rationale |
|--------|-----------|
| CNN5 channels `[k, 2k, 4k]` | Params ≈ O(k²); interpolation threshold visible ~k=8–16 for n=5000 |
| ResNet channels `[k, 2k, 4k, 8k]` | k=1 gives ~3K params (under-parameterized); DD peak near k=2–4 |
| Adam lr=1e-3, 300 ep (CNN) | Consistent with Nakkiran et al. (2020) |
| SGD cosine lr=0.1, 200 ep (ResNet) | Standard ResNet recipe |
| Checkpoint every 10 ep | Survives Colab disconnects; resume is automatic |
| Results as JSON | Simple, diffable, loadable by N1–N3 without code changes |

## References

- Nakkiran et al. (2020) *Deep Double Descent* — ICLR 2020
- Bartlett et al. (2020) *Benign Overfitting in Linear Regression* — PNAS
- Mallinar et al. (2022) *Benign, Tempered, or Catastrophic* — NeurIPS 2022
