# Benign Overfitting in Overparameterized Neural Networks

EECS-6699 Final Project | Columbia University | Spring 2026

**Team**: Yuxia Meng (A · Theory & Writing), Yixuan Ye (B · Engineering), Shurong Zhang (C · Main Experiments), Rui Li (D · Visualization & Coordination)

---

## Project Goal

We investigate the *benign overfitting* phenomenon: when an overparameterized network perfectly interpolates noisy training data, under what conditions does it still generalize well?

Our central question:

> **In the 2-D space of (width multiplier k × label noise η), where is overfitting benign, tempered, or catastrophic?**

This extends Nakkiran et al. (2020)'s model-wise double-descent study — which fixed η = 15% and swept width — by explicitly mapping both dimensions simultaneously and comparing the empirical boundary to the theoretical taxonomy of Mallinar et al. (2022).

---

## Experiment List

| ID | Name | Description | Owner | Status |
|----|------|-------------|-------|--------|
| **R2a** | Model-wise Double Descent | Reproduce Nakkiran 2020: ResNet-18 on CIFAR-10 (15% label noise), sweep width multiplier k ∈ {1,2,4,6,8,10,14,20,32,48,64}, 500 epochs | Yixuan (k≤8) + Shurong (k≥10) | Planned |
| **N1** | (Width × Noise) Phase Diagram | Core contribution: 5×4 grid — k ∈ {2,8,16,32,64}, η ∈ {0%,10%,20%,40%}, 4000-sample CIFAR-10 subset, 4000 epochs | Yixuan (η≤10%) + Shurong (η≥20%) | Planned |
| **N3** | Weight Decay Ablation | Fix η=15%, k ∈ {4,8,16}; sweep λ ∈ {0,1e-4,5e-4,1e-3,5e-3}; 15 runs | Rui Li | Planned |

### Classification Criteria (N1)

Define **Test Error Gap** = TestErr(η, k) − TestErr(0, k):

| Region | Condition |
|--------|-----------|
| Benign | Gap < 2% |
| Tempered | 2% ≤ Gap < 10% |
| Catastrophic | Gap ≥ 10% |

### Expected Output Figures

| Figure | Content |
|--------|---------|
| Fig. 1 | R2a: Test error vs. width multiplier k (model-wise double-descent curve) |
| Fig. 2 | N1: Test error heatmap in (k, η) space |
| Fig. 3 | N1: Four double-descent curves overlaid for η ∈ {0%,10%,20%,40%} |
| Fig. 4 | N1: Phase diagram — (k, η) plane colored benign / tempered / catastrophic |
| Fig. 5 | N3: Double-descent curves under different weight decay values |

---

## Directory Structure

```
EECS-6699/
├── README.md
├── requirements.txt
│
├── src/                        # Shared library code (no training scripts here)
│   ├── models/                 # ResNet-18 with width multiplier
│   ├── data/                   # CIFAR-10 loader with label noise injection
│   ├── training/               # Training loop, SGD + cosine LR schedule
│   └── utils/                  # Checkpointing, CSV logging, metrics
│
├── configs/                    # Per-experiment YAML configuration files
│   ├── R2a.yaml
│   ├── N1.yaml
│   └── N3.yaml
│
├── experiments/                # Runnable entry-point scripts
│   ├── run_R2a.py
│   ├── run_N1.py
│   └── run_N3.py
│
├── results/                    # Raw outputs: CSV logs, checkpoints (git-ignored large files)
│   ├── R2a/
│   ├── N1/
│   └── N3/
│
├── figures/                    # Generated plots (committed after review)
│   ├── R2a/
│   ├── N1/
│   └── N3/
│
└── notebooks/                  # Exploratory analysis and figure polishing
```

---

## Key Hyperparameters

| Setting | Value |
|---------|-------|
| Optimizer | SGD, momentum = 0.9 |
| Learning rate | 0.1, linear warmup + cosine decay |
| Batch size | 128 |
| R2a epochs | 500 |
| N1 epochs | 4000 |
| N1 dataset size | 4000-sample CIFAR-10 subset |
| N3 width multipliers | k ∈ {4, 8, 16} |
| N3 weight decay λ | {0, 1e-4, 5e-4, 1e-3, 5e-3} |

---

## Literature

| # | Paper |
|---|-------|
| P1 | Belkin et al. (2019), *Reconciling modern machine learning practice and the bias-variance trade-off*, PNAS |
| P2 | Bartlett et al. (2020), *Benign overfitting in linear regression*, PNAS |
| P3 | Nakkiran et al. (2020), *Deep double descent*, ICLR |
| P4 | Mallinar et al. (2022), *Benign, tempered, or catastrophic: a taxonomy of overfitting*, NeurIPS |
| P5 | Frei et al. (2023), *Benign overfitting without linearity* |

---

## Deadlines

| Milestone | Date |
|-----------|------|
| Presentation | May 4, 2026 (Day 12) |
| Final paper due | ~May 14, 2026 (Day 19) |

---

## Setup

```bash
pip install -r requirements.txt
```

See `experiments/` for runnable entry-point scripts.
