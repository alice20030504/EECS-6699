# Benign Overfitting in Overparameterized Neural Networks

## Updated Project Plan (12 + 7 Day Timeline)

*4-Person Team | Presentation Deadline: Day 12 | Paper Deadline: Day 19*

---

## 1. Project Overview

### Research Topic

Investigating the benign overfitting phenomenon in overparameterized neural networks: when a network perfectly fits (interpolates) the training data including noise, why can it still maintain good generalization?

### Core Research Question

**In the 2D space of (overparameterization degree × data noise level), when is overfitting benign?**

### Course Theme Alignment

- **Theme 1 (Approximation & Depth)**: Exploring how network capacity (width) affects interpolation
- **Theme 3 (Generalization)**: Core focus — why overparameterized networks generalize
- **Methodology**: Sits in the transition region from NTK/lazy training to finite-width regime

### Project Deliverables

A ~15-page paper containing:

- **1 baseline reproduction**: Nakkiran 2020 model-wise double descent
- **1 main new result**: (Width × Noise) benign overfitting phase diagram
- **1 supplementary experiment**: Weight decay's effect on eliminating the double descent peak
- Complete theoretical discussion and literature comparison

---

## 2. Literature Foundation

### Required Papers (5)

| # | Paper | Purpose | Status |
|---|-------|---------|--------|
| P1 | Belkin et al. (2019) PNAS | Origin of double descent | Required |
| P2 | Bartlett et al. (2020) PNAS | Theoretical definition of benign overfitting | Required |
| P3 | Nakkiran et al. (2020) ICLR | Primary reproduction target | Required |
| P4 | Mallinar et al. (2022) NeurIPS | Theoretical framework for main contribution | Required |
| P5 | Frei et al. (2023) | Benign overfitting theory in nonlinear regime | Required |

### Supplementary Papers (3)

| # | Paper | Purpose | Status |
|---|-------|---------|--------|
| P6 | Kornowski et al. (2024) | Phase transition comparison | Recommended |
| P7 | Schaeffer et al. (2023+) | Mechanism explanation | Recommended |
| P8 | 2024-2026 latest papers (to be searched) | Literature review currency | Recommended |

---

## 3. Experiment Design (Compressed)

### Adjustment Summary

The original plan was designed for 6 weeks (95-120 training runs). Now compressed to 12-day presentation + 7-day paper. Core strategy: cut non-essential experiments, reduce grid density, focus on the most compelling results.

| Experiment | Original Runs | Compressed | GPU-hours | Adjustment |
|------------|--------------|------------|-----------|------------|
| R1 (MNIST warm-up) | 9 | 0 | 0 | Cut: go directly to CIFAR-10 |
| R2a (model-wise DD) | 11 | 11 | ~35 | Keep: core reproduction |
| R2b (epoch-wise DD) | 1 | 0 | 0 | Cut: optional |
| R2c (sample-wise DD) | 6 | 0 | 0 | Cut: optional |
| N1 (phase diagram) | 35-60 | 20 | ~40 | Reduced grid: 5×4=20 points |
| N2 (activation function) | 18 | 0 | 0 | Cut |
| N3 (weight decay) | 15 | 15 | ~25 | Keep |
| **Total** | **95-120** | **46** | **~100** | **~50% reduction** |

---

## 4. Detailed Experiment Plans

### Experiment R2a: CIFAR-10 + ResNet-18 Model-wise Double Descent

- **Objective**: Reproduce model-wise double descent from Nakkiran (2020); validate pipeline correctness
- **Data**: Full CIFAR-10 training set with 15% label noise
- **Model**: ResNet-18 with width multiplier k = {1, 2, 4, 6, 8, 10, 14, 20, 32, 48, 64}
- **Training**: SGD with momentum=0.9, lr=0.1 (warmup + cosine), 500 epochs, batch=128
- **Output**: Figure 1 — Test error vs. k, showing peak near the interpolation threshold
- **Estimated time**: 2-3 days (Day 3-5)

### Main Experiment N1: (Width × Noise) 2D Phase Diagram ⭐ Core Contribution

**Motivation**: Nakkiran mainly fixed noise=15% and swept width; Mallinar proposed a three-way taxonomy but validated on kernel regression. Gap: does a clear three-phase diagram exist on deep networks? Where is the benign boundary?

**Experiment grid (compressed):**

| Dimension | Values | Points |
|-----------|--------|--------|
| Width multiplier k | {2, 8, 16, 32, 64} | 5 |
| Label noise η | {0%, 10%, 20%, 40%} | 4 |
| **Total training runs** | All single runs, no repeats | **20** |

**Fixed hyperparameters**: 4000-sample subset of CIFAR-10; ResNet-18; SGD momentum=0.9, lr=0.1 cosine decay, 4000 epochs, batch=128

**Classification criteria**: Define Test Error Gap = TestErr(η, k) - TestErr(0, k)

- **Benign**: Gap < 2%
- **Tempered**: 2% ≤ Gap < 10%
- **Catastrophic**: Gap ≥ 10%

**Key outputs (3 core figures):**

- **Figure 2**: Test error heatmap, x-axis k (log scale), y-axis η
- **Figure 3**: 4 double descent curves overlaid (different η), showing how noise changes peak shape
- **Figure 4 (core)**: Phase diagram partitioning the (k, η) plane into three colored regions with empirical benign boundary
- **Estimated time**: 3-4 days (Day 6-9)

### Supplementary Experiment N3: Weight Decay Eliminating Double Descent Peak

- **Motivation**: Answer a key question — is double descent an intrinsic phenomenon of deep learning or a symptom of under-regularization?
- **Setup**: Fix η=15%, k = {4, 8, 16}; sweep weight decay λ = {0, 1e-4, 5e-4, 1e-3, 5e-3}; total 3 × 5 = 15 runs
- **Output**: Figure 5 — Double descent curves under different weight decay values
- **Estimated time**: 2-3 days (Day 6-8, parallel with N1)

---

## 5. 12 + 7 Day Timeline & Team Assignments

### Phase 1: Day 1-12 (Presentation Deadline)

| Time | A: Theory & Writing | B: Engineering + N1 Low Noise | C: N1 High Noise | D: N3 + Visualization |
|------|--------------------|-----------------------------|-------------------|----------------------|
| **Day 1-2** | Deep read P1-P4; draft Intro skeleton | Build PyTorch pipeline (ResNet-18 + noise + logging + Git) | Assist B with width multiplier implementation | Set up visualization templates; experiment log sheets |
| **Day 3-5** | Write Sec 2 (Problem Desc) | Run R2a (first half) k={1,2,4,6,8} | Run R2a (second half) k={10,14,20,32,48,64} | Visualize + verify R2a intermediate results; prepare N3 code |
| **Day 6-9** | Complete Sec 1-2; draft Sec 4 skeleton | Run N1 (low noise) η={0%, 10%} | Run N1 (high noise) η={20%, 40%} | Run N3 (weight decay); complete by Day 8 |
| **Day 10-11** | Write Sec 3 + Sec 4 text | Aggregate N1 data; assist with figures | Assist with figures; align on conclusions | Unify figure style; produce Figures 1-5 |
| **Day 12** | Write speaking notes | Prepare Q&A | Prepare Q&A | Create slides; full team rehearsal |

### Phase 2: Day 13-19 (Paper Writing)

| Time | A: Theory & Writing | B: Engineering + N1 Low Noise | C: N1 High Noise | D: N3 + Visualization |
|------|--------------------|-----------------------------|-------------------|----------------------|
| **Day 13-14** | Full draft (expand from presentation) | Write Sec 3.1; code cleanup | Write Sec 4.1 (phase diagram) | Write Sec 4.3 (weight decay) |
| **Day 15-16** | Write Sec 5 (Discussion) | Hyperparameter appendix; supplementary materials | Finalize figures; insert into paper | Unify formatting; all figures |
| **Day 17-18** | Full paper review; polish | Team-wide review | Team-wide review | Team-wide review |
| **Day 19** | Final review; references | Final review; formatting | Final review | Final review & submit |

---

## 6. Detailed Member Responsibilities

### Member A: Theory & Writing Lead (Meng)

- **Day 1-2**: Deep read P1-P4; draft Introduction skeleton
- **Day 3-5**: Write Sec 2 (Problem Description): classical bias-variance vs modern double descent, Bartlett's benign overfitting definition, Mallinar's three-way taxonomy
- **Day 6-9**: Complete Sec 1-2 full draft; draft Sec 4 skeleton (leave blanks for data)
- **Day 10-12**: Write Sec 3 & Sec 4 text; write slide speaking notes
- **Day 13-19**: Complete full draft → Sec 5 Discussion → polish → final review
- **Output**: Sec 1, 2, 5 (~45% of paper) + bibliography management

### Member B: Engineering & Reproduction Lead (Ye)

- **Day 1-2**: Build unified PyTorch pipeline (ResNet-18 + label noise + width multiplier + logging); set up Git repo
- **Day 3-5**: Run R2a first half k={1,2,4,6,8}; deliver reusable code template
- **Day 6-9**: Run N1 low-noise portion η={0%, 10%}, 5×2=10 training runs
- **Day 10-12**: Aggregate N1 data; prepare Q&A
- **Day 13-19**: Write Sec 3.1 + code cleanup + hyperparameter appendix
- **Output**: Engineering infrastructure + Sec 3.1 (~15% of paper)

### Member C: Main Experiment Lead (Alice)

- **Day 1-2**: Assist B with ResNet-18 width multiplier implementation
- **Day 3-5**: Run R2a second half k={10,14,20,32,48,64}
- **Day 6-9**: Run N1 high-noise portion η={20%, 40%}, 5×2=10 training runs
- **Day 10-12**: Assist with figures; align on experiment conclusions; prepare Q&A
- **Day 13-19**: Write Sec 4.1 (phase diagram chapter, paper's core)
- **Output**: Sec 4.1 core chapter (~30% of paper)

### Member D: Supplementary Experiments + Visualization & Coordination Lead (Lyric)

- **Day 1-2**: Set up matplotlib/seaborn unified style template; create experiment log sheets
- **Day 3-5**: Visualize and verify R2a intermediate results; prepare N3 experiment code
- **Day 6-8**: Independently complete N3 (weight decay ablation), 15 training runs
- **Day 9-11**: Unify all figure styles (fonts, colors, labels); produce Figures 1-5
- **Day 12**: Lead presentation slide creation (12-15 slides)
- **Day 13-19**: Write Sec 4.3 + unified formatting + insert figures into paper
- **Output**: Sec 4.3 + all figure coordination + slides (~10% of paper text + 100% figure quality)

---

## 7. Risk Management

| Risk | Likelihood | Contingency |
|------|-----------|-------------|
| R2a fails to reproduce DD peak | Medium | Increase label noise to 20%; extend training; reference Nakkiran's official code |
| Insufficient GPU resources | Medium | Reduce N1 grid to 4×3=12 points; use 2000-sample subset |
| Phase diagram boundary unclear | Medium | Adjust classification thresholds; use continuous heatmap instead of discrete classification |
| Training run fails to converge | Low | Reduce learning rate or increase warmup; change random seed |
| Insufficient writing time | Medium | A starts drafting results skeleton from Day 6; reuse presentation content in paper |
| N3 results inconclusive | Low | Expand λ sweep range; pivot to qualitative discussion |

---

## 8. Expected Paper Structure (15 pages)

| Section | Pages | Main Content |
|---------|-------|-------------|
| 1. Introduction | 1.5 | DL generalization puzzle; introduce benign overfitting concept |
| 2. Problem Description | 3 | Classical vs modern generalization views; Bartlett theory; Mallinar taxonomy |
| 3. Reproduction | 2.5 | R2a model-wise double descent full reproduction |
| 4. New Results | 5.5 | N1 phase diagram (3.5 pages, core) + N3 weight decay (2 pages) |
| 5. Discussion & Conclusion | 2 | Experiment-theory comparison; limitations; future directions |
| References & Appendix | 0.5 | References + hyperparameter table |

---

## 9. Success Criteria

### Minimum (baseline)

- At least one clear model-wise double descent curve
- One readable (width, noise) heatmap or phase diagram
- Complete 15-page paper with all 5 sections

### Expected (target)

- Model-wise double descent fully reproduced
- Clear three-phase diagram (benign/tempered/catastrophic) with empirical boundary
- Weight decay experiment with definitive conclusion
- Discussion with qualitative comparison to Bartlett and Mallinar theory

### Stretch (highlights)

- Propose an empirical formula for the benign boundary
- Add R2b/R2c or N2 during paper period to enrich experiments
- Open-source code and experiment logs for full reproducibility
