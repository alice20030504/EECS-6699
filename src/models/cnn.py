"""5-layer CNN (3 Conv-BN-Act-Pool + 2 FC) with width multiplier.

Used by: R1 (DD reproduction), N1 (phase diagram), N2 (weight-decay ablation),
         N3 (activation comparison).

Width multiplier k controls all channel counts linearly:
    Conv1: 3  → k
    Conv2: k  → 2k
    Conv3: 2k → 4k
    FC1:   64k → 8k    (after 3× MaxPool(2) on 32×32, spatial = 4×4)
    FC2:   8k  → n_classes

Parameter count ≈ O(k²).  For k=1 → ~750 params, k=64 → ~3 M params.
Interpolation threshold at η=15% observed at k=6 (22K params) in R1.
"""
import torch.nn as nn


_ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
}


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}.")
    return _ACTIVATIONS[name]


class CNN5(nn.Module):
    """5-layer CNN with width multiplier k and swappable activation.

    Args:
        width_multiplier: Positive integer k; scales all channel counts.
        n_classes:        Output classes (10 for CIFAR-10).
        activation:       One of 'relu' (default), 'gelu', 'tanh'.
    """

    def __init__(
        self,
        width_multiplier: int = 1,
        n_classes: int = 10,
        activation: str = 'relu',
    ):
        super().__init__()
        k   = max(1, int(width_multiplier))
        Act = _act(activation)

        self.features = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────
            nn.Conv2d(3,     k,     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k),
            Act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ── Block 2 ──────────────────────────────────────────────────
            nn.Conv2d(k,     2 * k, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * k),
            Act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ── Block 3 ──────────────────────────────────────────────────
            nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4 * k),
            Act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # After 3× MaxPool(2) on 32×32 input: spatial = 4×4
        self.classifier = nn.Sequential(
            nn.Linear(4 * k * 4 * 4, 8 * k),
            Act(),
            nn.Linear(8 * k, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
