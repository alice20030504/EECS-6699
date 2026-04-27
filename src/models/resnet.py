"""Width-scalable ResNet-18 for CIFAR-10 (R2).

Channel progression across the 4 stages: [k, 2k, 4k, 8k].
Uses a 3×3 stem with no max-pool (standard CIFAR variant).

Parameter count ≈ O(k²):
    k=1  →  ~3 K params   (under-parameterized for n=5 000)
    k=2  →  ~12 K params  (near interpolation threshold)
    k=4  →  ~50 K params  (over-parameterized)
    k=32 →  ~3 M params   (deep benign region)
"""
import torch.nn as nn


class _BasicBlock(nn.Module):
    """Standard ResNet basic block (no bottleneck)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class WideResNet18(nn.Module):
    """ResNet-18 with width multiplier k.

    Args:
        width_multiplier: Positive integer k; channels = [k, 2k, 4k, 8k].
        n_classes:        Output classes (10 for CIFAR-10).
    """

    def __init__(self, width_multiplier: int = 1, n_classes: int = 10):
        super().__init__()
        k = max(1, int(width_multiplier))

        # ── Stem (no max-pool: preserve spatial for small k) ─────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
        )

        # ── 4 residual stages, 2 blocks each ─────────────────────────────
        self.layer1 = self._stage(k,     k,     n=2, stride=1)
        self.layer2 = self._stage(k,     2 * k, n=2, stride=2)
        self.layer3 = self._stage(2 * k, 4 * k, n=2, stride=2)
        self.layer4 = self._stage(4 * k, 8 * k, n=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(8 * k, n_classes)

    @staticmethod
    def _stage(in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n - 1):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
