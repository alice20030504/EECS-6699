"""ResNet-18 with a configurable width multiplier.

Standard ResNet-18 channel progression: [64, 128, 256, 512].
With width_mult k the progression becomes [64k, 128k, 256k, 512k].
"""

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class _WideResNet18(ResNet):
    """ResNet-18 whose channel counts are uniformly scaled by width_mult."""

    def __init__(self, width_mult: int, num_classes: int = 10):
        w = width_mult
        # TODO: call super().__init__() with the right arguments so that
        #   layer1 uses 64*w channels, layer2 128*w, layer3 256*w, layer4 512*w.
        #
        # torchvision.models.resnet.ResNet signature:
        #   ResNet(block, layers, num_classes, zero_init_residual,
        #          groups, width_per_group, replace_stride_with_dilation, norm_layer)
        #
        # The cleanest approach: pass width_per_group=64*w and groups=1, which
        # scales the bottleneck width.  For BasicBlock (ResNet-18/34) this
        # controls the number of planes directly.
        #
        # After super().__init__(), patch self.fc to accept 512*w inputs:
        #   self.fc = nn.Linear(512 * w, num_classes)
        raise NotImplementedError(
            "Instantiate _WideResNet18 with width_mult-scaled channels."
        )


def build_resnet18(width_mult: int, num_classes: int = 10) -> nn.Module:
    """Return a width-scaled ResNet-18.

    Args:
        width_mult: Integer multiplier k. k=1 ≈ standard ResNet-18.
        num_classes: Output dimension (10 for CIFAR-10).
    """
    return _WideResNet18(width_mult, num_classes)
