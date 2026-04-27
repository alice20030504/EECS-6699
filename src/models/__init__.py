from .cnn    import CNN5
from .resnet import WideResNet18


def build_resnet18(width_mult: int, num_classes: int = 10) -> WideResNet18:
    """Backward-compat alias used by experiments/run_*.py."""
    return WideResNet18(width_multiplier=width_mult, n_classes=num_classes)


__all__ = ['CNN5', 'WideResNet18', 'build_resnet18']
