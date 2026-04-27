"""CIFAR-10 subset with configurable symmetric label noise.

Shared by R1, R2, N1, N2, N3 — only noise_rate and seed differ per experiment.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


class NoisySubset(Dataset):
    """Fixed CIFAR-10 subset with symmetric label noise.

    Noise is applied once at construction; the same (data_seed, noise_rate)
    pair always produces the same noisy labels — reproducible across runs.
    """

    def __init__(
        self,
        base_dataset,
        indices: np.ndarray,
        noise_rate: float = 0.0,
        seed: int = 42,
        n_classes: int = 10,
    ):
        self.base    = base_dataset
        self.indices = np.asarray(indices)
        n            = len(self.indices)

        orig = np.array([base_dataset.targets[i] for i in self.indices])
        rng  = np.random.default_rng(seed)

        noise_mask = rng.random(n) < noise_rate
        # For each noisy sample, draw uniformly from the other (n_classes - 1) labels
        other = np.array([
            rng.choice([c for c in range(n_classes) if c != y])
            for y in orig
        ])
        self.labels            = np.where(noise_mask, other, orig).astype(np.int64)
        self.actual_noise_rate = float(noise_mask.mean())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base[self.indices[idx]]
        return img, int(self.labels[idx])


def get_cifar10_loaders(
    n_train: int   = 5000,
    noise_rate: float = 0.15,
    data_seed: int = 42,
    batch_size: int = 128,
    data_root: str = './data',
    num_workers: int = 2,
):
    """Return ``(train_loader, test_loader)`` for a CIFAR-10 subset.

    Args:
        n_train:    Number of training samples to keep (subset of 50 000).
        noise_rate: Fraction of training labels to flip randomly (symmetric).
        data_seed:  Controls both subset selection and label noise.
        batch_size: Mini-batch size for the training loader.
        data_root:  Directory to download / cache CIFAR-10.
        num_workers: DataLoader worker processes.

    Returns:
        train_loader: Shuffled loader over the noisy subset.
        test_loader:  Full 10 000-sample clean test set.
    """
    transform_train = T.Compose([T.ToTensor(), T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)])
    transform_test  = T.Compose([T.ToTensor(), T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)])

    train_full = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=transform_train
    )
    test_full = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    rng     = np.random.default_rng(data_seed)
    indices = rng.choice(len(train_full), n_train, replace=False)

    train_set = NoisySubset(train_full, indices, noise_rate=noise_rate, seed=data_seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_full, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(
        f"[data] n_train={len(train_set)}, "
        f"noise_rate={noise_rate:.0%}, "
        f"actual={train_set.actual_noise_rate:.3f}"
    )
    return train_loader, test_loader


# ── Backward-compatibility shim for experiments/run_*.py ─────────────────────
# Their runners call get_cifar10_loaders(noise_eta=..., subset_size=...)
# We accept those names and forward to the canonical params.
_orig_get_cifar10_loaders = get_cifar10_loaders

def get_cifar10_loaders(          # noqa: F811  (intentional re-def)
    n_train: int        = 5000,
    noise_rate: float   = 0.15,
    data_seed: int      = 42,
    batch_size: int     = 128,
    data_root: str      = './data',
    num_workers: int    = 2,
    # Legacy aliases
    noise_eta: float    = None,
    subset_size: int    = None,
):
    if noise_eta is not None:
        noise_rate = noise_eta
    if subset_size is not None:
        n_train = subset_size
    return _orig_get_cifar10_loaders(
        n_train=n_train, noise_rate=noise_rate, data_seed=data_seed,
        batch_size=batch_size, data_root=data_root, num_workers=num_workers,
    )
