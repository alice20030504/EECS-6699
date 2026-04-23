"""Training loop: SGD with linear warmup + cosine LR decay."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Encapsulates a single training run (one (model, noise, weight_decay) combo).

    Usage::

        trainer = Trainer(model, train_loader, test_loader, cfg["training"])
        history = trainer.run()   # returns list of per-epoch dicts
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cfg: dict,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        return torch.optim.SGD(
            self.model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=False,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Linear warmup for warmup_epochs, then cosine decay to 0.

        TODO:
            Use torch.optim.lr_scheduler.SequentialLR combining:
              1. LinearLR(start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
              2. CosineAnnealingLR(T_max=epochs - warmup_epochs, eta_min=0)
            torch docs: https://pytorch.org/docs/stable/optim.html
        """
        raise NotImplementedError("Implement warmup + cosine LR schedule.")

    # ------------------------------------------------------------------
    # Per-epoch steps
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> dict:
        """Run one full pass over train_loader. Return train metrics.

        Returns:
            dict with keys: train_loss, train_acc (float, averaged over batches)

        TODO:
            self.model.train()
            Loop over (inputs, targets) from self.train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                accumulate loss and correct predictions
            Return averaged metrics.
        """
        raise NotImplementedError("Implement training loop.")

    def evaluate(self) -> dict:
        """Evaluate on test_loader with no gradient.

        Returns:
            dict with keys: test_loss, test_acc, test_error (= 1 - test_acc)

        TODO:
            self.model.eval()
            with torch.no_grad():
                loop over test_loader, accumulate loss and correct predictions
            Return averaged metrics.
        """
        raise NotImplementedError("Implement evaluation loop.")

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def run(self, logger=None) -> list[dict]:
        """Train for cfg['epochs'] epochs.

        Args:
            logger: Optional CSVLogger. If provided, each epoch row is written
                    immediately (safe for long runs that may be interrupted).

        Returns:
            List of per-epoch metric dicts (epoch, train_loss, train_acc,
            test_loss, test_acc, test_error, lr).
        """
        epochs = self.cfg["epochs"]
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch()
            test_metrics = self.evaluate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "lr": lr, **train_metrics, **test_metrics}
            history.append(row)

            if logger is not None:
                logger.write_row(row)

        return history
