"""Training loop: SGD with linear warmup + cosine LR decay.

Used by experiments/run_R2a.py, run_N1.py, run_N3.py.
Compatible with the YAML-driven config format in configs/*.yaml.
"""
from __future__ import annotations

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
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg          = cfg
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = self._build_optimizer()
        self.scheduler    = self._build_scheduler()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        return torch.optim.SGD(
            self.model.parameters(),
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=False,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Linear warmup for warmup_epochs, then cosine decay to 0."""
        epochs        = self.cfg["epochs"]
        warmup_epochs = self.cfg.get("warmup_epochs", 5)

        warmup  = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=0.0,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    # ── Per-epoch steps ───────────────────────────────────────────────────────

    def train_one_epoch(self) -> dict:
        """One full pass over train_loader. Returns train metrics."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss   = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct    += (logits.argmax(1) == targets).sum().item()
            total      += targets.size(0)

        return {
            "train_loss": total_loss / total,
            "train_acc":  correct / total,
        }

    def evaluate(self) -> dict:
        """Evaluate on test_loader with no gradient."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits = self.model(inputs)
                loss   = self.criterion(logits, targets)
                total_loss += loss.item() * targets.size(0)
                correct    += (logits.argmax(1) == targets).sum().item()
                total      += targets.size(0)

        acc = correct / total
        return {
            "test_loss":  total_loss / total,
            "test_acc":   acc,
            "test_error": 1.0 - acc,
        }

    # ── Full training run ─────────────────────────────────────────────────────

    def run(self, logger=None) -> list[dict]:
        """Train for cfg['epochs'] epochs.

        Args:
            logger: Optional CSVLogger. Each epoch row written immediately
                    so partial results survive Colab disconnects.

        Returns:
            List of per-epoch metric dicts.
        """
        epochs  = self.cfg["epochs"]
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch()
            test_metrics  = self.evaluate()
            self.scheduler.step()

            lr  = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "lr": lr, **train_metrics, **test_metrics}
            history.append(row)

            if logger is not None:
                logger.write_row(row)

            if epoch % 50 == 0 or epoch == epochs:
                print(
                    f"  ep {epoch:4d}/{epochs} | "
                    f"train {train_metrics['train_acc']:.3f} | "
                    f"test  {test_metrics['test_acc']:.3f}"
                )

        return history
