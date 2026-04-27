"""Core training loop with checkpointing and structured result logging.

Compatible with all experiments:
  R1 / R2  — basic double descent
  N1       — 2-D phase diagram (same call, different noise_rate)
  N2       — weight-decay ablation  (pass weight_decay=λ)
  N3       — activation comparison  (model already has activation baked in)

Result dict schema (saved as JSON):
  run_id, experiment, width_multiplier, seed, noise_rate,
  train_loss, train_acc, train_error,
  test_loss,  test_acc,  test_error,
  n_params, epochs, wall_time_s,
  history: [{epoch, train_loss, train_acc, test_loss, test_acc}, ...]
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader, device: torch.device):
    """Return (mean_loss, accuracy) over *loader*."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total


# ── Main training function ────────────────────────────────────────────────────

def train_one_run(
    model: nn.Module,
    train_loader,
    test_loader,
    *,
    optimizer_name: str  = 'adam',
    lr: float            = 1e-3,
    weight_decay: float  = 0.0,
    epochs: int          = 300,
    device: torch.device = None,
    checkpoint_dir: str  = None,
    checkpoint_every: int = 10,
    eval_every: int       = 10,
    run_id: str           = 'run',
) -> dict:
    """Train *model* and return a result dict.

    Supports Adam (R1/N1/N2/N3 default) and SGD+cosine (R2).
    Resumes automatically from the latest checkpoint if *checkpoint_dir* is set.

    Args:
        optimizer_name:  'adam' or 'sgd'.
        lr:              Initial learning rate.
        weight_decay:    L2 regularisation coefficient (N2 sweeps this).
        epochs:          Total training epochs.
        device:          Torch device; auto-detected if None.
        checkpoint_dir:  Directory for .pt checkpoints (None = no saving).
        checkpoint_every: Save checkpoint every N epochs.
        eval_every:       Log metrics every N epochs.
        run_id:           Unique string used for checkpoint filenames.

    Returns:
        dict with final metrics, full history, and metadata.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = None
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Use 'adam' or 'sgd'.")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    history: list[dict] = []

    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / f'{run_id}.pt'
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            if scheduler and 'scheduler' in state:
                scheduler.load_state_dict(state['scheduler'])
            start_epoch = state['epoch']
            history     = state.get('history', [])
            print(f"  [ckpt] resumed from epoch {start_epoch}")

    # ── Training loop ──────────────────────────────────────────────────────
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # ── Periodic evaluation ───────────────────────────────────────────
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            tr_loss, tr_acc = evaluate(model, train_loader, device)
            te_loss, te_acc = evaluate(model, test_loader,  device)
            history.append({
                'epoch':      epoch + 1,
                'train_loss': tr_loss,
                'train_acc':  tr_acc,
                'test_loss':  te_loss,
                'test_acc':   te_acc,
            })
            print(
                f"  ep {epoch+1:3d}/{epochs} | "
                f"train {tr_acc:.3f} (loss {tr_loss:.4f}) | "
                f"test  {te_acc:.3f} (loss {te_loss:.4f})"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state = {
                'epoch':     epoch + 1,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history':   history,
            }
            if scheduler:
                state['scheduler'] = scheduler.state_dict()
            torch.save(state, Path(checkpoint_dir) / f'{run_id}.pt')

    # ── Final metrics ──────────────────────────────────────────────────────
    final_tr_loss, final_tr_acc = evaluate(model, train_loader, device)
    final_te_loss, final_te_acc = evaluate(model, test_loader,  device)

    return {
        'run_id':      run_id,
        'train_loss':  final_tr_loss,
        'train_acc':   final_tr_acc,
        'train_error': 1.0 - final_tr_acc,
        'test_loss':   final_te_loss,
        'test_acc':    final_te_acc,
        'test_error':  1.0 - final_te_acc,
        'n_params':    sum(p.numel() for p in model.parameters() if p.requires_grad),
        'epochs':      epochs,
        'wall_time_s': time.time() - t0,
        'history':     history,
    }
