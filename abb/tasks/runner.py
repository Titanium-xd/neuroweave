"""
abb.tasks.runner — Minimal training and evaluation harness for ABB tasks.

Connects any AbstractAgent to a TensorDataset task with a consistent
training loop and evaluation protocol.

Design decisions:
- Classification tasks only (cross-entropy loss)
- Adam optimizer, fixed learning rate
- No RL, no environments, no episodes
- Provenance dict is returned with every result
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from abb.models.base import AbstractAgent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: AbstractAgent,
    dataset: TensorDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compute classification accuracy and loss on a TensorDataset.

    Parameters
    ----------
    model : AbstractAgent
    dataset : TensorDataset — each item is (obs, label)
    batch_size : int
    device : torch.device | None

    Returns
    -------
    dict with keys: accuracy, loss, n_samples, n_correct
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.no_grad():
        for obs, labels in loader:
            obs, labels = obs.to(device), labels.to(device)
            model.reset_state(batch_size=len(obs), device=device)
            
            if obs.dim() == 3:
                # Sequence task: (B, T, D)
                for t in range(obs.size(1)):
                    logits = model(obs[:, t, :])
            else:
                # Static task: (B, D)
                logits = model(obs)                    # (B, D_act)
                
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_n += len(labels)

    return {
        "accuracy": total_correct / total_n,
        "loss": total_loss / total_n,
        "n_samples": total_n,
        "n_correct": total_correct,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    model: AbstractAgent,
    dataset: TensorDataset,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> dict:
    """One epoch of cross-entropy training."""
    device = device or torch.device("cpu")
    model.train()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss, total_correct, total_n = 0.0, 0, 0

    for obs, labels in loader:
        obs, labels = obs.to(device), labels.to(device)
        optimizer.zero_grad()
        model.reset_state(batch_size=len(obs), device=device)
        
        if obs.dim() == 3:
            # Sequence task: (B, T, D)
            for t in range(obs.size(1)):
                logits = model(obs[:, t, :])
        else:
            # Static task: (B, D)
            logits = model(obs)
            
        loss = criterion(logits, labels)
        loss.backward()
        # Clip gradients to prevent instability with large-logit graph models
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_n += len(labels)

    return {
        "train_loss": total_loss / total_n,
        "train_accuracy": total_correct / total_n,
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run_task(
    model: AbstractAgent,
    train_set: TensorDataset,
    dev_set: TensorDataset,
    test_set: TensorDataset,
    n_epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    log_every: int = 10,
) -> dict:
    """
    Train a model on a classification task and return results.

    Parameters
    ----------
    model : AbstractAgent
    train_set, dev_set, test_set : TensorDataset
    n_epochs : int
    lr : float — Adam learning rate
    batch_size : int
    device : torch.device | None
    verbose : bool
    log_every : int — print progress every N epochs

    Returns
    -------
    dict with:
        history : list of per-epoch dicts (train_loss, train_acc, dev_acc)
        best_dev_acc : float
        test_acc : float
        test_loss : float
        n_params : int
        wall_time_s : float
        arch_id : str
    """
    device = device or torch.device("cpu")
    model = model.to(device)

    # Only optimize trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        # Frozen model (A1-FROZEN): evaluate without training
        test_result = evaluate(model, test_set, batch_size, device)
        dev_result  = evaluate(model, dev_set,  batch_size, device)
        return {
            "history": [],
            "best_dev_acc": dev_result["accuracy"],
            "test_acc": test_result["accuracy"],
            "test_loss": test_result["loss"],
            "n_params": model.param_count()["trainable"],
            "wall_time_s": 0.0,
            "arch_id": model.config.arch_id,
            "note": "No trainable params — evaluated without training",
        }

    optimizer = optim.Adam(trainable, lr=lr)

    history = []
    best_dev_acc = 0.0
    t0 = time.monotonic()

    for epoch in range(1, n_epochs + 1):
        train_stats = train_epoch(model, train_set, optimizer, batch_size, device)
        dev_stats   = evaluate(model, dev_set, batch_size, device)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_acc": train_stats["train_accuracy"],
            "dev_acc": dev_stats["accuracy"],
            "dev_loss": dev_stats["loss"],
        }
        history.append(row)

        if dev_stats["accuracy"] > best_dev_acc:
            best_dev_acc = dev_stats["accuracy"]

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(
                f"  epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_stats['train_loss']:.4f}  "
                f"train_acc={train_stats['train_accuracy']:.3f}  "
                f"dev_acc={dev_stats['accuracy']:.3f}"
            )

    wall_time = time.monotonic() - t0
    test_result = evaluate(model, test_set, batch_size, device)

    return {
        "history": history,
        "best_dev_acc": best_dev_acc,
        "test_acc": test_result["accuracy"],
        "test_loss": test_result["loss"],
        "n_params": model.param_count()["trainable"],
        "wall_time_s": wall_time,
        "arch_id": model.config.arch_id,
    }
