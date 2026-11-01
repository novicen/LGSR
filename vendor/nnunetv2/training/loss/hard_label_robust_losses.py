"""Losses for independent hard-label robustness baselines.

These are pixel-wise adaptations of published classification losses. They do
not use the LGSR head, feature anchors, or score gate.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _prepare_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError(f"Expected singleton target channel, got {tuple(target.shape)}")
        target = target[:, 0]
    return target.long()


def _valid_target(target: torch.Tensor, num_classes: int, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    valid = target != ignore_index
    safe = target.clamp(min=0, max=num_classes - 1)
    return safe, valid


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Categorical label smoothing for dense mutually-exclusive segmentation."""

    def __init__(self, num_classes: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        safe_target, valid = _valid_target(target, logits.shape[1], self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0
        one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).movedim(-1, 1).to(logits.dtype)
        soft_target = one_hot * (1.0 - self.smoothing) + self.smoothing / logits.shape[1]
        loss = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1)
        return loss[valid].mean()


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized Cross Entropy (GCE) with q in (0, 1]."""

    def __init__(self, q: float = 0.7, ignore_index: int = -100, eps: float = 1e-8):
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {q}")
        self.q = float(q)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        safe_target, valid = _valid_target(target, logits.shape[1], self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0
        prob_true = torch.gather(F.softmax(logits, dim=1), 1, safe_target.unsqueeze(1)).squeeze(1)
        prob_true = prob_true.clamp_min(self.eps)
        loss = (1.0 - prob_true.pow(self.q)) / self.q
        return loss[valid].mean()


class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross Entropy: alpha*CE + beta*Reverse-CE."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        label_clip: float = 1e-4,
        ignore_index: int = -100,
    ):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if not 0.0 < label_clip < 1.0:
            raise ValueError(f"label_clip must be in (0, 1), got {label_clip}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.label_clip = float(label_clip)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        safe_target, valid = _valid_target(target, logits.shape[1], self.ignore_index)
        if not valid.any():
            return logits.sum() * 0.0
        prob = F.softmax(logits, dim=1).clamp_min(self.label_clip)
        log_prob = prob.log()
        hard_target = F.one_hot(safe_target, num_classes=logits.shape[1]).movedim(-1, 1).to(logits.dtype)
        ce = -(hard_target * log_prob).sum(dim=1)
        clipped_label = hard_target.clamp_min(self.label_clip)
        reverse_ce = -(prob * clipped_label.log()).sum(dim=1)
        loss = self.alpha * ce + self.beta * reverse_ce
        return loss[valid].mean()
