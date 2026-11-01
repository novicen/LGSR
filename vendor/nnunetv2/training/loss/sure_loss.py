from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.devis_loss import compute_evidential_statistics


def _ensure_target_shape(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim:
        target = target[:, 0]
    return target.long()


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target, num_classes=num_classes).movedim(-1, 1).float()


def _masked_mean(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weighted = value * valid_mask
    denom = torch.clamp(valid_mask.sum(), min=1.0)
    return weighted.sum() / denom


class SURESegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        deep_supervision: bool = False,
        ds_weights: Sequence[float] | None = None,
        lambda_uncertainty: float = 0.5,
        include_dice: bool = False,
        ignore_label: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.ds_weights = list(ds_weights) if ds_weights is not None else None
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.include_dice = include_dice
        self.ignore_label = ignore_label
        self.last_debug: dict[str, float] = {}

    def _lambda_anneal(self, current_epoch: int) -> float:
        return min(1.0, float(current_epoch + 1) / 50.0)

    def _sanitize_target(self, target: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = _ensure_target_shape(target, logits)
        if self.ignore_label is None:
            valid_mask = torch.ones_like(target, dtype=logits.dtype)
            return target, valid_mask.unsqueeze(1)

        valid = target != self.ignore_label
        target = torch.where(valid, target, 0)
        return target, valid.unsqueeze(1).to(dtype=logits.dtype)

    def _edl_classification_loss(self, alpha: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        target_1h = _one_hot(target, self.num_classes)
        digamma_strength = torch.digamma(alpha.sum(dim=1, keepdim=True))
        digamma_alpha = torch.digamma(alpha)
        per_pixel = (target_1h * (digamma_strength - digamma_alpha)).sum(dim=1, keepdim=True)
        return _masked_mean(per_pixel, valid_mask)

    def _kl_loss(self, alpha: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        target_1h = _one_hot(target, self.num_classes)
        alpha_tilde = target_1h + (1.0 - target_1h) * alpha

        sum_alpha = alpha_tilde.sum(dim=1, keepdim=True)
        num_classes = alpha.shape[1]

        first = torch.lgamma(sum_alpha) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        second = -torch.lgamma(torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype))
        third = ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
        per_pixel = first + second + third
        return _masked_mean(per_pixel, valid_mask)

    def _uncertainty_supervision_loss(
        self,
        belief: torch.Tensor,
        uncertainty: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = belief.argmax(dim=1)
        uncertainty_target = pred.ne(target).unsqueeze(1).to(dtype=belief.dtype)
        safe_u = uncertainty.clamp(min=1e-6, max=1.0 - 1e-6)
        per_pixel = F.binary_cross_entropy(safe_u, uncertainty_target, reduction="none")
        return _masked_mean(per_pixel, valid_mask)

    def _single_output_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        current_epoch: int,
        max_epochs: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del max_epochs
        stats = compute_evidential_statistics(logits)
        clean_target, valid_mask = self._sanitize_target(target, logits)

        loss_cls = self._edl_classification_loss(stats["alpha"], clean_target, valid_mask)
        loss_kl = self._kl_loss(stats["alpha"], clean_target, valid_mask) * self._lambda_anneal(current_epoch)
        loss_unc = self._uncertainty_supervision_loss(
            stats["belief"], stats["uncertainty"], clean_target, valid_mask
        )
        total = loss_cls + loss_kl + self.lambda_uncertainty * loss_unc
        return total, {
            "loss_cls": loss_cls.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_uncertainty": loss_unc.detach(),
        }

    def forward(self, net_output, target, current_epoch: int = 0, max_epochs: int = 250):
        if isinstance(net_output, (list, tuple)):
            outputs = list(net_output)
            targets = list(target) if isinstance(target, (list, tuple)) else [target] * len(outputs)
            weights = self.ds_weights if self.ds_weights is not None else [1.0] * len(outputs)
            total = outputs[0].new_zeros(())
            debug_acc = {"loss_cls": 0.0, "loss_kl": 0.0, "loss_uncertainty": 0.0}
            for out_i, tgt_i, w_i in zip(outputs, targets, weights):
                l_i, dbg_i = self._single_output_loss(out_i, tgt_i, current_epoch, max_epochs)
                total = total + float(w_i) * l_i
                for key in debug_acc:
                    debug_acc[key] += float(w_i) * float(dbg_i[key].cpu())
            self.last_debug = debug_acc
            return total

        total, dbg = self._single_output_loss(net_output, target, current_epoch, max_epochs)
        self.last_debug = {k: float(v.cpu()) for k, v in dbg.items()}
        return total
