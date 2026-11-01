from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lgsr.geometry.certified_region import build_certified_mask
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def class_balanced_nll(logits, target, active, feature_dim=None, per_dim=False):
    if target.ndim == logits.ndim:
        target = target[:, 0]
    target = target.long()
    active = active.bool()
    safe_target = torch.where(active, target, torch.zeros_like(target))
    true_logit = torch.gather(logits, 1, safe_target.unsqueeze(1)).squeeze(1)
    values = -true_logit
    class_values = []
    for class_idx in range(logits.shape[1]):
        mask = active & (target == class_idx)
        if mask.any():
            value = values[mask].mean()
            if per_dim:
                if feature_dim is None or int(feature_dim) <= 0:
                    raise ValueError("feature_dim must be positive when per_dim=True")
                value = value / float(feature_dim)
            class_values.append(value)
    return torch.stack(class_values).mean() if class_values else logits.sum() * 0.0


class LGSRSelectiveLoss(nn.Module):
    """The primary LGSR objective with the original tensor operations."""

    def __init__(
        self,
        use_gmm_nll=True,
        gmm_nll_weight=0.01,
        gmm_nll_per_dim=False,
        ignore_label=None,
        use_strict_safe_ball=True,
        safe_ball_rates=(1.0, 1.0, 1.0),
        require_safe_prediction_match=False,
        feature_dim=32,
        use_dice=True,
        dice_weight=1.0,
        dice_batch_dice=True,
        eps=1e-8,
    ):
        super().__init__()
        self.use_gmm_nll = bool(use_gmm_nll)
        self.gmm_nll_weight = float(gmm_nll_weight)
        self.gmm_nll_per_dim = bool(gmm_nll_per_dim)
        self.ignore_label = ignore_label
        self.use_strict_safe_ball = bool(use_strict_safe_ball)
        self.safe_ball_rates = tuple(float(v) for v in safe_ball_rates)
        self.require_safe_prediction_match = bool(require_safe_prediction_match)
        self.feature_dim = int(feature_dim)
        self.use_dice = bool(use_dice)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)
        self.dice_loss = (
            MemoryEfficientSoftDiceLoss(
                apply_nonlin=softmax_helper_dim1,
                batch_dice=bool(dice_batch_dice),
                do_bg=False,
                ddp=False,
            )
            if self.use_dice and self.dice_weight != 0.0 else None
        )
        self.last_debug = {}

    def _prepare_target(self, target, logits):
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()
        if tuple(target.shape[1:]) != tuple(logits.shape[2:]):
            target = F.interpolate(target[:, None].float(), size=logits.shape[2:], mode="nearest")[:, 0].long()
        valid = (target >= 0) & (target < logits.shape[1])
        if self.ignore_label is not None:
            valid = valid & (target != self.ignore_label)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        ce_target = torch.where(valid, target, torch.full_like(target, -100))
        return safe_target, ce_target, valid

    def forward(self, payload, target):
        logits = payload["logits"] if isinstance(payload, dict) else payload
        target, ce_target, valid = self._prepare_target(target, logits)
        zero = logits.sum() * 0.0
        if not valid.any():
            self.last_debug = {"loss_total": zero.detach(), "loss_gmm_ce": zero.detach(), "loss_gmm_nll": zero.detach(), "loss_dice": zero.detach()}
            return zero

        component_scores = payload.get("component_scores") if isinstance(payload, dict) else None
        safe = torch.zeros_like(valid, dtype=torch.bool)
        debug = {}
        if self.use_strict_safe_ball and isinstance(payload, dict):
            safe, debug = build_certified_mask(
                logits,
                component_scores,
                payload["features"],
                target,
                payload["mu"],
                payload["log_alpha"],
                payload["valid_component_mask"],
                self.safe_ball_rates,
                self.require_safe_prediction_match,
                self.eps,
                valid,
            )
        active = valid & ~safe
        ce_per_pixel = F.cross_entropy(logits, ce_target, ignore_index=-100, reduction="none")
        loss_ce = ce_per_pixel[active].mean() if active.any() else zero
        loss_nll = class_balanced_nll(logits, target, active, self.feature_dim, self.gmm_nll_per_dim)
        if self.dice_loss is not None and active.any():
            loss_dice = self.dice_loss(logits, target, loss_mask=active.unsqueeze(1))
        else:
            loss_dice = zero
        total = loss_ce + (self.gmm_nll_weight * loss_nll if self.use_gmm_nll else zero)
        total = total + (self.dice_weight * loss_dice if self.dice_loss is not None else zero)
        self.last_debug = {
            "loss_total": total.detach(), "loss_gmm_ce": loss_ce.detach(),
            "loss_gmm_nll": loss_nll.detach(), "loss_dice": loss_dice.detach(),
            "safe_ratio": safe[valid].float().mean().detach(),
            "outside_ratio": active[valid].float().mean().detach(),
        }
        self.last_debug.update(debug)
        return total
