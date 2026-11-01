from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


def compute_evidential_statistics(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    belief = evidence / torch.clamp(strength, min=1e-8)
    uncertainty = logits.shape[1] / torch.clamp(strength, min=1e-8)
    return {
        "logits": logits,
        "evidence": evidence,
        "alpha": alpha,
        "strength": strength,
        "belief": belief,
        "uncertainty": uncertainty,
    }


def _ensure_target_shape(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim:
        target = target[:, 0]
    return target.long()


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target, num_classes=num_classes).movedim(-1, 1).float()


class _BeliefDiceAdapter(nn.Module):
    def __init__(self, batch_dice: bool, do_bg: bool, ddp: bool):
        super().__init__()
        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=None,
            batch_dice=batch_dice,
            do_bg=do_bg,
            smooth=1e-5,
            ddp=ddp,
        )

    def forward(self, belief: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dc(belief, target)


class DEViSLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        deep_supervision: bool = False,
        ds_weights: Sequence[float] | None = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        ddp: bool = True,
        beta0: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.ds_weights = list(ds_weights) if ds_weights is not None else None
        self.beta0 = beta0
        self.dice = _BeliefDiceAdapter(batch_dice=batch_dice, do_bg=do_bg, ddp=ddp)
        self.last_debug: dict[str, float] = {}

    def _lambda_anneal(self, current_epoch: int) -> float:
        return min(1.0, float(current_epoch + 1) / 50.0)

    def _beta_t(self, current_epoch: int, max_epochs: int) -> float:
        if max_epochs <= 0:
            return 1.0
        beta0 = max(self.beta0, 1e-8)
        decay = -math.log(beta0) * float(current_epoch) / float(max_epochs)
        return float(beta0 * math.exp(decay))

    def _dice_loss(self, belief: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(belief, target)

    def _ice_loss(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _ensure_target_shape(target, alpha)
        target_1h = _one_hot(target, self.num_classes)
        digamma_strength = torch.digamma(alpha.sum(dim=1, keepdim=True))
        digamma_alpha = torch.digamma(alpha)
        return (target_1h * (digamma_strength - digamma_alpha)).sum(dim=1).mean()

    def _kl_loss(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _ensure_target_shape(target, alpha)
        target_1h = _one_hot(target, self.num_classes)
        alpha_tilde = target_1h + (1.0 - target_1h) * alpha

        sum_alpha = alpha_tilde.sum(dim=1, keepdim=True)
        num_classes = alpha.shape[1]

        first = torch.lgamma(sum_alpha) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        second = -torch.lgamma(torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype))
        second = first + second
        third = ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
        return (second + third).mean()

    def _cup_loss(self, belief: torch.Tensor, uncertainty: torch.Tensor, target: torch.Tensor, current_epoch: int, max_epochs: int) -> torch.Tensor:
        target = _ensure_target_shape(target, belief)
        pred = belief.argmax(dim=1)
        correct = pred.eq(target).unsqueeze(1)
        beta_t = self._beta_t(current_epoch, max_epochs)

        safe_u = uncertainty.clamp(min=1e-6, max=1.0 - 1e-6)
        correct_term = -(belief * torch.log(1.0 - safe_u)) * correct
        incorrect_term = -((1.0 - belief) * torch.log(safe_u)) * (~correct)
        return beta_t * correct_term.sum(dim=1).mean() + (1.0 - beta_t) * incorrect_term.sum(dim=1).mean()

    def _single_output_loss(self, logits: torch.Tensor, target: torch.Tensor, current_epoch: int, max_epochs: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stats = compute_evidential_statistics(logits)
        loss_ice = self._ice_loss(stats["alpha"], target)
        loss_kl = self._kl_loss(stats["alpha"], target) * self._lambda_anneal(current_epoch)
        loss_dice = self._dice_loss(stats["belief"], target)
        loss_cup = self._cup_loss(stats["belief"], stats["uncertainty"], target, current_epoch, max_epochs)
        total = loss_ice + loss_kl + loss_dice + loss_cup
        return total, {
            "loss_ice": loss_ice.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_cup": loss_cup.detach(),
        }

    def forward(self, net_output, target, current_epoch: int = 0, max_epochs: int = 250):
        if isinstance(net_output, (list, tuple)):
            outputs = list(net_output)
            targets = list(target) if isinstance(target, (list, tuple)) else [target] * len(outputs)
            weights = self.ds_weights if self.ds_weights is not None else [1.0] * len(outputs)
            total = outputs[0].new_zeros(())
            debug_acc = {"loss_ice": 0.0, "loss_kl": 0.0, "loss_dice": 0.0, "loss_cup": 0.0}
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
