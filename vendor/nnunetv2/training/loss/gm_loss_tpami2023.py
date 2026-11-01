from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def tpami_paper_coefficients(
    epoch: int,
    *,
    target_alpha: float = 0.3,
    target_likelihood_weight: float = 0.1,
    ramp_epochs: int = 30,
) -> tuple[float, float]:
    """Coefficient ramp used by the authors' high-dimensional CIFAR implementation."""
    if ramp_epochs <= 0 or epoch > ramp_epochs:
        return float(target_alpha), float(target_likelihood_weight)
    progress = max(0.0, float(epoch) / float(ramp_epochs))
    alpha = float(target_alpha) * (0.1 + 0.9 * progress)
    likelihood_weight = float(target_likelihood_weight) * (0.01 + 0.99 * progress)
    return alpha, likelihood_weight


def _prepare_dense_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_label: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_shape = tuple(logits.shape[2:])
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError("Dense class target must have a singleton channel dimension")
        target = target[:, 0]
    if target.ndim != logits.ndim - 1:
        raise ValueError(
            f"Expected target with {logits.ndim - 1} or {logits.ndim} dimensions, got {target.ndim}"
        )
    if tuple(target.shape[1:]) != spatial_shape:
        target = F.interpolate(target[:, None].float(), size=spatial_shape, mode="nearest")[:, 0]
    target = target.long()

    valid = (target >= 0) & (target < logits.shape[1])
    if ignore_label is not None:
        valid = valid & (target != int(ignore_label))
    target_safe = torch.where(valid, target, torch.zeros_like(target))
    return target_safe, valid


def apply_tpami_adaptive_margin(
    logits: torch.Tensor,
    distance_sq: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    ignore_label: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Eq. (19) with the paper's adaptive margin m=alpha*s_y."""
    if logits.shape != distance_sq.shape:
        raise ValueError(
            f"logits and distance_sq must have identical shapes, got {logits.shape} and {distance_sq.shape}"
        )
    if logits.ndim not in (4, 5):
        raise ValueError(f"Expected 2D/3D dense logits, got shape {tuple(logits.shape)}")

    target_safe, valid = _prepare_dense_target(logits, target, ignore_label)
    true_distance_sq = distance_sq.gather(1, target_safe[:, None])[:, 0]
    margin = float(alpha) * 0.5 * true_distance_sq
    margin = margin * valid.to(dtype=margin.dtype)
    margin_delta = torch.zeros_like(logits).scatter(
        1,
        target_safe[:, None],
        margin[:, None],
    )
    return logits - margin_delta, target_safe, valid


class GMMLossTPAMI2023(nn.Module):
    """Pixel-wise reproduction of the TPAMI 2023 Gaussian Mixture loss."""

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        likelihood_weight: float = 0.1,
        ignore_label: Optional[int] = None,
        use_adaptive_margin: bool = True,
        likelihood_mode: str = "all_class_mean",
    ) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if likelihood_weight < 0:
            raise ValueError("likelihood_weight must be non-negative")
        self.alpha = float(alpha)
        self.likelihood_weight = float(likelihood_weight)
        self.ignore_label = ignore_label
        self.use_adaptive_margin = bool(use_adaptive_margin)
        if likelihood_mode not in {"all_class_mean", "true_class"}:
            raise ValueError("likelihood_mode must be 'all_class_mean' or 'true_class'")
        self.likelihood_mode = likelihood_mode
        self.last_debug: dict[str, torch.Tensor] = {}

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        distance_sq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits_float = logits.float()
        distance_sq_float = (
            (-2.0 * logits_float).clamp_min(0.0)
            if distance_sq is None
            else distance_sq.float()
        )
        if self.use_adaptive_margin:
            classification_logits, target_safe, valid = apply_tpami_adaptive_margin(
                logits_float,
                distance_sq_float,
                target,
                alpha=self.alpha,
                ignore_label=self.ignore_label,
            )
        else:
            target_safe, valid = _prepare_dense_target(
                logits_float,
                target,
                self.ignore_label,
            )
            classification_logits = logits_float
        valid_float = valid.to(dtype=logits_float.dtype)
        valid_count = valid_float.sum()
        if int(valid_count.detach().item()) == 0:
            zero = logits_float.sum() * 0.0 + distance_sq_float.sum() * 0.0
            self.last_debug = {
                "loss_gm_total": zero.detach(),
                "loss_gm_cls": zero.detach(),
                "loss_gm_lkd": zero.detach(),
                "mean_true_class_distance_sq": zero.detach(),
                "mean_all_class_distance_sq": zero.detach(),
                "valid_pixel_count": valid_count.detach(),
                "gm_alpha": logits_float.new_tensor(self.alpha),
                "gm_likelihood_weight": logits_float.new_tensor(self.likelihood_weight),
            }
            return zero

        classification_map = F.cross_entropy(classification_logits, target_safe, reduction="none")
        true_distance_sq = distance_sq_float.gather(1, target_safe[:, None])[:, 0]
        likelihood_map = (
            true_distance_sq
            if self.likelihood_mode == "true_class"
            else distance_sq_float.mean(dim=1)
        )
        denominator = valid_count.clamp_min(1.0)
        loss_cls = (classification_map * valid_float).sum() / denominator
        loss_lkd = (likelihood_map * valid_float).sum() / denominator
        mean_true_distance_sq = (true_distance_sq * valid_float).sum() / denominator
        loss_total = loss_cls + self.likelihood_weight * loss_lkd

        self.last_debug = {
            "loss_gm_total": loss_total.detach(),
            "loss_gm_cls": loss_cls.detach(),
            "loss_gm_lkd": loss_lkd.detach(),
            "mean_true_class_distance_sq": mean_true_distance_sq.detach(),
            "mean_all_class_distance_sq": loss_lkd.detach(),
            "valid_pixel_count": valid_count.detach(),
            "gm_alpha": logits_float.new_tensor(self.alpha),
            "gm_likelihood_weight": logits_float.new_tensor(self.likelihood_weight),
        }
        return loss_total


class DeepSupervisionGMMLossTPAMI2023(nn.Module):
    """Apply the same paper loss at each nnU-Net deep-supervision scale."""

    def __init__(self, loss: GMMLossTPAMI2023, weights: Sequence[float]) -> None:
        super().__init__()
        if len(weights) == 0 or sum(float(weight) for weight in weights) <= 0:
            raise ValueError("Deep-supervision weights must contain positive mass")
        self.loss = loss
        self.weights = tuple(float(weight) for weight in weights)
        self.last_debug: dict[str, torch.Tensor] = {}

    def forward(self, logits, target, distance_sq=None) -> torch.Tensor:
        if torch.is_tensor(logits):
            result = self.loss(logits, target, distance_sq)
            self.last_debug = dict(self.loss.last_debug)
            return result
        if not isinstance(logits, (list, tuple)):
            raise TypeError("logits must be a tensor or a sequence of tensors")
        if not isinstance(target, (list, tuple)):
            raise TypeError("Deep-supervision target must be a sequence")
        if distance_sq is None:
            distance_sq = [None] * len(logits)
        if not isinstance(distance_sq, (list, tuple)):
            raise TypeError("Deep-supervision distance_sq must be a sequence when provided")
        if not (len(logits) == len(target) == len(distance_sq) == len(self.weights)):
            raise ValueError("Deep-supervision outputs, targets, distances, and weights must align")

        total = logits[0].float().sum() * 0.0
        weighted_debug: dict[str, torch.Tensor] = {}
        for weight, scale_logits, scale_target, scale_distance in zip(
            self.weights, logits, target, distance_sq
        ):
            if weight == 0.0:
                continue
            scale_loss = self.loss(scale_logits, scale_target, scale_distance)
            total = total + weight * scale_loss
            for key, value in self.loss.last_debug.items():
                if key in ("valid_pixel_count", "gm_alpha", "gm_likelihood_weight"):
                    continue
                weighted_debug[key] = weighted_debug.get(key, value.new_zeros(())) + weight * value
        weighted_debug["loss_gm_total"] = total.detach()
        weighted_debug["gm_alpha"] = total.detach().new_tensor(self.loss.alpha)
        weighted_debug["gm_likelihood_weight"] = total.detach().new_tensor(
            self.loss.likelihood_weight
        )
        self.last_debug = weighted_debug
        return total
