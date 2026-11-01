from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class SoftSpectralNorm(nn.Module):
    """Bound a weight's spectral norm without rescaling weights below the bound."""

    def __init__(
        self,
        weight: torch.Tensor,
        coeff: float = 3.0,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if coeff <= 0:
            raise ValueError(f"coeff must be positive, got {coeff}")
        if n_power_iterations <= 0:
            raise ValueError(f"n_power_iterations must be positive, got {n_power_iterations}")
        self.coeff = float(coeff)
        self.n_power_iterations = int(n_power_iterations)
        self.dim = int(dim)
        self.eps = float(eps)

        weight_matrix = self._reshape_weight_to_matrix(weight.detach())
        self.register_buffer("u", F.normalize(weight.new_empty(weight_matrix.shape[0]).normal_(), dim=0, eps=eps))
        self.register_buffer("v", F.normalize(weight.new_empty(weight_matrix.shape[1]).normal_(), dim=0, eps=eps))
        self.register_buffer("sigma", weight.new_ones(()))

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if self.dim != 0:
            order = [self.dim] + [idx for idx in range(weight.ndim) if idx != self.dim]
            weight = weight.permute(order)
        return weight.reshape(weight.shape[0], -1)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight_matrix = self._reshape_weight_to_matrix(weight)
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    self.v.copy_(F.normalize(torch.mv(weight_matrix.t(), self.u), dim=0, eps=self.eps))
                    self.u.copy_(F.normalize(torch.mv(weight_matrix, self.v), dim=0, eps=self.eps))

        u = self.u.clone()
        v = self.v.clone()
        sigma = torch.dot(u, torch.mv(weight_matrix, v))
        self.sigma.copy_(sigma.detach())
        scale = torch.clamp(sigma / self.coeff, min=1.0)
        return weight / scale


def apply_soft_spectral_norm(
    module: nn.Module,
    coeff: float = 3.0,
    n_power_iterations: int = 1,
    exclude_modules: Iterable[nn.Module] = (),
) -> int:
    """Apply DDU-style soft spectral normalization to convolutional/linear layers."""
    excluded_ids = set()
    for excluded in exclude_modules:
        excluded_ids.update(id(child) for child in excluded.modules())

    supported = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    transpose_types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    count = 0
    for child in module.modules():
        if id(child) in excluded_ids or not isinstance(child, supported):
            continue
        if parametrize.is_parametrized(child, "weight"):
            continue
        dim = 1 if isinstance(child, transpose_types) else 0
        parametrize.register_parametrization(
            child,
            "weight",
            SoftSpectralNorm(
                child.weight,
                coeff=coeff,
                n_power_iterations=n_power_iterations,
                dim=dim,
            ),
        )
        count += 1
    return count


def _as_segmentation_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim >= 4 and target.shape[1] == 1:
        target = target[:, 0]
    return target.long()


def prepare_dense_feature_target(
    features: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_label: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened [N,C] features, [N] labels, and a valid-label mask.

    The target is nearest-neighbor resized when its spatial shape differs from
    the feature map, which keeps feature losses aligned with nnU-Net outputs.
    """
    if features.ndim not in (4, 5):
        raise ValueError(f"features must be [B,C,H,W] or [B,C,D,H,W], got {tuple(features.shape)}")
    target = _as_segmentation_target(target)
    if tuple(target.shape[1:]) != tuple(features.shape[2:]):
        target = F.interpolate(target[:, None].float(), size=features.shape[2:], mode="nearest")[:, 0].long()

    flat_features = features.permute(0, *range(2, features.ndim), 1).reshape(-1, features.shape[1])
    flat_target = target.reshape(-1)
    valid = (flat_target >= 0) & (flat_target < int(num_classes))
    if ignore_label is not None:
        valid &= flat_target != int(ignore_label)
    return flat_features, flat_target, valid


def prepare_dense_logit_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_label: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return class-label target aligned to logits and a valid-label mask."""
    if logits.ndim not in (4, 5):
        raise ValueError(f"logits must be [B,C,H,W] or [B,C,D,H,W], got {tuple(logits.shape)}")
    target = _as_segmentation_target(target)
    if tuple(target.shape[1:]) != tuple(logits.shape[2:]):
        target = F.interpolate(target[:, None].float(), size=logits.shape[2:], mode="nearest")[:, 0].long()

    valid = (target >= 0) & (target < int(num_classes))
    if ignore_label is not None:
        valid &= target != int(ignore_label)
    target_safe = target.clone()
    target_safe[~valid] = 0
    return target_safe.long(), valid


class BalanceLoss(nn.Module):
    """Multiclass adaptation of Balance Loss (Inter-CBL + Intra-CBL)."""

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.9,
        alpha: float = 0.5,
        inter_start_epoch: int = 50,
        ignore_label: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.inter_start_epoch = int(inter_start_epoch)
        self.ignore_label = ignore_label
        self.eps = float(eps)
        self.current_epoch = 0
        self.last_debug = {}

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _zero(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.sum() * 0.0

    def _inter_cbl(
        self,
        per_pixel_ce: torch.Tensor,
        flat_target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        counts = torch.stack([(valid & (flat_target == c)).sum() for c in range(self.num_classes)])
        nonzero_counts = counts[counts > 0]
        if nonzero_counts.numel() == 0:
            return per_pixel_ce.sum() * 0.0
        target_count = int(nonzero_counts.min().item())
        terms = []
        for class_idx in range(self.num_classes):
            class_ce = per_pixel_ce[valid & (flat_target == class_idx)]
            if class_ce.numel() == 0:
                continue
            sorted_ce = torch.sort(class_ce, descending=True).values
            k = max(1, min(target_count, sorted_ce.numel()))
            terms.append(sorted_ce[:k].mean())
        if not terms:
            return per_pixel_ce.sum() * 0.0
        return torch.stack(terms).mean()

    def _intra_cbl(
        self,
        per_pixel_ce: torch.Tensor,
        true_probs: torch.Tensor,
        flat_target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        for class_idx in range(self.num_classes):
            class_mask = valid & (flat_target == class_idx)
            if not bool(class_mask.any()):
                continue
            class_ce = per_pixel_ce[class_mask]
            class_probs = true_probs[class_mask]
            parts = []
            hard = class_ce[class_probs <= self.threshold]
            easy = class_ce[class_probs > self.threshold]
            if hard.numel() > 0:
                parts.append(hard.mean())
            if easy.numel() > 0:
                parts.append(easy.mean())
            if parts:
                terms.append(torch.stack(parts).sum())
        if not terms:
            return per_pixel_ce.sum() * 0.0
        return torch.stack(terms).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_safe, valid_mask = prepare_dense_logit_target(logits, target, self.num_classes, self.ignore_label)
        if not bool(valid_mask.any()):
            return self._zero(logits)

        ce_map = F.cross_entropy(logits, target_safe, reduction="none")
        true_probs = torch.softmax(logits, dim=1).gather(1, target_safe[:, None])[:, 0]
        flat_valid = valid_mask.reshape(-1)
        flat_target = target_safe.reshape(-1)
        per_pixel_ce = ce_map.reshape(-1)
        true_probs = true_probs.reshape(-1)

        intra_loss = self._intra_cbl(per_pixel_ce, true_probs, flat_target, flat_valid)
        inter_loss = self._inter_cbl(per_pixel_ce, flat_target, flat_valid)
        if self.current_epoch < self.inter_start_epoch:
            total = intra_loss
        else:
            total = self.alpha * intra_loss + (1.0 - self.alpha) * inter_loss
        self.last_debug = {
            "loss_intra_cbl": intra_loss.detach(),
            "loss_inter_cbl": inter_loss.detach(),
            "balance_alpha": torch.as_tensor(self.alpha, device=logits.device),
            "balance_inter_enabled": torch.as_tensor(float(self.current_epoch >= self.inter_start_epoch), device=logits.device),
        }
        return total


class AdaptiveRegionSpecificLoss(nn.Module):
    """PyTorch adaptation of Adaptive Region-Specific Tversky Loss."""

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        num_regions_per_axis: int | Sequence[int] = (16, 16, 16),
        do_bg: bool = False,
        batch_dice: bool = True,
        A: float = 0.3,
        B: float = 0.4,
        ignore_label: int | None = None,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)
        self.num_regions_per_axis = num_regions_per_axis
        self.do_bg = bool(do_bg)
        self.batch_dice = bool(batch_dice)
        self.A = float(A)
        self.B = float(B)
        self.ignore_label = ignore_label

    def _regions_for_dim(self, dim: int) -> tuple[int, ...]:
        regions = self.num_regions_per_axis
        if isinstance(regions, int):
            return (int(regions),) * dim
        regions = tuple(int(v) for v in regions)
        if len(regions) == dim:
            return regions
        if len(regions) == 3 and dim == 2:
            return regions[-2:]
        raise ValueError(f"num_regions_per_axis={regions} is incompatible with spatial dim {dim}")

    def _pool(self, x: torch.Tensor, output_size: tuple[int, ...]) -> torch.Tensor:
        if len(output_size) == 2:
            return F.adaptive_avg_pool2d(x, output_size)
        if len(output_size) == 3:
            return F.adaptive_avg_pool3d(x, output_size)
        raise ValueError(f"AdaptiveRegionSpecificLoss supports 2D/3D only, got {len(output_size)}D")

    def _pool_sum(self, x: torch.Tensor, output_size: tuple[int, ...]) -> torch.Tensor:
        """Pool regional sums while retaining adaptive partition boundaries."""
        pooled = self._pool(x, output_size)
        region_volume = torch.ones(output_size, device=x.device, dtype=x.dtype)
        for axis, (input_size, output_size_axis) in enumerate(zip(x.shape[2:], output_size)):
            sizes = [
                ((idx + 1) * input_size + output_size_axis - 1) // output_size_axis
                - (idx * input_size) // output_size_axis
                for idx in range(output_size_axis)
            ]
            shape = [1] * len(output_size)
            shape[axis] = output_size_axis
            region_volume = region_volume * torch.as_tensor(sizes, device=x.device, dtype=x.dtype).reshape(shape)
        return pooled * region_volume.reshape((1, 1, *output_size))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_safe, valid_mask = prepare_dense_logit_target(logits, target, self.num_classes, self.ignore_label)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target_safe, num_classes=self.num_classes).movedim(-1, 1).to(dtype=probs.dtype)
        valid = valid_mask[:, None].to(dtype=probs.dtype)
        probs = probs * valid
        one_hot = one_hot * valid
        if not self.do_bg:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]

        tp = probs * one_hot
        fp = probs * (1.0 - one_hot) * valid
        fn = (1.0 - probs) * one_hot
        regions = self._regions_for_dim(logits.ndim - 2)
        region_tp = self._pool_sum(tp, regions)
        region_fp = self._pool_sum(fp, regions)
        region_fn = self._pool_sum(fn, regions)
        if self.batch_dice:
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)
        tversky = (region_tp + self.smooth) / (
            region_tp + alpha * region_fp + beta * region_fn + self.smooth
        )
        return (1.0 - tversky).mean()


class BaselinePlusAdaptiveRegionSpecificLoss(nn.Module):
    """Baseline DiceCE augmented with adaptive region-specific Tversky loss."""

    def __init__(self, baseline_loss: nn.Module, adaptive_loss: nn.Module, adaptive_weight: float = 1.0):
        super().__init__()
        self.baseline_loss = baseline_loss
        self.adaptive_loss = adaptive_loss
        self.adaptive_weight = float(adaptive_weight)
        self.last_debug = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_baseline = self.baseline_loss(logits, target)
        loss_adaptive = self.adaptive_loss(logits, target)
        loss_total = loss_baseline + self.adaptive_weight * loss_adaptive
        self.last_debug = {
            "loss_baseline": loss_baseline.detach(),
            "loss_adaptive": loss_adaptive.detach(),
            "adaptive_weight": logits.new_tensor(self.adaptive_weight),
            "loss_total": loss_total.detach(),
        }
        return loss_total


class UnifiedFocalLoss(nn.Module):
    """Multiclass PyTorch port of asymmetric Unified Focal Loss."""

    def __init__(
        self,
        num_classes: int,
        weight: float | None = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        smooth: float = 1e-6,
        ignore_label: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        self.num_classes = int(num_classes)
        self.weight = None if weight is None else float(weight)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.ignore_label = ignore_label
        self.eps = float(eps)

    def _asymmetric_focal_ce(
        self,
        probs: torch.Tensor,
        target_safe: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        true_probs = probs.gather(1, target_safe[:, None])[:, 0].clamp(self.eps, 1.0 - self.eps)
        ce = -true_probs.log()
        background = target_safe == 0
        pixel_loss = torch.where(
            background,
            (1.0 - self.delta) * (1.0 - true_probs).pow(self.gamma) * ce,
            torch.full_like(ce, self.delta) * ce,
        )
        return pixel_loss[valid_mask].mean()

    def _asymmetric_focal_tversky(
        self,
        probs: torch.Tensor,
        target_safe: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask[:, None].to(dtype=probs.dtype)
        one_hot = F.one_hot(target_safe, num_classes=self.num_classes).movedim(-1, 1).to(dtype=probs.dtype)
        probs = probs * valid
        one_hot = one_hot * valid
        spatial_axes = tuple(range(2, probs.ndim))
        tp = (one_hot * probs).sum(dim=spatial_axes)
        fn = (one_hot * (1.0 - probs)).sum(dim=spatial_axes)
        fp = ((1.0 - one_hot) * probs * valid).sum(dim=spatial_axes)
        dice_class = (tp + self.smooth) / (tp + self.delta * fn + (1.0 - self.delta) * fp + self.smooth)
        dice_loss = 1.0 - dice_class
        if self.num_classes > 1:
            foreground = dice_loss[:, 1:].clamp_min(self.eps).pow(1.0 - self.gamma)
            combined = torch.cat((dice_loss[:, :1], foreground), dim=1)
        else:
            combined = dice_loss
        return combined.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_safe, valid_mask = prepare_dense_logit_target(logits, target, self.num_classes, self.ignore_label)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        probs = torch.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        focal_tversky = self._asymmetric_focal_tversky(probs, target_safe, valid_mask)
        focal_ce = self._asymmetric_focal_ce(probs, target_safe, valid_mask)
        if self.weight is None:
            return focal_tversky + focal_ce
        return self.weight * focal_tversky + (1.0 - self.weight) * focal_ce


class PixelCenterLoss(nn.Module):
    """Pixel-wise adaptation of Center Loss (Wen et al., ECCV 2016)."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        class_balanced: bool = False,
        ignore_label: int | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.class_balanced = bool(class_balanced)
        self.ignore_label = ignore_label
        self.centers = nn.Parameter(torch.empty(self.num_classes, self.feature_dim))
        nn.init.kaiming_normal_(self.centers, a=1e-2)

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_features, flat_target, valid = prepare_dense_feature_target(
            features, target, self.num_classes, self.ignore_label
        )
        if not bool(valid.any()):
            return features.sum() * 0.0

        selected_features = flat_features[valid]
        selected_target = flat_target[valid]
        diffs_sq = (selected_features - self.centers[selected_target]).square().sum(dim=1)
        if not self.class_balanced:
            return diffs_sq.mean()

        losses = []
        for class_idx in range(self.num_classes):
            class_mask = selected_target == class_idx
            if bool(class_mask.any()):
                losses.append(diffs_sq[class_mask].mean())
        if not losses:
            return features.sum() * 0.0
        return torch.stack(losses).mean()


class PixelSupConLoss(nn.Module):
    """Sampled dense supervised contrastive loss (Khosla et al., NeurIPS 2020)."""

    def __init__(
        self,
        num_classes: int,
        max_samples_per_class: int = 256,
        temperature: float = 0.1,
        ignore_label: int | None = None,
        foreground_labels: Iterable[int] | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_samples_per_class = int(max_samples_per_class)
        self.temperature = float(temperature)
        self.ignore_label = ignore_label
        self.foreground_labels = None if foreground_labels is None else tuple(int(v) for v in foreground_labels)

    def _sample_balanced(self, flat_features: torch.Tensor, flat_target: torch.Tensor, valid: torch.Tensor):
        sampled_indices = []
        class_range = self.foreground_labels if self.foreground_labels is not None else range(self.num_classes)
        for class_idx in class_range:
            class_indices = torch.where(valid & (flat_target == int(class_idx)))[0]
            if class_indices.numel() == 0:
                continue
            if class_indices.numel() > self.max_samples_per_class:
                perm = torch.randperm(class_indices.numel(), device=class_indices.device)[: self.max_samples_per_class]
                class_indices = class_indices[perm]
            sampled_indices.append(class_indices)
        if not sampled_indices:
            return None, None
        indices = torch.cat(sampled_indices, dim=0)
        return flat_features[indices], flat_target[indices]

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_features, flat_target, valid = prepare_dense_feature_target(
            features, target, self.num_classes, self.ignore_label
        )
        sampled_features, sampled_target = self._sample_balanced(flat_features, flat_target, valid)
        if sampled_features is None or sampled_features.shape[0] < 3:
            return features.sum() * 0.0

        z = F.normalize(sampled_features.float(), dim=1)
        labels = sampled_target
        logits = torch.matmul(z, z.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        positive = (labels[:, None] == labels[None, :]) & (~eye)
        valid_row = positive.any(dim=1)
        if not bool(valid_row.any()):
            return features.sum() * 0.0

        logits_mask = ~eye
        exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (log_prob * positive.to(dtype=logits.dtype)).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return -mean_log_prob_pos[valid_row].mean()


def set_dropout_train_mode(module: nn.Module) -> int:
    """Enable dropout layers while keeping the rest of a model in its current mode."""
    dropout_types = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
    enabled = 0
    for child in module.modules():
        if isinstance(child, dropout_types):
            child.train()
            enabled += 1
    return enabled
