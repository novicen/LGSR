import torch
from torch import nn, Tensor
import numpy as np
from torch import functional as F
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class RobustLogitNormCrossEntropyLoss(nn.Module):
    """
    LogitNorm CE for dense prediction.

    The logits are normalized along the class dimension before CE:
        z_hat = (z / ||z||_2) / tau
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        tau: float = 0.3,
        eps: float = 1e-6,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        norm = torch.linalg.vector_norm(input, ord=2, dim=1, keepdim=True)
        normalized_logits = input / torch.clamp(norm, min=self.eps)
        normalized_logits = normalized_logits / self.tau

        per_pixel = nn.functional.cross_entropy(
            normalized_logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        if self.reduction == "none":
            return per_pixel

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return input.sum() * 0.0

        valid_loss = per_pixel[valid_mask]
        if self.reduction == "sum":
            return valid_loss.sum()
        if self.reduction == "mean":
            return valid_loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class RobustSVLSCrossEntropyLoss(nn.Module):
    """
    Spatially Varying Label Smoothing (SVLS) cross-entropy for 2D dense prediction.

    Hard labels are converted to one-hot targets and spatially smoothed with a
    Gaussian kernel before computing soft-target CE.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        ksize: int = 3,
        sigma: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if ksize <= 0 or ksize % 2 == 0:
            raise ValueError(f"ksize must be a positive odd integer, got {ksize}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ksize = int(ksize)
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.register_buffer("svls_kernel", self._build_gaussian_kernel(self.ksize, self.sigma), persistent=False)

    @staticmethod
    def _build_gaussian_kernel(ksize: int, sigma: float) -> Tensor:
        coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"RobustSVLSCrossEntropyLoss expects 4D logits [B,C,H,W], got {tuple(input.shape)}")
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return input.sum() * 0.0

        num_classes = input.shape[1]
        safe_target = torch.where(valid_mask, target, 0)

        one_hot = nn.functional.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=input.dtype)
        valid = valid_mask.unsqueeze(1).to(dtype=input.dtype)
        one_hot = one_hot * valid

        kernel_2d = self.svls_kernel.to(device=input.device, dtype=input.dtype)
        kernel_2d = kernel_2d.view(1, 1, self.ksize, self.ksize)
        repeated_kernel = kernel_2d.repeat(num_classes, 1, 1, 1)
        padding = self.ksize // 2

        smoothed = nn.functional.conv2d(one_hot, repeated_kernel, padding=padding, groups=num_classes)
        valid_mass = nn.functional.conv2d(valid, kernel_2d, padding=padding).clamp_min(self.eps)
        soft_target = smoothed / valid_mass

        log_probs = nn.functional.log_softmax(input, dim=1)
        per_pixel = -(soft_target * log_probs).sum(dim=1)
        per_pixel = per_pixel * valid_mask.to(dtype=per_pixel.dtype)

        if self.reduction == "none":
            return per_pixel

        valid_loss = per_pixel[valid_mask]
        if self.reduction == "sum":
            return valid_loss.sum()
        if self.reduction == "mean":
            return valid_loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class HardL1ACELoss(nn.Module):
    """
    Hard-binned marginal L1 Average Calibration Error for dense segmentation.

    Each class probability map is calibrated independently against the binary
    one-vs-all target for that class, then averaged across classes.
    """

    def __init__(
        self,
        num_bins: int = 10,
        ignore_index: int = -100,
        include_background: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        self.num_bins = int(num_bins)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.reduction = reduction
        self.eps = float(eps)
        self.register_buffer("bin_edges", torch.linspace(0.0, 1.0, steps=self.num_bins + 1), persistent=False)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"HardL1ACELoss expects 4D logits [B,C,H,W], got {tuple(input.shape)}")
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return input.sum() * 0.0

        probs = nn.functional.softmax(input, dim=1)
        safe_target = torch.where(valid_mask, target, 0)
        one_hot = nn.functional.one_hot(safe_target, num_classes=input.shape[1]).permute(0, 3, 1, 2).to(dtype=probs.dtype)
        valid = valid_mask.unsqueeze(1).to(dtype=probs.dtype)
        one_hot = one_hot * valid
        probs = probs * valid

        class_indices = range(input.shape[1]) if self.include_background else range(1, input.shape[1])
        valid_count = valid_mask.sum().to(dtype=probs.dtype)
        total_loss = input.sum() * 0.0
        active_classes = 0
        bin_edges = self.bin_edges.to(device=input.device, dtype=probs.dtype)

        for class_idx in class_indices:
            class_prob = probs[:, class_idx][valid_mask]
            class_target = one_hot[:, class_idx][valid_mask]
            if class_prob.numel() == 0:
                continue

            class_loss = input.sum() * 0.0
            for bin_idx in range(self.num_bins):
                lower = bin_edges[bin_idx]
                upper = bin_edges[bin_idx + 1]
                if bin_idx == self.num_bins - 1:
                    in_bin = (class_prob >= lower) & (class_prob <= upper)
                else:
                    in_bin = (class_prob >= lower) & (class_prob < upper)
                if not in_bin.any():
                    continue

                bin_prob = class_prob[in_bin]
                bin_target = class_target[in_bin]
                bin_weight = bin_prob.numel() / valid_count
                bin_conf = bin_prob.mean()
                bin_acc = bin_target.mean()
                class_loss = class_loss + bin_weight * torch.abs(bin_conf - bin_acc)

            total_loss = total_loss + class_loss
            active_classes += 1

        if active_classes == 0:
            return input.sum() * 0.0

        if self.reduction == "sum":
            return total_loss
        if self.reduction == "mean":
            return total_loss / active_classes
        if self.reduction == "none":
            return total_loss / active_classes
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class SoftL1ACELoss(nn.Module):
    """
    Soft-binned marginal L1 Average Calibration Error for dense segmentation.

    Each probability contributes to all bins through differentiable soft
    assignment weights around fixed bin centers.
    """

    def __init__(
        self,
        num_bins: int = 10,
        ignore_index: int = -100,
        include_background: bool = True,
        reduction: str = "mean",
        sigma: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.num_bins = int(num_bins)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.reduction = reduction
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.register_buffer(
            "bin_centers",
            torch.linspace(0.5 / self.num_bins, 1.0 - 0.5 / self.num_bins, steps=self.num_bins),
            persistent=False,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"SoftL1ACELoss expects 4D logits [B,C,H,W], got {tuple(input.shape)}")
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return input.sum() * 0.0

        probs = nn.functional.softmax(input, dim=1)
        safe_target = torch.where(valid_mask, target, 0)
        one_hot = nn.functional.one_hot(safe_target, num_classes=input.shape[1]).permute(0, 3, 1, 2).to(dtype=probs.dtype)
        valid = valid_mask.unsqueeze(1).to(dtype=probs.dtype)
        one_hot = one_hot * valid
        probs = probs * valid

        class_indices = range(input.shape[1]) if self.include_background else range(1, input.shape[1])
        valid_count = valid_mask.sum().to(dtype=probs.dtype)
        total_loss = input.sum() * 0.0
        active_classes = 0
        bin_centers = self.bin_centers.to(device=input.device, dtype=probs.dtype)

        for class_idx in class_indices:
            class_prob = probs[:, class_idx][valid_mask]
            class_target = one_hot[:, class_idx][valid_mask]
            if class_prob.numel() == 0:
                continue

            distances = class_prob.unsqueeze(1) - bin_centers.unsqueeze(0)
            weights = torch.exp(-0.5 * (distances / self.sigma) ** 2)
            weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=self.eps)

            bin_mass = weights.sum(dim=0)
            bin_weight = bin_mass / valid_count
            weighted_conf = (weights * class_prob.unsqueeze(1)).sum(dim=0) / torch.clamp(bin_mass, min=self.eps)
            weighted_acc = (weights * class_target.unsqueeze(1)).sum(dim=0) / torch.clamp(bin_mass, min=self.eps)
            class_loss = (bin_weight * torch.abs(weighted_conf - weighted_acc)).sum()

            total_loss = total_loss + class_loss
            active_classes += 1

        if active_classes == 0:
            return input.sum() * 0.0

        if self.reduction == "sum":
            return total_loss
        if self.reduction == "mean":
            return total_loss / active_classes
        if self.reduction == "none":
            return total_loss / active_classes
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class MEEPLoss(nn.Module):
    """
    Maximum Entropy on Erroneous Predictions for dense segmentation.

    The auxiliary term is only applied to valid misclassified pixels. Because the
    training objective is minimized, we return the negative mean entropy so that
    minimizing the term corresponds to maximizing entropy on wrong predictions.
    """

    def __init__(
        self,
        ignore_index: int | None = -100,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"MEEPLoss expects 4D logits [B,C,H,W], got {tuple(input.shape)}")
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        probs = nn.functional.softmax(input, dim=1)
        pred = probs.argmax(dim=1)

        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index

        wrong_mask = (pred != target) & valid_mask
        if not wrong_mask.any():
            return input.sum() * 0.0

        entropy = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=1)
        wrong_entropy = entropy[wrong_mask]

        if self.reduction == "none":
            result = input.new_zeros(target.shape, dtype=entropy.dtype)
            result[wrong_mask] = -wrong_entropy
            return result
        if self.reduction == "sum":
            return -wrong_entropy.sum()
        if self.reduction == "mean":
            return -wrong_entropy.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class MultiAnchorSafeRegionCELoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        rho: float = 0.9,
        active_pixel_mixing_alpha: float = 0.25,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.rho = float(rho)
        self.active_pixel_mixing_alpha = float(active_pixel_mixing_alpha)
        self.reduction = reduction
        self.eps = float(eps)
        self.last_debug = {}

    def _compute_anchor_thresholds(self, anchors: Tensor, bias: Tensor, anchor_classes: Tensor):
        anchors_f32 = anchors.float()
        bias_f32 = bias.float()
        anchor_classes = anchor_classes.long()

        dist = torch.cdist(anchors_f32, anchors_f32, p=2)
        cross_class = anchor_classes.unsqueeze(0) != anchor_classes.unsqueeze(1)
        valid_pair = cross_class & (dist >= self.eps)

        pair_radius = torch.full_like(dist, float("inf"))
        numerator = dist.square() + bias_f32.unsqueeze(1) - bias_f32.unsqueeze(0)
        pair_radius[valid_pair] = numerator[valid_pair] / (2.0 * dist[valid_pair])

        radius_min = pair_radius.min(dim=1).values
        has_safe_ball = torch.isfinite(radius_min) & (radius_min > 0)
        thresholds = bias_f32 - (self.rho * radius_min).square()

        return thresholds.to(dtype=bias.dtype), radius_min.to(dtype=bias.dtype), has_safe_ball

    def forward(
        self,
        class_logits: Tensor,
        anchor_scores: Tensor,
        target: Tensor,
        anchors: Tensor,
        bias: Tensor,
        anchor_classes: Tensor,
    ) -> Tensor:
        if class_logits.ndim != 4:
            raise ValueError(f"MultiAnchorSafeRegionCELoss expects 4D logits [B,C,H,W], got {tuple(class_logits.shape)}")
        if anchor_scores.ndim != 4:
            raise ValueError(f"MultiAnchorSafeRegionCELoss expects 4D anchor scores [B,A,H,W], got {tuple(anchor_scores.shape)}")
        if target.ndim == class_logits.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        valid = (target >= 0) & (target < class_logits.shape[1]) & (target != self.ignore_index)
        ce_target = torch.where(valid, target, torch.full_like(target, self.ignore_index))

        per_pixel_ce = nn.functional.cross_entropy(
            class_logits,
            ce_target,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        if not valid.any():
            return class_logits.sum() * 0.0

        with torch.no_grad():
            thresholds, radius_min, has_safe_ball = self._compute_anchor_thresholds(anchors, bias, anchor_classes)
            safe_anchor = (anchor_scores >= thresholds.view(1, -1, 1, 1)) & has_safe_ball.view(1, -1, 1, 1)

            safe_class_masks = []
            for class_idx in range(class_logits.shape[1]):
                class_mask = anchor_classes == class_idx
                if torch.any(class_mask):
                    safe_class_masks.append(safe_anchor[:, class_mask].any(dim=1))
                else:
                    safe_class_masks.append(torch.zeros_like(valid))
            safe_class_masks = torch.stack(safe_class_masks, dim=1)

            safe_target = torch.where(valid, target, 0)
            safe_gt = torch.gather(safe_class_masks, 1, safe_target.unsqueeze(1)).squeeze(1)
            active = valid & (~safe_gt)

        masked = per_pixel_ce * active.to(dtype=per_pixel_ce.dtype)
        num_active = active.sum().to(dtype=per_pixel_ce.dtype)
        num_valid = valid.sum().to(dtype=per_pixel_ce.dtype)
        mixed_denominator = self.active_pixel_mixing_alpha * num_active + (1.0 - self.active_pixel_mixing_alpha) * num_valid
        mixed_denominator = torch.clamp(mixed_denominator, min=1.0)

        self.last_debug = {
            "num_active": num_active.detach(),
            "num_base": num_valid.detach(),
            "thresholds": thresholds.detach(),
            "radius_min": radius_min.detach(),
            "has_safe_ball": has_safe_ball.detach(),
        }

        if self.reduction == "mean":
            return masked.sum() / mixed_denominator
        if self.reduction == "sum":
            return masked.sum()
        if self.reduction == "none":
            return masked
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class RobustMbLSCrossEntropyLoss(nn.Module):
    """
    Margin-based Label Smoothing (MbLS) implemented as CE plus a logit-margin penalty.

    diff = max(logit) - logit
    loss = CE + alpha * mean(relu(diff - margin))
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        margin: float = 10.0,
        alpha: float = 0.1,
    ):
        super().__init__()
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.margin = float(margin)
        self.alpha = float(alpha)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        ce = nn.functional.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return input.sum() * 0.0

        diff = input.max(dim=1, keepdim=True).values - input
        margin_penalty = nn.functional.relu(diff - self.margin).mean(dim=1)
        per_pixel = ce + self.alpha * margin_penalty

        if self.reduction == "none":
            return per_pixel * valid_mask.to(dtype=per_pixel.dtype)

        valid_loss = per_pixel[valid_mask]
        if self.reduction == "sum":
            return valid_loss.sum()
        if self.reduction == "mean":
            return valid_loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")


# class RobustRelaxLoss(nn.CrossEntropyLoss):
#     """
#         input: [B, C, H, W] - 模型的原始输出 logits
#         target:[B, 1, H, W] 或 [B, H, W] - 真实标签
#         thresh:[C] - 每个类别对应的“及格线” Logit 阈值
#     """
#     def forward(self, input: Tensor, target: Tensor, thresh: Tensor, mask: Tensor, permit_label: list = []) -> Tensor:
#         # 1. 维度对齐
#         if target.ndim == input.ndim:
#             assert target.shape[1] == 1
#             target = target[:, 0]
#         target = target.long() # [B, H, W]
#         if mask.ndim != target.ndim:
#             mask = mask[:, 0]

#         ce_loss = nn.functional.cross_entropy(
#             input, target, weight=self.weight, 
#             ignore_index=self.ignore_index, reduction='none'
#         )

#         valid_target = target.clone()
#         ignore_mask = (target == self.ignore_index)
        
#         valid_target[ignore_mask] = 0 
            
#         true_logit = torch.gather(input, dim=1, index=valid_target.unsqueeze(1)).squeeze(1) #[B, H, W]
#         true_thresh = thresh[valid_target] #[B, H, W]

#         relax_mask = (true_logit <= true_thresh)  # [B, H, W]
#         permit_mask = torch.zeros_like(target, dtype=torch.bool, device=target.device)
#         for lbl in permit_label:
#             permit_mask |= (target == lbl)
#         relax_mask[permit_mask] = 1

#         # 确保 ignore_index 区域绝对不产生 Loss
#         relax_mask[ignore_mask] = 0
#         relax_mask &= mask

#         loss = ce_loss * relax_mask

#         # 6. 返回结果 (严谨的数学归约)
#         if self.reduction == 'mean':
#             # 标准 CE 的 mean 是除以【非 ignore 的总像素数】
#             # valid_pixels_count = (~ignore_mask).sum().float()
#             valid_pixels_count = relax_mask.sum().float()
#             return loss.sum() / torch.clamp(valid_pixels_count, min=1.0)
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


class RobustRelaxLoss(nn.CrossEntropyLoss):
    """
        input: [B, C, H, W] - 模型的原始输出 logits
        target:[B, 1, H, W] 或 [B, H, W] - 真实标签
        thresh:[C] - 每个类别对应的“及格线” Logit 阈值
    """
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        active_pixel_mixing_alpha: float = 0.0,
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        if not 0.0 <= active_pixel_mixing_alpha <= 1.0:
            raise ValueError(
                f"active_pixel_mixing_alpha must be in [0, 1], got {active_pixel_mixing_alpha}"
            )
        self.active_pixel_mixing_alpha = float(active_pixel_mixing_alpha)
        self.last_debug = {}

    def forward(self, input: Tensor, target: Tensor, mask: Tensor, base_mask: Tensor) -> Tensor:
        # 1. 维度对齐
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long() # [B, H, W]
        if mask.ndim != target.ndim:
            mask = mask[:, 0]
        if base_mask.ndim != target.ndim:
            base_mask = base_mask[:, 0]

        mask = mask.bool()
        base_mask = base_mask.bool()


        ce_loss = nn.functional.cross_entropy(
            input, target, weight=self.weight, 
            ignore_index=self.ignore_index, reduction='none'
        )

        loss = ce_loss * mask.to(dtype=ce_loss.dtype)
        # 6. 返回结果 (严谨的数学归约)
        num_active = mask.sum().float()
        num_base = base_mask.sum().float()
        safe_num_base = torch.clamp(num_base, min=1.0)
        mixed_denominator = (
            self.active_pixel_mixing_alpha * num_active
            + (1.0 - self.active_pixel_mixing_alpha) * num_base
        )
        safe_denominator = torch.clamp(mixed_denominator, min=1.0)
        self.last_debug = {
            'num_active': num_active.detach(),
            'num_base': num_base.detach(),
            'active_ratio': (num_active / safe_num_base).detach(),
            'mixed_denominator': safe_denominator.detach(),
        }
        if self.reduction == 'mean':
            return loss.sum() / safe_denominator
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class ErrorDrivenCELoss(nn.CrossEntropyLoss):
    """
        input: [B, C, H, W] - 模型的原始输出 logits
        target:[B, 1, H, W] 或 [B, H, W] - 真实标签
    """
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        active_pixel_mixing_alpha: float = 0.9,
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        if not 0.0 <= active_pixel_mixing_alpha <= 1.0:
            raise ValueError(
                f"active_pixel_mixing_alpha must be in [0, 1], got {active_pixel_mixing_alpha}"
            )
        self.active_pixel_mixing_alpha = float(active_pixel_mixing_alpha)
        self.last_debug = {}

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 1. 维度对齐
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long() # [B, H, W]


        ce_loss = nn.functional.cross_entropy(
            input, target, weight=self.weight, 
            ignore_index=self.ignore_index, reduction='none'
        )

        with torch.no_grad():
            # 2. 找到网络的“硬判决”结果
            preds = torch.argmax(input, dim=1) # [B, H, W]
            
            # 3. 找出所有“算错的”且“不是 ignore_index”的像素
            valid_mask = (target != self.ignore_index)
            wrong_mask = (preds != target) & valid_mask  # [B, H, W]
            num_wrong = wrong_mask.sum().float()
            num_valid = valid_mask.sum().float()
            
        masked_ce = ce_loss * wrong_mask.to(dtype=ce_loss.dtype)
        mixed_denominator = (
            self.active_pixel_mixing_alpha * num_wrong
            + (1.0 - self.active_pixel_mixing_alpha) * num_valid
        )
        safe_num_valid = torch.clamp(num_valid, min=1.0)
        safe_den = torch.clamp(mixed_denominator, min=1.0)
        self.last_debug = {
            'num_active': num_wrong.detach(),
            'num_base': num_valid.detach(),
            'active_ratio': (num_wrong / safe_num_valid).detach(),
            'mixed_denominator': safe_den.detach(),
        }

        if self.reduction == 'mean':
            return masked_ce.sum() / safe_den
        elif self.reduction == 'sum':
            return masked_ce.sum()
        else:
            return masked_ce

class RobustFocalLoss(nn.Module):
    def __init__(self, alpha: float | Tensor = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        # 难例挖掘系数（保持不变）
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        elif isinstance(alpha, Tensor):
            assert alpha.ndim == 1, "alpha张量必须是1维（长度=类别数）"
            self.alpha = alpha.float()
        else:
            raise ValueError(f"alpha必须是标量或1D张量，当前类型：{type(alpha)}")
        self.register_buffer('alpha_weight', self.alpha)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1, "target第二维必须为1"
            target = target[:, 0]
        target = target.long()  # 转为long类型（CrossEntropy要求）

        self.alpha_weight = self.alpha_weight.to(target.device)
        num_classes = input.shape[1]
        input_flat = input.permute(0, *range(2, input.ndim), 1).reshape(-1, num_classes)  # [B*H*W, C]
        target_flat = target.reshape(-1)

        if self.alpha_weight.numel() > 1:  # alpha是类别权重张量
            assert self.alpha_weight.shape[0] == num_classes, \
                f"alpha张量长度({self.alpha_weight.shape[0]})必须等于类别数({num_classes})"
        
        log_pt = torch.log_softmax(input_flat, dim=1)
        log_pt = log_pt.gather(1, target_flat.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        if self.alpha_weight.numel() == 1:
            alpha_t = self.alpha_weight[0]
        else:
            alpha_t = self.alpha_weight.gather(0, target_flat)

        # 3.5 Focal Loss核心公式：-α_t * (1-p_t)^γ * log(p_t)
        focal_loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        focal_loss = focal_loss.reshape(target.shape)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"无效的reduction模式：{self.reduction}，仅支持mean/sum/none")
        
class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
