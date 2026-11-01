from __future__ import annotations

import torch
import torch.nn as nn

from nnunetv2.training.loss.reproduction_baselines import prepare_dense_feature_target


class PaperCenterLoss(nn.Module):
    """Dense Center Loss with the mini-batch center update from Wen et al."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        alpha: float = 0.5,
        ignore_label: int | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or feature_dim <= 0:
            raise ValueError("num_classes must exceed one and feature_dim must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.ignore_label = ignore_label
        self.register_buffer("centers", torch.zeros(self.num_classes, self.feature_dim))

    def _valid_features(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_features, flat_target, valid = prepare_dense_feature_target(
            features,
            target,
            self.num_classes,
            self.ignore_label,
        )
        return flat_features[valid].float(), flat_target[valid]

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        selected_features, selected_target = self._valid_features(features, target)
        if selected_features.numel() == 0:
            return features.float().sum() * 0.0
        selected_centers = self.centers.float()[selected_target]
        return 0.5 * (selected_features - selected_centers).square().sum(dim=1).mean()

    @torch.no_grad()
    def batch_update_statistics(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_features, selected_target = self._valid_features(features.detach(), target)
        feature_sums = torch.zeros_like(self.centers, dtype=torch.float32)
        counts = torch.zeros(self.num_classes, device=self.centers.device, dtype=torch.float32)
        if selected_features.numel() == 0:
            return feature_sums, counts
        feature_sums.index_add_(0, selected_target, selected_features)
        counts.index_add_(0, selected_target, torch.ones_like(selected_target, dtype=torch.float32))
        return feature_sums, counts

    @torch.no_grad()
    def apply_center_update(self, feature_sums: torch.Tensor, counts: torch.Tensor) -> None:
        if tuple(feature_sums.shape) != tuple(self.centers.shape):
            raise ValueError("feature_sums must match the center tensor")
        if tuple(counts.shape) != (self.num_classes,):
            raise ValueError("counts must contain one value per class")
        active = counts > 0
        if not bool(active.any()):
            return
        delta = (
            counts[:, None] * self.centers.float() - feature_sums.float()
        ) / (1.0 + counts[:, None])
        self.centers[active] -= self.alpha * delta[active].to(dtype=self.centers.dtype)
