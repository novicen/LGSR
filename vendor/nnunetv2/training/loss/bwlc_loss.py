from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _sanitize_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4:
        target = target[:, 0]
    return target.long()


def compute_multiclass_boundary_mask(
    target: torch.Tensor,
    boundary_width: int = 1,
    ignore_label: int | None = None,
) -> torch.Tensor:
    target = _sanitize_target(target)
    kernel_size = boundary_width * 2 + 1

    target_float = target.float().unsqueeze(1)
    local_max = F.max_pool2d(target_float, kernel_size=kernel_size, stride=1, padding=boundary_width)
    local_min = -F.max_pool2d(-target_float, kernel_size=kernel_size, stride=1, padding=boundary_width)
    boundary = (local_max - local_min) > 0

    if ignore_label is not None:
        valid = target.ne(ignore_label).unsqueeze(1)
        ignore_neighborhood = F.max_pool2d(
            (~valid).float(), kernel_size=kernel_size, stride=1, padding=boundary_width
        ) > 0
        boundary = boundary & valid & (~ignore_neighborhood)

    return boundary[:, 0]


class BoundaryWeightedConsistencyLoss(nn.Module):
    def __init__(
        self,
        boundary_width: int = 1,
        boundary_factor: float = 3.0,
        ignore_label: int | None = None,
        use_kl: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.boundary_width = boundary_width
        self.boundary_factor = float(boundary_factor)
        self.ignore_label = ignore_label
        self.use_kl = use_kl
        self.temperature = float(temperature)
        self.last_debug: dict[str, float] = {}

    def forward(self, logits_1: torch.Tensor, logits_2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _sanitize_target(target)
        boundary = compute_multiclass_boundary_mask(
            target, boundary_width=self.boundary_width, ignore_label=self.ignore_label
        )

        if self.ignore_label is None:
            valid = torch.ones_like(target, dtype=logits_1.dtype)
        else:
            valid = target.ne(self.ignore_label).to(dtype=logits_1.dtype)

        weights = (1.0 + self.boundary_factor * boundary.to(dtype=logits_1.dtype)) * valid

        if self.use_kl:
            log_p1 = F.log_softmax(logits_1 / self.temperature, dim=1)
            log_p2 = F.log_softmax(logits_2 / self.temperature, dim=1)
            p1 = log_p1.exp()
            p2 = log_p2.exp()
            per_pixel = 0.5 * (
                F.kl_div(log_p1, p2, reduction="none").sum(dim=1) +
                F.kl_div(log_p2, p1, reduction="none").sum(dim=1)
            )
        else:
            per_pixel = (logits_1 - logits_2).pow(2).mean(dim=1)

        denom = torch.clamp(weights.sum(), min=1.0)
        loss = (per_pixel * weights).sum() / denom
        self.last_debug = {
            "loss_consistency": float(loss.detach().cpu()),
            "boundary_ratio": float(boundary.float().mean().detach().cpu()),
        }
        return loss
