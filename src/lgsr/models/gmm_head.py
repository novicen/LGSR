from __future__ import annotations

import math

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lgsr.geometry.gmm_scores import diagonal_gmm_logits


class LGSRHead(nn.Module):
    """State-compatible class-conditional GMM head used by LGSR."""

    def __init__(self, feat_dim: int, num_classes: int = 3, components_per_class=(1, 3, 3)):
        super().__init__()
        if num_classes != len(components_per_class):
            raise ValueError("num_classes must match components_per_class")
        self.num_classes = int(num_classes)
        self.components_per_class = tuple(int(v) for v in components_per_class)
        self.max_components = max(self.components_per_class)
        self.mu = nn.Parameter(torch.zeros(self.num_classes, self.max_components, feat_dim))
        valid = torch.zeros(self.num_classes, self.max_components, dtype=torch.bool)
        log_alpha = torch.full((self.num_classes, self.max_components), float("-inf"))
        for class_idx, count in enumerate(self.components_per_class):
            valid[class_idx, :count] = True
            log_alpha[class_idx, :count] = -math.log(float(count))
        self.register_buffer("valid_component_mask", valid, persistent=True)
        self.register_buffer("log_alpha", log_alpha, persistent=True)
        self.register_buffer("gmm_var", torch.ones(self.num_classes, self.max_components, feat_dim), persistent=True)
        self.register_buffer("gmm_log_pi", log_alpha.clone(), persistent=True)
        self.register_buffer("gmm_ready", torch.zeros((), dtype=torch.bool), persistent=True)
        trunc_normal_(self.mu, std=0.02)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in ("gmm_var", "gmm_log_pi", "gmm_ready"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def set_consistent_gmm(self, mu: torch.Tensor, var: torch.Tensor, log_pi: torch.Tensor):
        if tuple(mu.shape) != tuple(self.mu.shape) or tuple(var.shape) != tuple(self.gmm_var.shape):
            raise ValueError("GMM parameter shape mismatch")
        if tuple(log_pi.shape) != tuple(self.gmm_log_pi.shape):
            raise ValueError("GMM mixture-weight shape mismatch")
        self.mu.copy_(mu.to(self.mu))
        self.gmm_var.copy_(var.to(self.gmm_var).clamp_min(1e-8))
        self.gmm_log_pi.copy_(log_pi.to(self.gmm_log_pi))
        self.gmm_log_pi.masked_fill_(~self.valid_component_mask, float("-inf"))
        self.gmm_ready.fill_(True)

    def disable_consistent_gmm(self):
        self.gmm_ready.fill_(False)

    def forward(self, feat: torch.Tensor, return_components: bool = False):
        batch, dim = feat.shape[:2]
        spatial_shape = feat.shape[2:]
        flat = feat.movedim(1, -1).reshape(-1, dim)
        if bool(self.gmm_ready.item()):
            logits_flat, scores = diagonal_gmm_logits(
                flat.float(), self.mu.float(), self.gmm_var.float(), self.gmm_log_pi.float(),
                self.valid_component_mask, include_log2pi=True,
            )
            logits = logits_flat.to(feat.dtype)
            scores = scores.to(feat.dtype)
        else:
            mu_flat = self.mu.reshape(self.num_classes * self.max_components, dim)
            dist_sq = (
                flat.square().sum(dim=1, keepdim=True)
                + mu_flat.square().sum(dim=1).unsqueeze(0)
                - 2.0 * flat.matmul(mu_flat.t())
            ).clamp_min(0.0)
            scores = -0.5 * dist_sq.view(-1, self.num_classes, self.max_components) + self.log_alpha[None]
            scores = scores.masked_fill(~self.valid_component_mask[None], float("-inf"))
            logits = torch.logsumexp(scores, dim=-1)
        logits = logits.view(batch, *spatial_shape, self.num_classes).movedim(-1, 1).contiguous()
        scores = scores.view(batch, *spatial_shape, self.num_classes, self.max_components)
        scores = scores.permute(0, len(spatial_shape) + 1, len(spatial_shape) + 2, *range(1, len(spatial_shape) + 1)).contiguous()
        if not return_components:
            return logits
        return {
            "logits": logits,
            "component_scores": scores,
            "mu": self.mu,
            "log_alpha": self.gmm_log_pi if bool(self.gmm_ready.item()) else self.log_alpha,
            "gmm_var": self.gmm_var,
            "gmm_ready": self.gmm_ready,
            "valid_component_mask": self.valid_component_mask,
        }
