from __future__ import annotations

import math

import torch


def diagonal_gmm_component_scores(
    feat: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    log_pi: torch.Tensor,
    valid_component_mask: torch.Tensor,
    include_log2pi: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return diagonal Gaussian component log scores for flattened features."""
    if feat.ndim != 2:
        raise ValueError(f"feat must have shape [N, D], got {tuple(feat.shape)}")
    if mu.ndim != 3 or var.shape != mu.shape:
        raise ValueError("mu and var must have shape [C, K, D]")
    if log_pi.shape != mu.shape[:2] or valid_component_mask.shape != mu.shape[:2]:
        raise ValueError("log_pi and valid_component_mask must have shape [C, K]")
    if feat.shape[1] != mu.shape[2]:
        raise ValueError("feature dimension does not match GMM means")

    dtype = feat.dtype
    device = feat.device
    mu = mu.to(device=device, dtype=dtype)
    var = var.to(device=device, dtype=dtype).clamp_min(float(eps))
    log_pi = log_pi.to(device=device, dtype=dtype)
    valid_component_mask = valid_component_mask.to(device=device)
    classes, components, dim = mu.shape

    inv_var = torch.reciprocal(var)
    mu_inv_var = mu * inv_var
    quad_weight = inv_var.reshape(classes * components, dim)
    linear_weight = mu_inv_var.reshape(classes * components, dim)
    constant = (mu.square() * inv_var).sum(dim=-1) + torch.log(var).sum(dim=-1)
    mahalanobis = feat.square().matmul(quad_weight.t())
    mahalanobis = mahalanobis - 2.0 * feat.matmul(linear_weight.t())
    mahalanobis = mahalanobis + constant.reshape(1, classes * components)
    scores = log_pi.reshape(1, classes * components) - 0.5 * mahalanobis
    if include_log2pi:
        scores = scores - 0.5 * float(dim) * math.log(2.0 * math.pi)
    scores = scores.view(feat.shape[0], classes, components)
    return scores.masked_fill(~valid_component_mask[None, :, :], float("-inf"))


def diagonal_gmm_logits(
    feat: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    log_pi: torch.Tensor,
    valid_component_mask: torch.Tensor,
    include_log2pi: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = diagonal_gmm_component_scores(
        feat, mu, var, log_pi, valid_component_mask, include_log2pi, eps
    )
    return torch.logsumexp(scores, dim=-1), scores
