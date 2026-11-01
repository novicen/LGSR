from __future__ import annotations

import torch


def compute_strict_radii(
    mu: torch.Tensor,
    log_alpha: torch.Tensor,
    valid_component_mask: torch.Tensor,
    rates: tuple[float, ...] = (1.0, 1.0, 1.0),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reproduce the paper trainer's strict component-radius calculation."""
    num_classes, max_components, _ = mu.shape
    rates_tensor = mu.new_tensor(rates)
    if rates_tensor.numel() != num_classes:
        raise ValueError("rates must have one value per class")
    radii = torch.zeros((num_classes, max_components), device=mu.device, dtype=mu.dtype)
    component_counts = valid_component_mask.sum(dim=1).to(dtype=mu.dtype).clamp_min(1.0)
    for class_idx in range(num_classes):
        for comp_idx in range(max_components):
            if not bool(valid_component_mask[class_idx, comp_idx]):
                continue
            candidates = []
            center = mu[class_idx, comp_idx]
            beta = log_alpha[class_idx, comp_idx]
            for other_class in range(num_classes):
                if other_class == class_idx:
                    continue
                log_k_other = torch.log(component_counts[other_class])
                for other_comp in range(max_components):
                    if not bool(valid_component_mask[other_class, other_comp]):
                        continue
                    distance = torch.linalg.vector_norm(
                        center - mu[other_class, other_comp]
                    ).clamp_min(eps)
                    beta_other = log_alpha[other_class, other_comp]
                    rho = distance * 0.5 + (beta - beta_other - log_k_other) / distance
                    candidates.append(rho)
            if candidates:
                radii[class_idx, comp_idx] = (
                    torch.stack(candidates).min().clamp_min(0.0) * rates_tensor[class_idx]
                )
    return radii


def build_certified_mask(
    logits: torch.Tensor,
    component_scores: torch.Tensor,
    features: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_alpha: torch.Tensor,
    valid_component_mask: torch.Tensor,
    rates: tuple[float, ...] = (1.0, 1.0, 1.0),
    require_prediction_match: bool = False,
    eps: float = 1e-8,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the winner-conditioned training gate used by the primary trainer."""
    if target.ndim == logits.ndim:
        target = target[:, 0]
    valid = (target >= 0) & (target < logits.shape[1]) if valid_mask is None else valid_mask.bool()
    target_safe = torch.where(valid, target, torch.zeros_like(target)).long()
    component_scores_flat = component_scores.permute(
        0, *range(3, component_scores.ndim), 1, 2
    ).reshape(-1, component_scores.shape[1], component_scores.shape[2])
    target_flat = target_safe.reshape(-1)
    valid_flat = valid.reshape(-1)
    radii = compute_strict_radii(mu, log_alpha, valid_component_mask, rates, eps)
    winner = torch.zeros_like(target_flat)
    selected_radii = torch.zeros_like(target_flat, dtype=logits.dtype)
    for class_idx in range(logits.shape[1]):
        class_mask = valid_flat & (target_flat == class_idx)
        if not class_mask.any():
            continue
        scores = component_scores_flat[class_mask, class_idx].masked_fill(
            ~valid_component_mask[class_idx].unsqueeze(0), float("-inf")
        )
        best = scores.argmax(dim=1)
        winner[class_mask] = best
        selected_radii[class_mask] = radii[class_idx, best]

    feat = features
    if tuple(feat.shape[2:]) != tuple(target.shape[1:]):
        mode = "trilinear" if feat.ndim == 5 else "bilinear"
        feat = torch.nn.functional.interpolate(feat, size=target.shape[1:], mode=mode, align_corners=False)
    feat_flat = feat.permute(0, *range(2, feat.ndim), 1).reshape(-1, feat.shape[1])
    geometry_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
    safe_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
    if valid_flat.any():
        centers = mu[target_flat[valid_flat], winner[valid_flat]]
        distance = torch.linalg.vector_norm(feat_flat[valid_flat] - centers, dim=1)
        geometry_flat[valid_flat] = distance <= selected_radii[valid_flat]
        if require_prediction_match:
            pred = logits.argmax(dim=1).reshape(-1)
            safe_flat[valid_flat] = (
                (pred[valid_flat] == target_flat[valid_flat]) & geometry_flat[valid_flat]
            )
        else:
            safe_flat[valid_flat] = geometry_flat[valid_flat]
    safe = safe_flat.view_as(valid)
    debug = {
        "safe_ratio": safe[valid].float().mean() if valid.any() else logits.sum() * 0.0,
        "outside_ratio": (~safe[valid]).float().mean() if valid.any() else logits.sum() * 0.0,
        "radii": radii.detach(),
    }
    return safe, debug
