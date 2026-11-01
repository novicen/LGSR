from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np
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
    """Return per-component diagonal-Gaussian log scores.

    Args:
        feat: flattened features, shape [N, D].
        mu: component means, shape [C, K, D].
        var: diagonal variances, shape [C, K, D].
        log_pi: class-internal mixture log weights, shape [C, K].
        valid_component_mask: True for valid components, shape [C, K].

    Returns:
        Tensor with shape [N, C, K].
    """
    if feat.ndim != 2:
        raise ValueError(f"feat must have shape [N, D], got {tuple(feat.shape)}")
    if mu.ndim != 3 or var.shape != mu.shape:
        raise ValueError(f"mu/var must have matching shape [C, K, D], got {tuple(mu.shape)} and {tuple(var.shape)}")
    if log_pi.shape != mu.shape[:2]:
        raise ValueError(f"log_pi must have shape {tuple(mu.shape[:2])}, got {tuple(log_pi.shape)}")
    if valid_component_mask.shape != mu.shape[:2]:
        raise ValueError(
            f"valid_component_mask must have shape {tuple(mu.shape[:2])}, got {tuple(valid_component_mask.shape)}"
        )
    if feat.shape[1] != mu.shape[2]:
        raise ValueError(f"feature dim mismatch: feat={feat.shape[1]}, mu={mu.shape[2]}")

    dtype = feat.dtype
    device = feat.device
    mu = mu.to(device=device, dtype=dtype)
    var = var.to(device=device, dtype=dtype).clamp_min(float(eps))
    log_pi = log_pi.to(device=device, dtype=dtype)
    valid_component_mask = valid_component_mask.to(device=device)

    # Compute the diagonal Mahalanobis distance without materializing
    # [N, C, K, D]. For 2D nnU-Net batches that tensor is several GiB.
    c, k, d = mu.shape
    inv_var = torch.reciprocal(var)
    mu_inv_var = mu * inv_var
    quad_weight = inv_var.reshape(c * k, d)
    linear_weight = mu_inv_var.reshape(c * k, d)
    const = (mu.square() * inv_var).sum(dim=-1) + torch.log(var).sum(dim=-1)
    mahalanobis = feat.square().matmul(quad_weight.t()) - 2.0 * feat.matmul(linear_weight.t())
    mahalanobis = mahalanobis + const.reshape(1, c * k)
    scores = log_pi.reshape(1, c * k) - 0.5 * mahalanobis
    if include_log2pi:
        scores = scores - 0.5 * float(feat.shape[1]) * math.log(2.0 * math.pi)
    scores = scores.view(feat.shape[0], c, k)
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
    component_scores = diagonal_gmm_component_scores(
        feat,
        mu,
        var,
        log_pi,
        valid_component_mask,
        include_log2pi=include_log2pi,
        eps=eps,
    )
    return torch.logsumexp(component_scores, dim=-1), component_scores


def _diagonal_trust_region_minimum(
    quadratic: np.ndarray,
    linear: np.ndarray,
    constant: float,
    radius: float,
    tol: float,
) -> tuple[float, np.ndarray]:
    """Globally minimize a diagonal quadratic over an L2 ball."""
    if radius <= 0.0:
        z = np.zeros_like(linear)
        return float(constant), z

    min_quadratic = float(np.min(quadratic))
    positive = quadratic > tol
    null = np.abs(quadratic) <= tol
    if min_quadratic >= -tol and not np.any(null & (np.abs(linear) > tol)):
        unconstrained = np.zeros_like(linear)
        unconstrained[positive] = -linear[positive] / quadratic[positive]
        if float(np.linalg.norm(unconstrained)) <= radius * (1.0 + tol):
            value = (
                np.dot(quadratic * unconstrained, unconstrained)
                + 2.0 * np.dot(linear, unconstrained)
                + constant
            )
            return float(value), unconstrained

    lower = max(0.0, -min_quadratic)
    denominator = quadratic + lower
    singular = np.abs(denominator) <= tol
    nonsingular = ~singular
    if np.any(singular) and not np.any(np.abs(linear[singular]) > tol):
        hard_case = np.zeros_like(linear)
        hard_case[nonsingular] = -linear[nonsingular] / denominator[nonsingular]
        hard_norm_sq = float(np.dot(hard_case, hard_case))
        if hard_norm_sq <= radius * radius * (1.0 + tol):
            first_singular = int(np.flatnonzero(singular)[0])
            hard_case[first_singular] = math.sqrt(max(radius * radius - hard_norm_sq, 0.0))
            value = (
                np.dot(quadratic * hard_case, hard_case)
                + 2.0 * np.dot(linear, hard_case)
                + constant
            )
            return float(value), hard_case

    offset = max(1e-12, 1e-12 * (1.0 + abs(lower)))
    lo = lower + offset

    def norm_sq(lagrange: float) -> float:
        denom = quadratic + lagrange
        return float(np.sum(np.square(linear / denom)))

    hi = max(1.0, lo * 2.0)
    target = radius * radius
    while norm_sq(hi) > target:
        hi = max(hi * 2.0, hi + 1.0)
        if hi > 1e16:
            raise RuntimeError("failed to bracket diagonal trust-region multiplier")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if norm_sq(mid) > target:
            lo = mid
        else:
            hi = mid
    z = -linear / (quadratic + hi)
    value = np.dot(quadratic * z, z) + 2.0 * np.dot(linear, z) + constant
    return float(value), z


def _closest_diagonal_quadratic_zero(
    quadratic: np.ndarray,
    linear: np.ndarray,
    constant: float,
    tol: float,
) -> tuple[bool, float, np.ndarray]:
    """Return the first L2 ball touching q(z)=0, if that boundary exists."""
    if constant <= tol:
        return True, 0.0, np.zeros_like(linear)

    min_quadratic = float(np.min(quadratic))
    if min_quadratic >= -tol:
        null = np.abs(quadratic) <= tol
        if not np.any(null & (np.abs(linear) > tol)):
            positive = quadratic > tol
            global_minimizer = np.zeros_like(linear)
            global_minimizer[positive] = -linear[positive] / quadratic[positive]
            global_minimum = (
                np.dot(quadratic * global_minimizer, global_minimizer)
                + 2.0 * np.dot(linear, global_minimizer)
                + constant
            )
            if global_minimum > tol:
                return False, float("inf"), np.full_like(linear, np.nan)
            hi = max(float(np.linalg.norm(global_minimizer)), 1e-8)
        else:
            hi = 1.0
    else:
        hi = 1.0

    hi_value, hi_point = _diagonal_trust_region_minimum(quadratic, linear, constant, hi, tol)
    for _ in range(80):
        if hi_value <= 0.0:
            break
        hi *= 2.0
        hi_value, hi_point = _diagonal_trust_region_minimum(quadratic, linear, constant, hi, tol)
    else:
        raise RuntimeError("failed to bracket a diagonal-Gaussian score boundary")

    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mid_value, _ = _diagonal_trust_region_minimum(quadratic, linear, constant, mid, tol)
        if mid_value > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    _, boundary_point = _diagonal_trust_region_minimum(quadratic, linear, constant, hi, tol)
    return True, float(hi), boundary_point


def solve_diagonal_gaussian_pair_boundary(
    mu_a: torch.Tensor,
    var_a: torch.Tensor,
    log_pi_a: torch.Tensor,
    mu_b: torch.Tensor,
    var_b: torch.Tensor,
    log_pi_b: torch.Tensor,
    margin: float = 0.0,
    include_log2pi: bool = True,
    eps: float = 1e-8,
    tol: float = 1e-8,
) -> dict[str, torch.Tensor | bool]:
    """Find the maximum component-a density on ``S_a = S_b + margin``.

    The problem is whitened by component a. The resulting diagonal quadratic
    boundary is intersected with expanding L2 balls, and each ball subproblem
    is solved globally as a diagonal trust-region problem.
    """
    reference = mu_a
    tensors = (mu_a, var_a, mu_b, var_b)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("mu_a, var_a, mu_b and var_b must be one-dimensional")
    if not (mu_a.shape == var_a.shape == mu_b.shape == var_b.shape):
        raise ValueError("pair-boundary means and variances must have matching shapes")

    mu_a_np = mu_a.detach().cpu().double().numpy()
    mu_b_np = mu_b.detach().cpu().double().numpy()
    var_a_np = np.maximum(var_a.detach().cpu().double().numpy(), float(eps))
    var_b_np = np.maximum(var_b.detach().cpu().double().numpy(), float(eps))
    log_pi_a_value = float(log_pi_a.detach().cpu())
    log_pi_b_value = float(log_pi_b.detach().cpu())
    if not (math.isfinite(log_pi_a_value) and math.isfinite(log_pi_b_value)):
        raise ValueError("valid components must have finite log mixture weights")

    dim = int(mu_a_np.size)
    log2pi_term = dim * math.log(2.0 * math.pi) if include_log2pi else 0.0
    peak_a = log_pi_a_value - 0.5 * (
        float(np.log(var_a_np).sum()) + log2pi_term
    )
    peak_b = log_pi_b_value - 0.5 * (
        float(np.log(var_b_np).sum()) + log2pi_term
    )
    delta = mu_a_np - mu_b_np
    sqrt_var_a = np.sqrt(var_a_np)
    quadratic = var_a_np / var_b_np - 1.0
    linear = delta * sqrt_var_a / var_b_np
    constant = float(np.sum(np.square(delta) / var_b_np)) + 2.0 * (
        peak_a - peak_b - float(margin)
    )
    center_gap = 0.5 * constant

    scalar_kwargs = {"device": reference.device, "dtype": reference.dtype}
    if center_gap <= tol:
        return {
            "center_dominates": False,
            "boundary_exists": True,
            "mahalanobis_radius": torch.zeros((), **scalar_kwargs),
            "threshold": torch.full((), float("inf"), **scalar_kwargs),
            "boundary_point": mu_a.detach().clone(),
            "boundary_residual": torch.as_tensor(center_gap, **scalar_kwargs),
        }

    boundary_exists, radius, boundary_z = _closest_diagonal_quadratic_zero(
        quadratic=quadratic,
        linear=linear,
        constant=constant,
        tol=float(tol),
    )
    if not boundary_exists:
        return {
            "center_dominates": True,
            "boundary_exists": False,
            "mahalanobis_radius": torch.full((), float("inf"), **scalar_kwargs),
            "threshold": torch.full((), float("-inf"), **scalar_kwargs),
            "boundary_point": torch.full_like(reference, float("nan")),
            "boundary_residual": torch.zeros((), **scalar_kwargs),
        }

    point_np = mu_a_np + sqrt_var_a * boundary_z
    threshold = peak_a - 0.5 * radius * radius
    residual = (
        np.dot(quadratic * boundary_z, boundary_z)
        + 2.0 * np.dot(linear, boundary_z)
        + constant
    ) * 0.5
    return {
        "center_dominates": True,
        "boundary_exists": True,
        "mahalanobis_radius": torch.as_tensor(radius, **scalar_kwargs),
        "threshold": torch.as_tensor(threshold, **scalar_kwargs),
        "boundary_point": torch.as_tensor(point_np, **scalar_kwargs),
        "boundary_residual": torch.as_tensor(residual, **scalar_kwargs),
    }


def compute_full_diagonal_gmm_safe_geometry(
    mu: torch.Tensor,
    var: torch.Tensor,
    log_pi: torch.Tensor,
    valid_component_mask: torch.Tensor,
    mode: str,
    rates: Sequence[float],
    include_log2pi: bool = True,
    eps: float = 1e-8,
    tol: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute per-component full diagonal-GMM safe density thresholds."""
    if mode not in ("pairwise", "strict"):
        raise ValueError(f"mode must be 'pairwise' or 'strict', got {mode!r}")
    if mu.ndim != 3 or var.shape != mu.shape:
        raise ValueError("mu and var must have matching shape [C, K, D]")
    if log_pi.shape != mu.shape[:2] or valid_component_mask.shape != mu.shape[:2]:
        raise ValueError("log_pi and valid_component_mask must match mu[:2]")
    if len(rates) != int(mu.shape[0]):
        raise ValueError("rates length must match the number of classes")

    device = mu.device
    dtype = mu.dtype
    valid = valid_component_mask.detach().cpu().bool()
    thresholds = torch.full(mu.shape[:2], float("inf"), dtype=dtype, device=device)
    radii = torch.zeros(mu.shape[:2], dtype=dtype, device=device)
    safe_components = valid_component_mask.to(device=device).clone()

    for class_idx in range(int(mu.shape[0])):
        for comp_idx in torch.where(valid[class_idx])[0].tolist():
            limiting_radius = float("inf")
            component_is_safe = True
            for other_class in range(int(mu.shape[0])):
                if other_class == class_idx:
                    continue
                other_indices = torch.where(valid[other_class])[0].tolist()
                margin = math.log(float(len(other_indices))) if mode == "strict" and other_indices else 0.0
                for other_comp in other_indices:
                    result = solve_diagonal_gaussian_pair_boundary(
                        mu[class_idx, comp_idx],
                        var[class_idx, comp_idx],
                        log_pi[class_idx, comp_idx],
                        mu[other_class, other_comp],
                        var[other_class, other_comp],
                        log_pi[other_class, other_comp],
                        margin=margin,
                        include_log2pi=include_log2pi,
                        eps=eps,
                        tol=tol,
                    )
                    if not bool(result["center_dominates"]):
                        component_is_safe = False
                        limiting_radius = 0.0
                        break
                    if bool(result["boundary_exists"]):
                        limiting_radius = min(limiting_radius, float(result["mahalanobis_radius"]))
                if not component_is_safe:
                    break

            safe_components[class_idx, comp_idx] = component_is_safe
            if not component_is_safe:
                continue
            scaled_radius = limiting_radius * float(rates[class_idx])
            radii[class_idx, comp_idx] = scaled_radius
            peak = float(log_pi[class_idx, comp_idx].detach().cpu()) - 0.5 * (
                float(torch.log(var[class_idx, comp_idx].detach().cpu().clamp_min(eps)).sum())
                + (int(mu.shape[-1]) * math.log(2.0 * math.pi) if include_log2pi else 0.0)
            )
            thresholds[class_idx, comp_idx] = (
                float("-inf") if math.isinf(scaled_radius) else peak - 0.5 * scaled_radius * scaled_radius
            )

    return {
        "thresholds": thresholds,
        "mahalanobis_radii": radii,
        "safe_component_mask": safe_components,
    }


@torch.no_grad()
def summarize_diag_gmm_components(
    mu: torch.Tensor,
    var: torch.Tensor,
    log_pi: torch.Tensor,
    valid_component_mask: torch.Tensor,
    components_per_class: Sequence[int],
    samples_by_class: Mapping[int, torch.Tensor] | None = None,
    nk: torch.Tensor | None = None,
    class_counts: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Summarize class-internal diagonal-GMM component usage.

    The summary is diagnostic only. It reports parameter-level quantities
    (mixture weights, variance and component distances), and optionally sample
    assignment quantities if bounded samples are provided.
    """
    if len(components_per_class) != int(mu.shape[0]):
        raise ValueError("components_per_class length must match number of classes")
    if tuple(var.shape) != tuple(mu.shape):
        raise ValueError("mu and var must have matching shape")
    if tuple(log_pi.shape) != tuple(mu.shape[:2]):
        raise ValueError("log_pi shape must match mu[:2]")
    if tuple(valid_component_mask.shape) != tuple(mu.shape[:2]):
        raise ValueError("valid_component_mask shape must match mu[:2]")

    mu = mu.detach().cpu().float()
    var = var.detach().cpu().float()
    log_pi = log_pi.detach().cpu().float()
    valid_component_mask = valid_component_mask.detach().cpu().bool()
    nk_cpu = nk.detach().cpu().float() if nk is not None else None
    class_counts_cpu = class_counts.detach().cpu().float() if class_counts is not None else None

    summary: Dict[int, Dict[str, torch.Tensor]] = {}
    for class_idx, num_components in enumerate(components_per_class):
        k = int(num_components)
        valid = valid_component_mask[class_idx, :k]
        valid_indices = torch.where(valid)[0]
        if valid_indices.numel() == 0:
            continue

        mu_c = mu[class_idx, :k]
        var_c = var[class_idx, :k]
        log_pi_c = log_pi[class_idx, :k]
        diag: Dict[str, torch.Tensor] = {
            "pi": torch.exp(log_pi_c[valid_indices]),
            "var_mean": var_c[valid_indices].mean(dim=1),
        }

        if valid_indices.numel() > 1:
            pairdist = torch.cdist(mu_c[valid_indices], mu_c[valid_indices])
            upper = pairdist[torch.triu(torch.ones_like(pairdist, dtype=torch.bool), diagonal=1)]
            diag["pairdist_min"] = upper.min()
            diag["pairdist_mean"] = upper.mean()
        else:
            diag["pairdist_min"] = torch.tensor(0.0)
            diag["pairdist_mean"] = torch.tensor(0.0)

        if nk_cpu is not None:
            denom = nk_cpu[class_idx, :k].sum()
            if class_counts_cpu is not None and class_idx < int(class_counts_cpu.numel()):
                denom = class_counts_cpu[class_idx]
            denom = denom.clamp_min(float(eps))
            diag["nk_frac"] = nk_cpu[class_idx, valid_indices] / denom

        if samples_by_class is not None and class_idx in samples_by_class:
            samples = samples_by_class[class_idx].detach().cpu().float()
            if samples.numel() > 0:
                scores = diagonal_gmm_component_scores(
                    samples,
                    mu_c.unsqueeze(0),
                    var_c.unsqueeze(0),
                    log_pi_c.unsqueeze(0),
                    valid.unsqueeze(0),
                    include_log2pi=True,
                    eps=eps,
                )[:, 0, :k]
                resp = torch.softmax(scores, dim=1)
                hard = resp.argmax(dim=1)
                diag["resp_mean"] = resp[:, valid_indices].mean(dim=0)
                diag["hard_frac"] = torch.stack([(hard == int(idx)).float().mean() for idx in valid_indices])

        summary[class_idx] = diag
    return summary


def class_balanced_gmm_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    feature_dim: int | None = None,
    per_dim: bool = False,
) -> torch.Tensor:
    """Class-balanced true-class NLL from class log-likelihood logits."""
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError("target with logits ndim must have singleton channel dimension")
        target = target[:, 0]
    target = target.long()
    active = active.bool()
    safe_target = torch.where(active, target, torch.zeros_like(target))
    ell_y = torch.gather(logits, dim=1, index=safe_target.unsqueeze(1)).squeeze(1)
    nll = -ell_y
    class_values = []
    for class_idx in range(logits.shape[1]):
        class_mask = active & (target == class_idx)
        if class_mask.any():
            value = nll[class_mask].mean()
            if per_dim:
                if feature_dim is None or int(feature_dim) <= 0:
                    raise ValueError("feature_dim must be positive when per_dim=True")
                value = value / float(feature_dim)
            class_values.append(value)
    if not class_values:
        return logits.sum() * 0.0
    return torch.stack(class_values).mean()


def balanced_sinkhorn_assignment(
    scores: torch.Tensor,
    temperature: float = 0.2,
    num_iters: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return row-normalized balanced assignments from [N, K] component scores.

    The returned matrix has row sums close to 1 and column sums close to N / K.
    This is the SwAV-style Sinkhorn normalization adapted to per-class pixels.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N, K], got {tuple(scores.shape)}")
    n, k = int(scores.shape[0]), int(scores.shape[1])
    if n <= 0 or k <= 0:
        raise ValueError("scores must have positive N and K")
    if float(temperature) <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    dtype = scores.dtype
    logits = (scores.float() / float(temperature))
    logits = logits - logits.max()
    q = torch.exp(logits).t().contiguous()
    q = q.clamp_min(float(eps))
    q = q / q.sum().clamp_min(float(eps))

    for _ in range(max(int(num_iters), 1)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(float(eps))
        q = q / float(k)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(float(eps))
        q = q / float(n)

    return (q * float(n)).t().to(dtype=dtype)


@torch.no_grad()
def sinkhorn_hard_assignment(
    scores: torch.Tensor,
    epsilon: float = 0.05,
    num_iters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GMMSeg-style equipartition Sinkhorn followed by hard assignment."""
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N,K], got {tuple(scores.shape)}")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        empty_indices = torch.empty(scores.shape[0], dtype=torch.long, device=scores.device)
        return torch.zeros_like(scores), empty_indices, torch.zeros_like(scores)
    if float(epsilon) <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if int(num_iters) <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")

    dtype = scores.dtype
    normalized = scores.float()
    normalized = normalized / normalized.abs().amax().clamp_min(torch.finfo(normalized.dtype).eps)
    q = torch.exp(normalized / float(epsilon)).t().contiguous()
    num_components, num_samples = q.shape
    q = q / q.sum().clamp_min(torch.finfo(q.dtype).tiny)

    for _ in range(int(num_iters)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        q = q / float(num_components)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        q = q / float(num_samples)

    soft = (q * float(num_samples)).t().to(dtype=dtype)
    indices = soft.argmax(dim=1)
    hard = torch.nn.functional.one_hot(indices, num_classes=num_components).to(dtype=dtype)
    return hard, indices, soft


def init_balanced_mean_streaming_stats(num_classes: int, max_components: int, feat_dim: int) -> Dict[str, torch.Tensor]:
    """Create CPU accumulators for balanced-assignment mean estimation."""
    if int(num_classes) <= 0 or int(max_components) <= 0 or int(feat_dim) <= 0:
        raise ValueError("num_classes, max_components and feat_dim must be positive")
    return {
        "nk": torch.zeros(int(num_classes), int(max_components), dtype=torch.float64),
        "sum": torch.zeros(int(num_classes), int(max_components), int(feat_dim), dtype=torch.float64),
        "class_counts": torch.zeros(int(num_classes), dtype=torch.float64),
        "skipped_classes": torch.zeros(int(num_classes), dtype=torch.float64),
    }


@torch.no_grad()
def accumulate_balanced_mean_streaming_stats(
    stats: Dict[str, torch.Tensor],
    feat: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    mu: torch.Tensor,
    valid_component_mask: torch.Tensor,
    components_per_class: Sequence[int],
    temperature: float = 0.2,
    sinkhorn_iters: int = 50,
    min_pixels_per_component: int = 1,
) -> Dict[str, torch.Tensor]:
    """Accumulate balanced class-internal component means for one feature batch."""
    if feat.ndim != 2:
        raise ValueError(f"feat must be [N, D], got {tuple(feat.shape)}")
    target = target.reshape(-1).long()
    active = active.reshape(-1).bool()
    if target.shape[0] != feat.shape[0] or active.shape[0] != feat.shape[0]:
        raise ValueError("feat, target and active must have matching first dimension")
    if len(components_per_class) != mu.shape[0]:
        raise ValueError("components_per_class length must match number of classes")
    if stats["sum"].shape[-1] != feat.shape[1]:
        raise ValueError(f"feature dim mismatch: stats={stats['sum'].shape[-1]}, feat={feat.shape[1]}")

    mu = mu.detach().to(device=feat.device, dtype=feat.dtype)
    valid_component_mask = valid_component_mask.detach().to(device=feat.device, dtype=torch.bool)
    min_pixels_per_component = max(int(min_pixels_per_component), 1)

    for class_idx, num_components in enumerate(components_per_class):
        k = int(num_components)
        class_mask = active & (target == int(class_idx))
        if not class_mask.any():
            continue
        class_feat = feat[class_mask]
        if class_feat.shape[0] < k * min_pixels_per_component:
            stats["skipped_classes"][class_idx] += 1.0
            continue

        valid = valid_component_mask[class_idx, :k]
        if int(valid.sum().item()) != k:
            raise ValueError("balanced mean expects the first K components to be valid for each class")
        if k == 1:
            assignment = torch.ones(class_feat.shape[0], 1, device=class_feat.device, dtype=class_feat.dtype)
        else:
            scores = -0.5 * torch.cdist(class_feat, mu[class_idx, :k]).square()
            assignment = balanced_sinkhorn_assignment(
                scores,
                temperature=float(temperature),
                num_iters=int(sinkhorn_iters),
            )

        assignment64 = assignment.to(dtype=torch.float64)
        class_feat64 = class_feat.to(dtype=torch.float64)
        stats["nk"][class_idx, :k] += assignment64.sum(dim=0).detach().cpu()
        stats["sum"][class_idx, :k] += assignment64.t().matmul(class_feat64).detach().cpu()
        stats["class_counts"][class_idx] += float(class_feat.shape[0])
    return stats


def finalize_balanced_mean_streaming_stats(
    stats: Mapping[str, torch.Tensor],
    components_per_class: Sequence[int],
    valid_component_mask: torch.Tensor,
    initial_mu: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finalize balanced mean accumulators into component means."""
    nk_stats = stats["nk"].detach().cpu().double()
    sum_stats = stats["sum"].detach().cpu().double()
    class_counts = stats["class_counts"].detach().cpu().double()
    num_classes, max_components, feat_dim = sum_stats.shape
    if len(components_per_class) != int(num_classes):
        raise ValueError("components_per_class length must match stats")
    if tuple(valid_component_mask.shape) != (int(num_classes), int(max_components)):
        raise ValueError("valid_component_mask shape mismatch")
    if tuple(initial_mu.shape) != (int(num_classes), int(max_components), int(feat_dim)):
        raise ValueError("initial_mu shape mismatch")

    mu_out = initial_mu.detach().cpu().float().clone()
    nk_out = torch.zeros(num_classes, max_components, dtype=torch.float32)
    valid_component_mask = valid_component_mask.detach().cpu().bool()
    for class_idx, num_components in enumerate(components_per_class):
        k = int(num_components)
        if class_counts[class_idx] <= 0:
            continue
        nk = nk_stats[class_idx, :k].clamp_min(float(eps))
        mu_out[class_idx, :k] = (sum_stats[class_idx, :k] / nk[:, None]).float()
        nk_out[class_idx, :k] = nk.float()

    mu_out = mu_out.masked_fill(~valid_component_mask[:, :, None], 0.0)
    return mu_out, nk_out, class_counts.float()


def _farthest_point_kmeans_init(samples: torch.Tensor, k: int) -> torch.Tensor:
    if samples.shape[0] == 0:
        raise ValueError("cannot initialize KMeans from empty samples")
    if samples.shape[0] == 1:
        return samples.repeat(k, 1)
    first_idx = torch.argmin(samples[:, 0])
    centers = [samples[first_idx]]
    min_dist = torch.cdist(samples, centers[0].unsqueeze(0)).squeeze(1).square()
    for _ in range(1, k):
        next_idx = torch.argmax(min_dist)
        centers.append(samples[next_idx])
        new_dist = torch.cdist(samples, centers[-1].unsqueeze(0)).squeeze(1).square()
        min_dist = torch.minimum(min_dist, new_dist)
    return torch.stack(centers, dim=0)


def _kmeans(samples: torch.Tensor, k: int, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    centers = _farthest_point_kmeans_init(samples, k)
    labels = torch.zeros(samples.shape[0], dtype=torch.long, device=samples.device)
    for _ in range(max(int(iters), 1)):
        dist = torch.cdist(samples, centers).square()
        labels = dist.argmin(dim=1)
        new_centers = centers.clone()
        for comp_idx in range(k):
            mask = labels == comp_idx
            if mask.any():
                new_centers[comp_idx] = samples[mask].mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels


def _estimate_from_responsibilities(
    samples: torch.Tensor,
    resp: torch.Tensor,
    var_min: float,
    var_max: float,
    dirichlet_alpha: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nk = resp.sum(dim=0).clamp_min(float(eps))
    mu = resp.t().matmul(samples) / nk[:, None]
    second = resp.t().matmul(samples.square()) / nk[:, None]
    var = (second - mu.square()).clamp(min=float(var_min), max=float(var_max))
    pi = (nk + float(dirichlet_alpha)) / (float(samples.shape[0]) + float(resp.shape[1]) * float(dirichlet_alpha))
    log_pi = torch.log(pi.clamp_min(float(eps)))
    return mu, var, log_pi, nk


def _uniform_log_pi(k: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.full((int(k),), -math.log(float(k)), dtype=dtype)


def balanced_greedy_hard_assignment(scores: torch.Tensor) -> torch.Tensor:
    """Balanced one-hot assignment with per-component capacity constraints.

    This is intended for diagnostic/Stage-B GMM updates, not backpropagation.
    Unlike soft balanced assignments, it cannot average every component over
    the same pixels when component scores are nearly tied.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N, K], got {tuple(scores.shape)}")
    n, k = int(scores.shape[0]), int(scores.shape[1])
    if n <= 0 or k <= 0:
        raise ValueError("scores must have positive N and K")
    device = scores.device
    dtype = scores.dtype
    scores_cpu = scores.detach().cpu().float()
    capacities = torch.full((k,), n // k, dtype=torch.long)
    capacities[: n % k] += 1
    counts = torch.zeros(k, dtype=torch.long)
    assigned = torch.full((n,), -1, dtype=torch.long)

    order = torch.argsort(scores_cpu.reshape(-1), descending=True)
    remaining = n
    for flat_idx in order.tolist():
        sample_idx = flat_idx // k
        comp_idx = flat_idx % k
        if assigned[sample_idx] >= 0 or counts[comp_idx] >= capacities[comp_idx]:
            continue
        assigned[sample_idx] = comp_idx
        counts[comp_idx] += 1
        remaining -= 1
        if remaining == 0:
            break

    if remaining > 0:
        available = counts < capacities
        for sample_idx in torch.where(assigned < 0)[0].tolist():
            candidate_scores = scores_cpu[sample_idx].clone()
            candidate_scores[~available] = float("-inf")
            comp_idx = int(candidate_scores.argmax())
            assigned[sample_idx] = comp_idx
            counts[comp_idx] += 1
            available = counts < capacities

    return torch.nn.functional.one_hot(assigned.to(device=device), num_classes=k).to(dtype=dtype)


def _estimate_from_balanced_hard_assignment(
    samples: torch.Tensor,
    scores: torch.Tensor,
    var_min: float,
    var_max: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resp = balanced_greedy_hard_assignment(scores)
    mu, var, _, nk = _estimate_from_responsibilities(
        samples,
        resp,
        var_min=var_min,
        var_max=var_max,
        dirichlet_alpha=1.0,
        eps=eps,
    )
    log_pi = _uniform_log_pi(resp.shape[1], dtype=samples.dtype)
    return mu, var, log_pi, nk


def init_diag_gmm_streaming_stats(num_classes: int, max_components: int, feat_dim: int) -> Dict[str, torch.Tensor]:
    """Create CPU accumulators for one streaming diagonal-GMM M-step."""
    if int(num_classes) <= 0 or int(max_components) <= 0 or int(feat_dim) <= 0:
        raise ValueError("num_classes, max_components and feat_dim must be positive")
    return {
        "nk": torch.zeros(int(num_classes), int(max_components), dtype=torch.float64),
        "sum": torch.zeros(int(num_classes), int(max_components), int(feat_dim), dtype=torch.float64),
        "sumsq": torch.zeros(int(num_classes), int(max_components), int(feat_dim), dtype=torch.float64),
        "class_counts": torch.zeros(int(num_classes), dtype=torch.float64),
    }


@torch.no_grad()
def accumulate_diag_gmm_streaming_stats(
    stats: Dict[str, torch.Tensor],
    feat: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    log_pi: torch.Tensor,
    valid_component_mask: torch.Tensor,
    components_per_class: Sequence[int],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Accumulate one batch of E-step responsibilities for a diagonal GMM.

    The accumulators stay on CPU, while responsibility computation follows the
    device of ``feat``. This keeps GPU memory bounded by the current batch.
    """
    if feat.ndim != 2:
        raise ValueError(f"feat must be [N, D], got {tuple(feat.shape)}")
    target = target.reshape(-1).long()
    active = active.reshape(-1).bool()
    if target.shape[0] != feat.shape[0] or active.shape[0] != feat.shape[0]:
        raise ValueError("feat, target and active must have matching first dimension")
    if len(components_per_class) != mu.shape[0]:
        raise ValueError("components_per_class length must match number of classes")
    if stats["sum"].shape[-1] != feat.shape[1]:
        raise ValueError(f"feature dim mismatch: stats={stats['sum'].shape[-1]}, feat={feat.shape[1]}")

    mu = mu.detach().to(device=feat.device, dtype=feat.dtype)
    var = var.detach().to(device=feat.device, dtype=feat.dtype)
    log_pi = log_pi.detach().to(device=feat.device, dtype=feat.dtype)
    valid_component_mask = valid_component_mask.detach().to(device=feat.device, dtype=torch.bool)

    for class_idx, num_components in enumerate(components_per_class):
        class_mask = active & (target == int(class_idx))
        if not class_mask.any():
            continue
        class_feat = feat[class_mask]
        k = int(num_components)
        valid = valid_component_mask[class_idx : class_idx + 1, :k]
        scores = diagonal_gmm_component_scores(
            class_feat,
            mu[class_idx : class_idx + 1, :k],
            var[class_idx : class_idx + 1, :k],
            log_pi[class_idx : class_idx + 1, :k],
            valid,
            include_log2pi=True,
            eps=eps,
        )[:, 0, :k]
        resp = torch.softmax(scores, dim=1).to(dtype=torch.float64)
        class_feat64 = class_feat.to(dtype=torch.float64)
        stats["nk"][class_idx, :k] += resp.sum(dim=0).detach().cpu()
        stats["sum"][class_idx, :k] += resp.t().matmul(class_feat64).detach().cpu()
        stats["sumsq"][class_idx, :k] += resp.t().matmul(class_feat64.square()).detach().cpu()
        stats["class_counts"][class_idx] += float(class_feat.shape[0])
    return stats


@torch.no_grad()
def accumulate_diag_gmm_streaming_stats_sinkhorn_hard(
    stats: Dict[str, torch.Tensor],
    feat: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    log_pi: torch.Tensor,
    valid_component_mask: torch.Tensor,
    components_per_class: Sequence[int],
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 3,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Accumulate one streaming Sinkhorn-hard M-step for a diagonal GMM."""
    if feat.ndim != 2:
        raise ValueError(f"feat must be [N, D], got {tuple(feat.shape)}")
    target = target.reshape(-1).long()
    active = active.reshape(-1).bool()
    if target.shape[0] != feat.shape[0] or active.shape[0] != feat.shape[0]:
        raise ValueError("feat, target and active must have matching first dimension")
    if len(components_per_class) != mu.shape[0]:
        raise ValueError("components_per_class length must match number of classes")
    if stats["sum"].shape[-1] != feat.shape[1]:
        raise ValueError(f"feature dim mismatch: stats={stats['sum'].shape[-1]}, feat={feat.shape[1]}")

    mu = mu.detach().to(device=feat.device, dtype=feat.dtype)
    var = var.detach().to(device=feat.device, dtype=feat.dtype)
    log_pi = log_pi.detach().to(device=feat.device, dtype=feat.dtype)
    valid_component_mask = valid_component_mask.detach().to(device=feat.device, dtype=torch.bool)

    for class_idx, num_components in enumerate(components_per_class):
        class_mask = active & (target == int(class_idx))
        if not class_mask.any():
            continue
        class_feat = feat[class_mask]
        k = int(num_components)
        valid = valid_component_mask[class_idx : class_idx + 1, :k]
        if k == 1:
            resp = torch.ones(class_feat.shape[0], 1, device=class_feat.device, dtype=class_feat.dtype)
        else:
            scores = diagonal_gmm_component_scores(
                class_feat,
                mu[class_idx : class_idx + 1, :k],
                var[class_idx : class_idx + 1, :k],
                log_pi[class_idx : class_idx + 1, :k],
                valid,
                include_log2pi=True,
                eps=eps,
            )[:, 0, :k]
            resp, _, _ = sinkhorn_hard_assignment(
                scores,
                epsilon=float(sinkhorn_epsilon),
                num_iters=int(sinkhorn_iters),
            )

        resp64 = resp.to(dtype=torch.float64)
        class_feat64 = class_feat.to(dtype=torch.float64)
        stats["nk"][class_idx, :k] += resp64.sum(dim=0).detach().cpu()
        stats["sum"][class_idx, :k] += resp64.t().matmul(class_feat64).detach().cpu()
        stats["sumsq"][class_idx, :k] += resp64.t().matmul(class_feat64.square()).detach().cpu()
        stats["class_counts"][class_idx] += float(class_feat.shape[0])
    return stats


def finalize_diag_gmm_streaming_stats(
    stats: Mapping[str, torch.Tensor],
    components_per_class: Sequence[int],
    valid_component_mask: torch.Tensor,
    var_min: float = 0.1,
    var_max: float = 10.0,
    dirichlet_alpha: float = 1.0,
    initial_mu: torch.Tensor | None = None,
    initial_var: torch.Tensor | None = None,
    initial_log_pi: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finalize streaming diagonal-GMM accumulators into mu/var/log_pi."""
    nk_stats = stats["nk"].detach().cpu().double()
    sum_stats = stats["sum"].detach().cpu().double()
    sumsq_stats = stats["sumsq"].detach().cpu().double()
    class_counts = stats["class_counts"].detach().cpu().double()
    num_classes, max_components, feat_dim = sum_stats.shape
    if len(components_per_class) != int(num_classes):
        raise ValueError("components_per_class length must match stats")
    if tuple(valid_component_mask.shape) != (int(num_classes), int(max_components)):
        raise ValueError("valid_component_mask shape mismatch")

    dtype = torch.float32
    mu_out = torch.zeros(num_classes, max_components, feat_dim, dtype=dtype)
    var_out = torch.ones_like(mu_out)
    log_pi_out = torch.full((num_classes, max_components), float("-inf"), dtype=dtype)
    nk_out = torch.zeros(num_classes, max_components, dtype=dtype)
    valid_component_mask = valid_component_mask.detach().cpu().bool()

    for class_idx, num_components in enumerate(components_per_class):
        k = int(num_components)
        if class_counts[class_idx] <= 0:
            if initial_mu is not None:
                mu_out[class_idx, :k] = initial_mu[class_idx, :k].detach().cpu().float()
            if initial_var is not None:
                var_out[class_idx, :k] = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
            if initial_log_pi is not None:
                log_pi_out[class_idx, :k] = initial_log_pi[class_idx, :k].detach().cpu().float()
            else:
                log_pi_out[class_idx, :k] = -math.log(float(k))
            continue

        nk = nk_stats[class_idx, :k].clamp_min(float(eps))
        mu = sum_stats[class_idx, :k] / nk[:, None]
        second = sumsq_stats[class_idx, :k] / nk[:, None]
        var = (second - mu.square()).clamp(min=float(var_min), max=float(var_max))
        pi = (nk + float(dirichlet_alpha)) / (class_counts[class_idx] + float(k) * float(dirichlet_alpha))
        mu_out[class_idx, :k] = mu.float()
        var_out[class_idx, :k] = var.float()
        log_pi_out[class_idx, :k] = torch.log(pi.clamp_min(float(eps))).float()
        nk_out[class_idx, :k] = nk.float()

    log_pi_out = log_pi_out.masked_fill(~valid_component_mask, float("-inf"))
    return mu_out, var_out, log_pi_out, nk_out, class_counts.float()


def estimate_diag_gmm_from_samples(
    samples_by_class: Mapping[int, torch.Tensor],
    num_classes: int,
    max_components: int,
    components_per_class: Sequence[int],
    valid_component_mask: torch.Tensor,
    kmeans_iters: int = 10,
    em_iters: int = 3,
    var_min: float = 0.1,
    var_max: float = 10.0,
    dirichlet_alpha: float = 1.0,
    initial_mu: torch.Tensor | None = None,
    initial_var: torch.Tensor | None = None,
    initial_log_pi: torch.Tensor | None = None,
    use_kmeans_init: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate class-wise diagonal GMM parameters from bounded CPU samples."""
    if len(components_per_class) != int(num_classes):
        raise ValueError("components_per_class length must match num_classes")
    if tuple(valid_component_mask.shape) != (int(num_classes), int(max_components)):
        raise ValueError("valid_component_mask shape mismatch")

    dtype = torch.float32
    device = torch.device("cpu")
    feat_dim = None
    for samples in samples_by_class.values():
        if samples.numel() > 0:
            feat_dim = int(samples.shape[1])
            break
    if feat_dim is None:
        if initial_mu is None:
            raise ValueError("Cannot infer feature dimension from empty samples without initial_mu")
        feat_dim = int(initial_mu.shape[-1])

    mu = torch.zeros(num_classes, max_components, feat_dim, dtype=dtype, device=device)
    var = torch.ones_like(mu)
    log_pi = torch.full((num_classes, max_components), float("-inf"), dtype=dtype, device=device)
    nk_out = torch.zeros(num_classes, max_components, dtype=dtype, device=device)

    valid_component_mask = valid_component_mask.to(device=device, dtype=torch.bool)
    for class_idx in range(num_classes):
        k = int(components_per_class[class_idx])
        valid = valid_component_mask[class_idx, :k]
        samples = samples_by_class.get(class_idx)
        if samples is None or samples.numel() == 0:
            if initial_mu is not None:
                mu[class_idx, :k] = initial_mu[class_idx, :k].detach().cpu().float()
            if initial_var is not None:
                var[class_idx, :k] = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
            if initial_log_pi is not None:
                log_pi[class_idx, :k] = initial_log_pi[class_idx, :k].detach().cpu().float()
            else:
                log_pi[class_idx, :k] = -math.log(float(k))
            continue
        samples = samples.detach().cpu().float()
        if samples.shape[1] != feat_dim:
            raise ValueError(f"class {class_idx} feature dim mismatch")

        if use_kmeans_init or initial_mu is None:
            centers, labels = _kmeans(samples, k, kmeans_iters)
            resp = torch.nn.functional.one_hot(labels, num_classes=k).float()
            mu_c, var_c, log_pi_c, nk = _estimate_from_responsibilities(
                samples, resp, var_min, var_max, dirichlet_alpha, eps
            )
        else:
            mu_c = initial_mu[class_idx, :k].detach().cpu().float()
            var_c = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
            log_pi_c = initial_log_pi[class_idx, :k].detach().cpu().float()
            nk = torch.zeros(k, dtype=dtype)

        for _ in range(max(int(em_iters), 0)):
            scores = diagonal_gmm_component_scores(
                samples,
                mu_c.unsqueeze(0),
                var_c.unsqueeze(0),
                log_pi_c.unsqueeze(0),
                valid.unsqueeze(0),
                include_log2pi=True,
                eps=eps,
            )[:, 0, :k]
            resp = torch.softmax(scores, dim=1)
            mu_c, var_c, log_pi_c, nk = _estimate_from_responsibilities(
                samples, resp, var_min, var_max, dirichlet_alpha, eps
            )

        mu[class_idx, :k] = mu_c
        var[class_idx, :k] = var_c
        log_pi[class_idx, :k] = log_pi_c
        nk_out[class_idx, :k] = nk

    log_pi = log_pi.masked_fill(~valid_component_mask, float("-inf"))
    return mu, var, log_pi, nk_out


def estimate_diag_gmm_from_samples_balanced(
    samples_by_class: Mapping[int, torch.Tensor],
    num_classes: int,
    max_components: int,
    components_per_class: Sequence[int],
    valid_component_mask: torch.Tensor,
    kmeans_iters: int = 10,
    em_iters: int = 3,
    var_min: float = 0.5,
    var_max: float = 10.0,
    initial_mu: torch.Tensor | None = None,
    initial_var: torch.Tensor | None = None,
    use_kmeans_init: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate class-wise diagonal GMM with balanced hard assignments.

    This deliberately differs from maximum-likelihood EM: mixture weights are
    kept uniform and each component receives a near-equal assignment budget.
    The goal is stable class-internal component coverage, not unconstrained
    density fitting.
    """
    if len(components_per_class) != int(num_classes):
        raise ValueError("components_per_class length must match num_classes")
    if tuple(valid_component_mask.shape) != (int(num_classes), int(max_components)):
        raise ValueError("valid_component_mask shape mismatch")

    dtype = torch.float32
    device = torch.device("cpu")
    feat_dim = None
    for samples in samples_by_class.values():
        if samples.numel() > 0:
            feat_dim = int(samples.shape[1])
            break
    if feat_dim is None:
        if initial_mu is None:
            raise ValueError("Cannot infer feature dimension from empty samples without initial_mu")
        feat_dim = int(initial_mu.shape[-1])

    mu = torch.zeros(num_classes, max_components, feat_dim, dtype=dtype, device=device)
    var = torch.ones_like(mu).clamp(var_min, var_max)
    log_pi = torch.full((num_classes, max_components), float("-inf"), dtype=dtype, device=device)
    nk_out = torch.zeros(num_classes, max_components, dtype=dtype, device=device)
    valid_component_mask = valid_component_mask.to(device=device, dtype=torch.bool)

    for class_idx in range(num_classes):
        k = int(components_per_class[class_idx])
        valid = valid_component_mask[class_idx, :k]
        samples = samples_by_class.get(class_idx)
        uniform_log_pi = _uniform_log_pi(k, dtype=dtype)
        if samples is None or samples.numel() == 0:
            if initial_mu is not None:
                mu[class_idx, :k] = initial_mu[class_idx, :k].detach().cpu().float()
            if initial_var is not None:
                var[class_idx, :k] = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
            log_pi[class_idx, :k] = uniform_log_pi
            continue
        samples = samples.detach().cpu().float()
        if samples.shape[1] != feat_dim:
            raise ValueError(f"class {class_idx} feature dim mismatch")

        if use_kmeans_init or initial_mu is None:
            centers, labels = _kmeans(samples, k, kmeans_iters)
            resp = torch.nn.functional.one_hot(labels, num_classes=k).float()
            mu_c, var_c, _, nk = _estimate_from_responsibilities(
                samples, resp, var_min, var_max, dirichlet_alpha=1.0, eps=eps
            )
        else:
            mu_c = initial_mu[class_idx, :k].detach().cpu().float()
            if initial_var is not None:
                var_c = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
            else:
                var_c = torch.ones(k, feat_dim, dtype=dtype).clamp(var_min, var_max)
            nk = torch.zeros(k, dtype=dtype)
        log_pi_c = uniform_log_pi

        for _ in range(max(int(em_iters), 0)):
            scores = diagonal_gmm_component_scores(
                samples,
                mu_c.unsqueeze(0),
                var_c.unsqueeze(0),
                log_pi_c.unsqueeze(0),
                valid.unsqueeze(0),
                include_log2pi=True,
                eps=eps,
            )[:, 0, :k]
            mu_c, var_c, log_pi_c, nk = _estimate_from_balanced_hard_assignment(
                samples,
                scores,
                var_min=var_min,
                var_max=var_max,
                eps=eps,
            )

        mu[class_idx, :k] = mu_c
        var[class_idx, :k] = var_c
        log_pi[class_idx, :k] = uniform_log_pi
        nk_out[class_idx, :k] = nk

    log_pi = log_pi.masked_fill(~valid_component_mask, float("-inf"))
    return mu, var, log_pi, nk_out


def estimate_diag_gmm_from_samples_sinkhorn_hard(
    samples_by_class: Mapping[int, torch.Tensor],
    num_classes: int,
    max_components: int,
    components_per_class: Sequence[int],
    valid_component_mask: torch.Tensor,
    kmeans_iters: int = 10,
    em_iters: int = 3,
    var_min: float = 0.1,
    var_max: float = 10.0,
    dirichlet_alpha: float = 1.0,
    initial_mu: torch.Tensor | None = None,
    initial_var: torch.Tensor | None = None,
    initial_log_pi: torch.Tensor | None = None,
    use_kmeans_init: bool = True,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate class-wise diagonal GMM using Sinkhorn-balanced hard assignment."""
    if len(components_per_class) != int(num_classes):
        raise ValueError("components_per_class length must match num_classes")
    if tuple(valid_component_mask.shape) != (int(num_classes), int(max_components)):
        raise ValueError("valid_component_mask shape mismatch")

    dtype = torch.float32
    device = torch.device("cpu")
    feat_dim = None
    for samples in samples_by_class.values():
        if samples.numel() > 0:
            feat_dim = int(samples.shape[1])
            break
    if feat_dim is None:
        if initial_mu is None:
            raise ValueError("Cannot infer feature dimension from empty samples without initial_mu")
        feat_dim = int(initial_mu.shape[-1])

    mu = torch.zeros(num_classes, max_components, feat_dim, dtype=dtype, device=device)
    var = torch.ones_like(mu)
    log_pi = torch.full((num_classes, max_components), float("-inf"), dtype=dtype, device=device)
    nk_out = torch.zeros(num_classes, max_components, dtype=dtype, device=device)
    valid_component_mask = valid_component_mask.to(device=device, dtype=torch.bool)

    for class_idx in range(num_classes):
        k = int(components_per_class[class_idx])
        valid = valid_component_mask[class_idx, :k]
        samples = samples_by_class.get(class_idx)
        if samples is None or samples.numel() == 0:
            if initial_mu is not None:
                mu[class_idx, :k] = initial_mu[class_idx, :k].detach().cpu().float()
            if initial_var is not None:
                var[class_idx, :k] = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
            if initial_log_pi is not None:
                log_pi[class_idx, :k] = initial_log_pi[class_idx, :k].detach().cpu().float()
            else:
                log_pi[class_idx, :k] = -math.log(float(k))
            continue
        samples = samples.detach().cpu().float()
        if samples.shape[1] != feat_dim:
            raise ValueError(f"class {class_idx} feature dim mismatch")

        if k == 1:
            resp = torch.ones(samples.shape[0], 1, dtype=dtype)
            mu_c, var_c, log_pi_c, nk = _estimate_from_responsibilities(
                samples, resp, var_min, var_max, dirichlet_alpha, eps
            )
        else:
            if use_kmeans_init or initial_mu is None:
                _, labels = _kmeans(samples, k, kmeans_iters)
                resp = torch.nn.functional.one_hot(labels, num_classes=k).float()
                mu_c, var_c, log_pi_c, nk = _estimate_from_responsibilities(
                    samples, resp, var_min, var_max, dirichlet_alpha, eps
                )
            else:
                mu_c = initial_mu[class_idx, :k].detach().cpu().float()
                var_c = initial_var[class_idx, :k].detach().cpu().float().clamp(var_min, var_max)
                log_pi_c = initial_log_pi[class_idx, :k].detach().cpu().float()
                nk = torch.zeros(k, dtype=dtype)

            for _ in range(max(int(em_iters), 0)):
                scores = diagonal_gmm_component_scores(
                    samples,
                    mu_c.unsqueeze(0),
                    var_c.unsqueeze(0),
                    log_pi_c.unsqueeze(0),
                    valid.unsqueeze(0),
                    include_log2pi=True,
                    eps=eps,
                )[:, 0, :k]
                resp, _, _ = sinkhorn_hard_assignment(
                    scores,
                    epsilon=float(sinkhorn_epsilon),
                    num_iters=int(sinkhorn_iters),
                )
                mu_c, var_c, log_pi_c, nk = _estimate_from_responsibilities(
                    samples, resp, var_min, var_max, dirichlet_alpha, eps
                )

        mu[class_idx, :k] = mu_c
        var[class_idx, :k] = var_c
        log_pi[class_idx, :k] = log_pi_c
        nk_out[class_idx, :k] = nk

    log_pi = log_pi.masked_fill(~valid_component_mask, float("-inf"))
    return mu, var, log_pi, nk_out
