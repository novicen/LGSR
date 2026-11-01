from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles
from medpy.metric.binary import hd95
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask, compute_tp_fp_fn_tn
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from scipy import ndimage


confidence_thresholds = [0.7, 0.8, 0.9]
normalized_liver_tumor_hce_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
normalized_global_hce_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
lesion_low_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
lesion_high_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
coverage_levels = [0.5, 0.8, 0.9, 0.95]
default_low_thr = 0.30
default_high_thr = 0.60
min_component_size = 0
connected_dim = "3d"


def _as_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denom = int(denominator.item())
    if denom == 0:
        return 0.0
    return float(numerator.item()) / float(denom)


def _safe_ratio_or_nan(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.1f}"


def _module_attr(root: torch.nn.Module, dotted_name: str):
    current = root
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _flatten_class_vectors(tensor: torch.Tensor, num_classes: int, name: str) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.ndim < 2 or tensor.shape[0] < num_classes:
        raise RuntimeError(
            f"{name} must have first dimension >= {num_classes} and at least 2 dims, got {tuple(tensor.shape)}"
        )
    tensor = tensor[:num_classes]
    return tensor.reshape(num_classes, -1)


def _find_euclidean_anchors(model: torch.nn.Module, num_classes: int) -> torch.Tensor:
    preferred_paths = (
        "decoder.center_ema.centers",
        "decoder.centers",
        "decoder.class_centers",
        "decoder.prototypes",
        "decoder.anchors",
        "center_ema.centers",
        "centers",
        "class_centers",
        "prototypes",
        "anchors",
        "class_anchors",
        "mu",
    )
    for path in preferred_paths:
        value = _module_attr(model, path)
        if isinstance(value, torch.Tensor):
            vectors = _flatten_class_vectors(value, num_classes, path)
            if vectors.shape[0] == num_classes:
                return vectors

    candidates = []
    for name, value in model.named_parameters(recurse=True):
        lname = name.lower()
        if any(token in lname for token in ("anchor", "prototype", "center", "mu")) and value.ndim >= 2:
            if value.shape[0] >= num_classes:
                candidates.append((name, value))
    for name, value in model.named_buffers(recurse=True):
        lname = name.lower()
        if any(token in lname for token in ("anchor", "prototype", "center", "mu")) and value.ndim >= 2:
            if value.shape[0] >= num_classes:
                candidates.append((name, value))
    if candidates:
        name, value = candidates[0]
        return _flatten_class_vectors(value, num_classes, name)
    raise RuntimeError(
        "Could not find Euclidean anchors/prototypes. Tried decoder.center_ema.centers and common names "
        "including anchors/prototypes/class_anchors/mu/centers/class_centers."
    )


def _find_linear_classifier_weight(model: torch.nn.Module, num_classes: int) -> torch.Tensor:
    preferred_modules = (
        "decoder.seg_layers",
        "classifier",
        "seg_head",
        "final_conv",
        "output_layer",
        "decoder.classifier",
        "decoder.seg_head",
        "decoder.final_conv",
        "decoder.output_layer",
    )
    for path in preferred_modules:
        value = _module_attr(model, path)
        if value is None:
            continue
        if isinstance(value, torch.nn.ModuleList) and len(value) > 0:
            value = value[-1]
        weight = getattr(value, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 2 and weight.shape[0] >= num_classes:
            return _flatten_class_vectors(weight, num_classes, f"{path}.weight")

    candidates = []
    for name, value in model.named_parameters(recurse=True):
        lname = name.lower()
        if not lname.endswith("weight"):
            continue
        if value.ndim >= 2 and value.shape[0] >= num_classes:
            if any(token in lname for token in ("classifier", "seg", "final", "output", "head")):
                candidates.append((name, value))
    if candidates:
        name, value = candidates[-1]
        return _flatten_class_vectors(value, num_classes, name)
    raise RuntimeError(
        "Could not find linear/dot-product classifier weight. Tried decoder.seg_layers[-1], classifier, "
        "seg_head, final_conv, output_layer and common *.weight names."
    )


def _collect_candidate_parameter_names(model: torch.nn.Module) -> str:
    tokens = ("anchor", "prototype", "center", "mu", "classifier", "seg", "final", "output", "head", "bias")
    names = []
    for name, _value in model.named_parameters(recurse=True):
        if any(token in name.lower() for token in tokens):
            names.append(name)
    for name, _value in model.named_buffers(recurse=True):
        if any(token in name.lower() for token in tokens):
            names.append(name)
    return ", ".join(names[:80]) if names else "<none>"


def _find_euclidean_bias(model: torch.nn.Module, num_classes: int, vectors: torch.Tensor) -> torch.Tensor:
    preferred_paths = (
        "decoder.bias",
        "bias",
        "classifier.bias",
        "seg_head.bias",
        "final_conv.bias",
        "output_layer.bias",
    )
    for path in preferred_paths:
        value = _module_attr(model, path)
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] >= num_classes:
            decoder = _module_attr(model, "decoder")
            decoder_name = decoder.__class__.__name__.lower() if decoder is not None else ""
            decoder_mode = str(getattr(decoder, "mode", "")).lower() if decoder is not None else ""
            uses_softplus_bias = (
                path == "decoder.bias"
                and (
                    "imbalance" in decoder_name
                    or decoder_mode in {"dist", "relax", "relax_multi5", "relax_multi3", "relax_multi7"}
                )
            )
            if uses_softplus_bias:
                return torch.nn.functional.softplus(value.detach().float()[:num_classes]).to(device=vectors.device)
            return value.detach().float()[:num_classes].to(device=vectors.device)
    return torch.zeros(num_classes, dtype=vectors.dtype, device=vectors.device)


def _find_linear_classifier_module(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    preferred_modules = (
        "decoder.seg_layers",
        "classifier",
        "seg_head",
        "final_conv",
        "output_layer",
        "decoder.classifier",
        "decoder.seg_head",
        "decoder.final_conv",
        "decoder.output_layer",
    )
    for path in preferred_modules:
        value = _module_attr(model, path)
        if value is None:
            continue
        if isinstance(value, torch.nn.ModuleList) and len(value) > 0:
            value = value[-1]
        weight = getattr(value, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 2 and weight.shape[0] >= num_classes:
            return value

    candidates = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        lname = name.lower()
        if weight.ndim >= 2 and weight.shape[0] >= num_classes:
            if any(token in lname for token in ("classifier", "seg", "final", "output", "head")):
                candidates.append((name, module))
    if candidates:
        return candidates[-1][1]
    raise RuntimeError(
        "Could not find linear/dot-product classifier module. Candidate names: "
        + _collect_candidate_parameter_names(model)
    )


def _find_sqeuclidean_bias_head(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    preferred_modules = (
        "decoder.classify",
        "classify",
        "decoder.sqeuclidean_bias",
        "sqeuclidean_bias",
    )
    for path in preferred_modules:
        value = _module_attr(model, path)
        anchors = getattr(value, "anchors", None)
        bias = getattr(value, "bias", None)
        if (
            isinstance(value, torch.nn.Module)
            and isinstance(anchors, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and anchors.ndim >= 2
            and anchors.shape[0] >= num_classes
            and bias.ndim == 1
            and bias.shape[0] >= num_classes
        ):
            return value

    candidates = []
    for name, module in model.named_modules():
        lname = name.lower()
        anchors = getattr(module, "anchors", None)
        bias = getattr(module, "bias", None)
        if (
            isinstance(anchors, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and anchors.ndim >= 2
            and anchors.shape[0] >= num_classes
            and bias.ndim == 1
            and bias.shape[0] >= num_classes
            and ("sqeuclidean" in lname or "classify" in lname)
        ):
            candidates.append((name, module))
    if candidates:
        return candidates[-1][1]
    raise RuntimeError(
        "Could not find squared-Euclidean+bias head with anchors and bias. Candidate names: "
        + _collect_candidate_parameter_names(model)
    )


def extract_head_params(
    model: torch.nn.Module,
    head_type: str,
    num_classes: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract class anchors/weights and effective bias for normalized logit export."""
    if head_type in ("auto", "euclidean"):
        try:
            vectors = _find_euclidean_anchors(model, num_classes)
            bias = _find_euclidean_bias(model, num_classes, vectors)
            return vectors, bias
        except RuntimeError:
            if head_type == "euclidean":
                raise RuntimeError(
                    "Could not extract Euclidean head params. Candidate names: "
                    + _collect_candidate_parameter_names(model)
                )
    if head_type == "sqeuclidean_bias":
        module = _find_sqeuclidean_bias_head(model, num_classes)
        anchors = _flatten_class_vectors(module.anchors, num_classes, "sqeuclidean_bias.anchors")
        raw_bias = module.bias.detach().float()[:num_classes].to(device=anchors.device)
        effective_weight = 2.0 * anchors
        effective_bias = raw_bias - anchors.square().sum(dim=1)
        return effective_weight, effective_bias
    if head_type in ("auto", "linear", "dot"):
        module = _find_linear_classifier_module(model, num_classes)
        weight = getattr(module, "weight")
        vectors = _flatten_class_vectors(weight, num_classes, "linear_head.weight")
        bias_value = getattr(module, "bias", None)
        if isinstance(bias_value, torch.Tensor):
            bias = bias_value.detach().float()[:num_classes].to(device=vectors.device)
        else:
            bias = torch.zeros(num_classes, dtype=vectors.dtype, device=vectors.device)
        return vectors, bias
    raise RuntimeError(f"Unsupported head_type for head parameter extraction: {head_type}")


def compute_pair_scales_and_centers(
    model: torch.nn.Module,
    head_type: str,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if head_type == "none":
        raise RuntimeError("head_type='none' does not define pair scales")
    resolved = (
        "euclidean"
        if head_type == "euclidean"
        else "linear"
        if head_type in ("linear", "dot", "sqeuclidean_bias")
        else None
    )
    vectors, bias = extract_head_params(model, head_type=head_type, num_classes=num_classes)
    if resolved is None:
        try:
            _find_euclidean_anchors(model, num_classes)
            resolved = "euclidean"
        except RuntimeError:
            resolved = "linear"

    pair_scales: Dict[str, float] = {}
    pair_centers: Dict[str, float] = {}
    vector_norm_sq = vectors.square().sum(dim=1)
    for a in range(num_classes):
        for b in range(a + 1, num_classes):
            diff = vectors[a] - vectors[b]
            if resolved == "euclidean":
                scale = torch.linalg.vector_norm(diff, ord=2)
            else:
                scale = 0.5 * torch.linalg.vector_norm(diff, ord=2).square()
            scale_value = float(scale.detach().cpu().item())
            if not np.isfinite(scale_value) or scale_value <= eps:
                raise RuntimeError(f"Invalid pair scale U_{a}{b}={scale_value}; must be > {eps}")
            pair_scales[f"{a}-{b}"] = scale_value

    for a in range(num_classes):
        for b in range(num_classes):
            if a == b:
                continue
            if resolved == "euclidean":
                center = bias[b] - bias[a]
            else:
                center = 0.5 * (vector_norm_sq[b] - vector_norm_sq[a]) + (bias[b] - bias[a])
            pair_centers[f"{a}->{b}"] = float(center.detach().cpu().item())
    return pair_scales, pair_centers


def _pair_scale_tensor(pair_scales: Dict[str, float] | Dict[str, torch.Tensor], key: str, logits: torch.Tensor, eps: float) -> torch.Tensor:
    if key not in pair_scales:
        raise KeyError(f"Missing pair scale {key}")
    scale = torch.as_tensor(pair_scales[key], dtype=logits.dtype, device=logits.device)
    if not torch.isfinite(scale) or float(scale.detach().cpu().item()) <= eps:
        raise RuntimeError(f"Invalid pair scale {key}={float(scale.detach().cpu().item())}")
    return scale + float(eps)


def _pair_center_tensor(pair_centers: Dict[str, float] | Dict[str, torch.Tensor], key: str, logits: torch.Tensor) -> torch.Tensor:
    if key not in pair_centers:
        raise KeyError(f"Missing pair center {key}")
    return torch.as_tensor(pair_centers[key], dtype=logits.dtype, device=logits.device)


def compute_lt_scaled_logits(
    logits: torch.Tensor,
    pair_scales: Dict[str, float] | Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    return logits / _pair_scale_tensor(pair_scales, "1-2", logits, eps)


def compute_pairwise_bias_margins(
    logits: torch.Tensor,
    pair_scales: Dict[str, float] | Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    return {
        "T_to_L": (logits[:, 1] - logits[:, 2]) / _pair_scale_tensor(pair_scales, "1-2", logits, eps),
        "L_to_T": (logits[:, 2] - logits[:, 1]) / _pair_scale_tensor(pair_scales, "1-2", logits, eps),
        "T_to_BG": (logits[:, 0] - logits[:, 2]) / _pair_scale_tensor(pair_scales, "0-2", logits, eps),
        "BG_to_T": (logits[:, 2] - logits[:, 0]) / _pair_scale_tensor(pair_scales, "0-2", logits, eps),
        "L_to_BG": (logits[:, 0] - logits[:, 1]) / _pair_scale_tensor(pair_scales, "0-1", logits, eps),
        "BG_to_L": (logits[:, 1] - logits[:, 0]) / _pair_scale_tensor(pair_scales, "0-1", logits, eps),
    }


def compute_pairwise_geo_margins(
    logits: torch.Tensor,
    pair_scales: Dict[str, float] | Dict[str, torch.Tensor],
    pair_centers: Dict[str, float] | Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    return {
        "T_to_L": ((logits[:, 1] - logits[:, 2]) - _pair_center_tensor(pair_centers, "2->1", logits)) / _pair_scale_tensor(pair_scales, "1-2", logits, eps),
        "L_to_T": ((logits[:, 2] - logits[:, 1]) - _pair_center_tensor(pair_centers, "1->2", logits)) / _pair_scale_tensor(pair_scales, "1-2", logits, eps),
        "T_to_BG": ((logits[:, 0] - logits[:, 2]) - _pair_center_tensor(pair_centers, "2->0", logits)) / _pair_scale_tensor(pair_scales, "0-2", logits, eps),
        "BG_to_T": ((logits[:, 2] - logits[:, 0]) - _pair_center_tensor(pair_centers, "0->2", logits)) / _pair_scale_tensor(pair_scales, "0-2", logits, eps),
        "L_to_BG": ((logits[:, 0] - logits[:, 1]) - _pair_center_tensor(pair_centers, "1->0", logits)) / _pair_scale_tensor(pair_scales, "0-1", logits, eps),
        "BG_to_L": ((logits[:, 1] - logits[:, 0]) - _pair_center_tensor(pair_centers, "0->1", logits)) / _pair_scale_tensor(pair_scales, "0-1", logits, eps),
    }


def compute_liver_tumor_pair_probability_from_lt_scaled_logits(
    lt_scaled_logits: torch.Tensor,
    pair_scales: Dict[str, float] | Dict[str, torch.Tensor],
    pair_centers: Dict[str, float] | Dict[str, torch.Tensor],
    tau: float,
    margin_mode: str = "bias",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return L->T margin and pairwise probabilities from logits already divided by U_LT."""
    if lt_scaled_logits.ndim not in (4, 5):
        raise ValueError(f"lt_scaled_logits must be [B,C,H,W] or [B,C,D,H,W], got {tuple(lt_scaled_logits.shape)}")
    if lt_scaled_logits.shape[1] < 3:
        raise ValueError(f"lt_scaled_logits must have at least 3 channels, got {tuple(lt_scaled_logits.shape)}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    l_to_t_margin = lt_scaled_logits[:, 2] - lt_scaled_logits[:, 1]
    if margin_mode == "geo":
        center = _pair_center_tensor(pair_centers, "1->2", lt_scaled_logits)
        scale = _pair_scale_tensor(pair_scales, "1-2", lt_scaled_logits, eps)
        l_to_t_margin = l_to_t_margin - center / scale
    elif margin_mode != "bias":
        raise ValueError(f"Unsupported liver-tumor margin mode: {margin_mode}")

    p_tumor_norm = torch.sigmoid(l_to_t_margin / float(tau))
    p_liver_norm = 1.0 - p_tumor_norm
    return l_to_t_margin, p_tumor_norm, p_liver_norm


def compute_global_score_scale(
    model: torch.nn.Module,
    head_type: str,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute model-level global score scale for normalized softmax confidence.

    Euclidean: U_ij = ||mu_i - mu_j||.
    Linear/dot: U_ij = 0.5 * ||W_i - W_j||^2.
    """
    if head_type == "none":
        raise RuntimeError("head_type='none' does not define a global score scale")
    if head_type in ("auto", "euclidean"):
        try:
            vectors = _find_euclidean_anchors(model, num_classes)
            resolved = "euclidean"
        except RuntimeError:
            if head_type == "euclidean":
                raise
            vectors = _find_linear_classifier_weight(model, num_classes)
            resolved = "linear"
    elif head_type == "sqeuclidean_bias":
        vectors, _bias = extract_head_params(model, head_type=head_type, num_classes=num_classes)
        resolved = "linear"
    elif head_type in ("linear", "dot"):
        vectors = _find_linear_classifier_weight(model, num_classes)
        resolved = "linear"
    else:
        raise RuntimeError(f"Unsupported score normalization head_type: {head_type}")

    pair_values = []
    pair_scales: Dict[str, float] = {}
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            diff = vectors[i] - vectors[j]
            if resolved == "euclidean":
                value = torch.linalg.vector_norm(diff, ord=2)
            else:
                value = 0.5 * torch.linalg.vector_norm(diff, ord=2).square()
            pair_values.append(value)
            pair_scales[f"{i}-{j}"] = float(value.detach().cpu().item())

    scale = torch.stack(pair_values).mean()
    if not torch.isfinite(scale) or float(scale.detach().cpu().item()) <= eps:
        raise RuntimeError(f"Invalid global score scale {float(scale.detach().cpu().item())}; must be > {eps}")
    return scale, pair_scales


def normalize_logits_by_global_scale(
    logits: torch.Tensor,
    S_global: torch.Tensor | float,
    tau: float = 0.25,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.as_tensor(S_global, device=logits.device, dtype=logits.dtype) * float(tau) + float(eps)
    if not torch.isfinite(scale) or float(scale.detach().cpu().item()) <= eps:
        raise RuntimeError(f"Invalid normalization denominator {float(scale.detach().cpu().item())}")
    norm_logits = logits / scale
    norm_probs = torch.softmax(norm_logits, dim=1)
    return norm_logits, norm_probs


def _ratio_or_nan(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _store_ratio_metric(result: Dict[str, Any], key: str, numerator: int, denominator: int) -> None:
    result[key] = _ratio_or_nan(numerator, denominator)
    result[f"{key}_num"] = int(numerator)
    result[f"{key}_den"] = int(denominator)


def _squeeze_prediction_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    return tensor


def compute_normalized_hce_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    norm_probs: torch.Tensor,
    thresholds: Iterable[float] = tuple(normalized_global_hce_thresholds),
    num_classes: int = 3,
) -> Dict[str, Any]:
    """Compute HCE metrics from raw predictions and global-score normalized probabilities."""
    pred = _squeeze_prediction_tensor(torch.as_tensor(pred)).long()
    target = _squeeze_prediction_tensor(torch.as_tensor(target)).long()
    norm_probs = torch.as_tensor(norm_probs).float()
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if norm_probs.ndim != pred.ndim + 1 or norm_probs.shape[0] != pred.shape[0] or norm_probs.shape[1] < num_classes:
        raise ValueError(
            f"norm_probs must be [B,C,...] with C>={num_classes}; got {tuple(norm_probs.shape)} for pred {tuple(pred.shape)}"
        )
    if tuple(norm_probs.shape[2:]) != tuple(pred.shape[1:]):
        raise ValueError(
            f"norm_probs spatial shape {tuple(norm_probs.shape[2:])} must match pred {tuple(pred.shape[1:])}"
        )

    pred_for_gather = pred.clamp(min=0, max=norm_probs.shape[1] - 1)
    pred_conf = norm_probs.gather(1, pred_for_gather.unsqueeze(1)).squeeze(1)
    err = pred != target
    result: Dict[str, Any] = {}

    direction_specs = (
        ("T_to_L", 2, 1),
        ("L_to_T", 1, 2),
        ("T_to_BG", 2, 0),
        ("L_to_BG", 1, 0),
        ("BG_to_L", 0, 1),
        ("BG_to_T", 0, 2),
    )

    for threshold in thresholds:
        q = float(threshold)
        q_key = _threshold_key(q)
        _store_ratio_metric(
            result,
            f"nHCE_error_cond@{q_key}",
            int((err & (pred_conf > q)).sum().item()),
            int(err.sum().item()),
        )
        for cls in range(num_classes):
            gt_cls = target == cls
            _store_ratio_metric(
                result,
                f"nHCE_GT{cls}_wrong@{q_key}",
                int((gt_cls & (pred != cls) & (pred_conf > q)).sum().item()),
                int(gt_cls.sum().item()),
            )
        for name, gt_cls, pred_cls in direction_specs:
            gt_mask = target == gt_cls
            direction = gt_mask & (pred == pred_cls)
            _store_ratio_metric(
                result,
                f"nHCE_{name}@{q_key}",
                int((direction & (norm_probs[:, pred_cls] > q)).sum().item()),
                int(gt_mask.sum().item()),
            )
    return result


def aggregate_normalized_hce_metrics(
    case_metrics: List[Dict[str, Any]],
    thresholds: Iterable[float] = tuple(normalized_global_hce_thresholds),
    num_classes: int = 3,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    base_keys = []
    for threshold in thresholds:
        q_key = _threshold_key(float(threshold))
        base_keys.append(f"nHCE_error_cond@{q_key}")
        base_keys.extend(f"nHCE_GT{cls}_wrong@{q_key}" for cls in range(num_classes))
        base_keys.extend(
            f"nHCE_{direction}@{q_key}"
            for direction in ("T_to_L", "L_to_T", "T_to_BG", "L_to_BG", "BG_to_L", "BG_to_T")
        )
    for key in base_keys:
        numerator = int(sum(int(case.get(f"{key}_num", 0)) for case in case_metrics))
        denominator = int(sum(int(case.get(f"{key}_den", 0)) for case in case_metrics))
        _store_ratio_metric(result, key, numerator, denominator)
    return result


def _as_class_vector(value: torch.Tensor | np.ndarray | Iterable[float], feat: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=feat.device, dtype=feat.dtype)
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    if tensor.numel() != feat.shape[1]:
        raise ValueError(
            f"Class vector length {tensor.numel()} does not match feature channels {feat.shape[1]}"
        )
    return tensor


def _as_scalar_tensor(value: torch.Tensor | float | int, feat: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=feat.device, dtype=feat.dtype).reshape(())


def _channel_broadcast_shape(feat: torch.Tensor) -> Tuple[int, ...]:
    return (1, feat.shape[1], *([1] * (feat.ndim - 2)))


def compute_pair_prob_euclidean(
    feat: torch.Tensor,
    mu_liver: torch.Tensor | np.ndarray | Iterable[float],
    mu_tumor: torch.Tensor | np.ndarray | Iterable[float],
    b_liver: torch.Tensor | float | int,
    b_tumor: torch.Tensor | float | int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return liver-tumor normalized margin and pairwise probabilities for an Euclidean head.

    feat must be [B, C, H, W] or [B, C, D, H, W]. Returned tensors are [B, H, W]
    or [B, D, H, W].
    """
    if feat.ndim not in (4, 5):
        raise ValueError(f"feat must be [B,C,H,W] or [B,C,D,H,W], got shape {tuple(feat.shape)}")

    mu_l = _as_class_vector(mu_liver, feat)
    mu_t = _as_class_vector(mu_tumor, feat)
    b_l = _as_scalar_tensor(b_liver, feat)
    b_t = _as_scalar_tensor(b_tumor, feat)

    view_shape = _channel_broadcast_shape(feat)
    dist_l = torch.linalg.vector_norm(feat - mu_l.view(view_shape), dim=1)
    dist_t = torch.linalg.vector_norm(feat - mu_t.view(view_shape), dim=1)
    denom = torch.linalg.vector_norm(mu_t - mu_l) + float(eps)
    eta_tl = ((b_t - b_l) + dist_l - dist_t) / denom
    p_tumor_norm = torch.sigmoid(eta_tl)
    p_liver_norm = 1.0 - p_tumor_norm
    return eta_tl, p_tumor_norm, p_liver_norm


def compute_pair_prob_dot(
    feat: torch.Tensor,
    W_liver: torch.Tensor | np.ndarray | Iterable[float],
    W_tumor: torch.Tensor | np.ndarray | Iterable[float],
    b_liver: torch.Tensor | float | int,
    b_tumor: torch.Tensor | float | int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return liver-tumor normalized margin and pairwise probabilities for a dot-product head.

    feat must be [B, C, H, W] or [B, C, D, H, W]. Returned tensors are [B, H, W]
    or [B, D, H, W].
    """
    if feat.ndim not in (4, 5):
        raise ValueError(f"feat must be [B,C,H,W] or [B,C,D,H,W], got shape {tuple(feat.shape)}")

    w_l = _as_class_vector(W_liver, feat)
    w_t = _as_class_vector(W_tumor, feat)
    b_l = _as_scalar_tensor(b_liver, feat)
    b_t = _as_scalar_tensor(b_tumor, feat)

    w_delta = w_t - w_l
    linear_margin = (feat * w_delta.view(_channel_broadcast_shape(feat))).sum(dim=1) + (b_t - b_l)
    denom = w_delta.square().sum() + float(eps)
    eta_tl = 2.0 * linear_margin / denom
    p_tumor_norm = torch.sigmoid(eta_tl)
    p_liver_norm = 1.0 - p_tumor_norm
    return eta_tl, p_tumor_norm, p_liver_norm


def _squeeze_optional_singleton_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    return tensor


def compute_liver_tumor_normalized_hce(
    pred: torch.Tensor,
    target: torch.Tensor,
    p_tumor_norm: torch.Tensor,
    thresholds: Iterable[float] = tuple(normalized_liver_tumor_hce_thresholds),
    liver_label: int = 1,
    tumor_label: int = 2,
) -> Dict[str, Any]:
    """Compute GT-normalized liver-tumor pairwise high-confidence errors.

    pred and target are [B,H,W] or [B,D,H,W]. p_tumor_norm has the same shape and
    is only used for liver-tumor confidence, not for deriving pred.
    """
    pred = _squeeze_optional_singleton_channel(torch.as_tensor(pred)).long()
    target = _squeeze_optional_singleton_channel(torch.as_tensor(target)).long()
    p_tumor_norm = _squeeze_optional_singleton_channel(torch.as_tensor(p_tumor_norm)).float()

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
    if p_tumor_norm.shape != target.shape:
        raise ValueError(
            f"p_tumor_norm must match target shape, got {tuple(p_tumor_norm.shape)} and {tuple(target.shape)}"
        )

    gt_tumor = target == tumor_label
    gt_liver = target == liver_label
    pred_liver = pred == liver_label
    pred_tumor = pred == tumor_label
    p_liver_norm = 1.0 - p_tumor_norm

    tumor_den = int(gt_tumor.sum().item())
    liver_den = int(gt_liver.sum().item())
    result: Dict[str, Any] = {}

    for threshold in thresholds:
        q = float(threshold)
        q_key = _threshold_key(q)
        t_to_l_num = int((gt_tumor & pred_liver & (p_liver_norm > q)).sum().item())
        l_to_t_num = int((gt_liver & pred_tumor & (p_tumor_norm > q)).sum().item())

        result[f"T_to_L@{q_key}"] = _safe_ratio_or_nan(t_to_l_num, tumor_den)
        result[f"T_to_L@{q_key}_num"] = t_to_l_num
        result[f"T_to_L@{q_key}_den"] = tumor_den
        result[f"L_to_T@{q_key}"] = _safe_ratio_or_nan(l_to_t_num, liver_den)
        result[f"L_to_T@{q_key}_num"] = l_to_t_num
        result[f"L_to_T@{q_key}_den"] = liver_den
    return result


def aggregate_liver_tumor_normalized_hce(
    case_metrics: List[Dict[str, Any]],
    thresholds: Iterable[float] = tuple(normalized_liver_tumor_hce_thresholds),
) -> Dict[str, Any]:
    """Aggregate normalized liver-tumor HCE by summing numerators and denominators."""
    result: Dict[str, Any] = {}
    for threshold in thresholds:
        q_key = _threshold_key(float(threshold))
        for direction in ("T_to_L", "L_to_T"):
            num_key = f"{direction}@{q_key}_num"
            den_key = f"{direction}@{q_key}_den"
            ratio_key = f"{direction}@{q_key}"
            numerator = int(sum(int(case.get(num_key, 0)) for case in case_metrics))
            denominator = int(sum(int(case.get(den_key, 0)) for case in case_metrics))
            result[ratio_key] = _safe_ratio_or_nan(numerator, denominator)
            result[num_key] = numerator
            result[den_key] = denominator
    return result


def _make_neighbors(ndim: int, connected_dim: str) -> List[tuple[int, ...]]:
    if connected_dim not in {"2d", "3d"}:
        raise ValueError(f"Unsupported connected_dim: {connected_dim}")

    if connected_dim == "2d" and ndim >= 2:
        axes = [ndim - 2, ndim - 1]
    else:
        axes = list(range(ndim))

    neighbors = []
    for axis in axes:
        delta = [0] * ndim
        delta[axis] = -1
        neighbors.append(tuple(delta))
        delta = [0] * ndim
        delta[axis] = 1
        neighbors.append(tuple(delta))
    return neighbors


def _connected_components(mask: np.ndarray, connected_dim: str) -> List[List[tuple[int, ...]]]:
    structure = np.zeros((3,) * mask.ndim, dtype=np.uint8)
    structure[(1,) * mask.ndim] = 1
    if connected_dim == "2d" and mask.ndim >= 2:
        structure[(1,) * (mask.ndim - 2) + (0, 1)] = 1
        structure[(1,) * (mask.ndim - 2) + (2, 1)] = 1
        structure[(1,) * (mask.ndim - 2) + (1, 0)] = 1
        structure[(1,) * (mask.ndim - 2) + (1, 2)] = 1
    else:
        for axis in range(mask.ndim):
            low = [1] * mask.ndim
            low[axis] = 0
            structure[tuple(low)] = 1
            high = [1] * mask.ndim
            high[axis] = 2
            structure[tuple(high)] = 1

    labeled, num_features = ndimage.label(mask.astype(np.uint8), structure=structure)
    components: List[List[tuple[int, ...]]] = []
    for idx in range(1, num_features + 1):
        component = np.argwhere(labeled == idx)
        components.append([tuple(int(i) for i in coord) for coord in component])
    return components


def double_threshold_lesion_postprocess(
    prob: torch.Tensor,
    low_thr: float = default_low_thr,
    high_thr: float = default_high_thr,
    min_component_size: int = min_component_size,
    connected_dim: str = connected_dim,
) -> torch.Tensor:
    if high_thr <= low_thr:
        raise ValueError("high_thr must be greater than low_thr")

    lesion_prob = prob[:, 2]
    low_mask = lesion_prob > low_thr
    high_mask = lesion_prob > high_thr

    result = torch.zeros_like(low_mask, dtype=torch.uint8)
    low_np = low_mask.cpu().numpy()
    high_np = high_mask.cpu().numpy()

    for b in range(low_np.shape[0]):
        components = _connected_components(low_np[b], connected_dim=connected_dim)
        keep = np.zeros_like(low_np[b], dtype=np.uint8)
        for component in components:
            if min_component_size > 0 and len(component) < min_component_size:
                continue
            if not any(high_np[b][idx] for idx in component):
                continue
            for idx in component:
                keep[idx] = 1
        result[b] = torch.from_numpy(keep)
    return result


def compute_confidence_error_metrics(
    prob: torch.Tensor,
    target: torch.Tensor,
    pred: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if pred is None:
        pred = prob.argmax(dim=1)

    max_prob = prob.max(dim=1).values
    err_mask = pred != target
    result: Dict[str, Any] = {}
    class_specs = ((0, "bg"), (1, "liver"), (2, "lesion"))

    if int(err_mask.sum().item()) == 0:
        result["mean_error_confidence"] = 0.0
    else:
        result["mean_error_confidence"] = _as_float(max_prob[err_mask].mean())

    for tau in confidence_thresholds:
        tau_key = f"{tau:.1f}"
        result[f"hcer_voxel_{tau_key}"] = _as_float(((err_mask) & (max_prob > tau)).float().mean())
        if int(err_mask.sum().item()) == 0:
            result[f"hcer_error_{tau_key}"] = 0.0
        else:
            result[f"hcer_error_{tau_key}"] = _as_float((max_prob[err_mask] > tau).float().mean())

        for class_id, class_name in class_specs:
            gt_class = target == class_id
            gt_class_error = gt_class & err_mask
            gt_class_hce = gt_class_error & (max_prob > tau)
            gt_class_lcer = gt_class_error & (max_prob <= tau)
            result[f"gt_{class_name}_hcer_error_{tau_key}"] = _safe_ratio(
                gt_class_hce.sum(), gt_class_error.sum()
            )
            result[f"gt_{class_name}_hcer_voxel_{tau_key}"] = _safe_ratio(
                gt_class_hce.sum(), gt_class.sum()
            )
            result[f"gt_{class_name}_lcer_error_{tau_key}"] = _safe_ratio(
                gt_class_lcer.sum(), gt_class_error.sum()
            )
            result[f"gt_{class_name}_lcer_voxel_{tau_key}"] = _safe_ratio(
                gt_class_lcer.sum(), gt_class.sum()
            )

        gt_liver = target == 1
        num_gt_liver = ((gt_liver) & (prob[:, 2] > tau)).sum()
        result[f"gt_liver_p_lesion_{tau_key}"] = _safe_ratio(num_gt_liver, gt_liver.sum())

        gt_lesion = target == 2
        num_gt_lesion = ((gt_lesion) & (prob[:, 1] > tau)).sum()
        result[f"gt_lesion_p_liver_{tau_key}"] = _safe_ratio(num_gt_lesion, gt_lesion.sum())

        non_lesion = target != 2
        num_non_lesion = ((non_lesion) & (prob[:, 2] > tau)).sum()
        result[f"nonlesion_p_lesion_{tau_key}"] = _safe_ratio(num_non_lesion, non_lesion.sum())

    return result


def summarize_confidence_profile_for_console(payload: Dict[str, Any], tau: str = "0.7") -> Dict[str, float]:
    metrics = payload.get("dataset_metrics", {})
    keys = [
        f"hcer_error_{tau}",
        f"hcer_voxel_{tau}",
        f"gt_bg_hcer_voxel_{tau}",
        f"gt_liver_hcer_voxel_{tau}",
        f"gt_lesion_hcer_voxel_{tau}",
        f"gt_lesion_p_liver_{tau}",
        f"gt_bg_lcer_voxel_{tau}",
        f"gt_liver_lcer_voxel_{tau}",
        f"gt_lesion_lcer_voxel_{tau}",
    ]
    return {
        key: float(metrics[key])
        for key in keys
        if isinstance(metrics.get(key), (int, float))
    }


def profile_case_from_probabilities(
    case_id: str,
    prob: torch.Tensor,
    target: torch.Tensor,
    pred: Optional[torch.Tensor] = None,
    spacing: Optional[Iterable[float]] = None,
    file_ending: Optional[str] = None,
    prediction_file: Optional[str] = None,
    reference_file: Optional[str] = None,
) -> Dict[str, Any]:
    if pred is None:
        pred = prob.argmax(dim=1)

    pred_liver = pred == 1
    pred_lesion = pred == 2
    tgt_liver = target == 1
    tgt_lesion = target == 2

    tp_liver, fp_liver, fn_liver, _ = compute_tp_fp_fn_tn(
        tgt_liver.cpu().numpy().astype(bool),
        pred_liver.cpu().numpy().astype(bool),
    )
    tp_lesion, fp_lesion, fn_lesion, _ = compute_tp_fp_fn_tn(
        tgt_lesion.cpu().numpy().astype(bool),
        pred_lesion.cpu().numpy().astype(bool),
    )

    result: Dict[str, Any] = {
        "case_id": case_id,
        "dice_liver_raw": 0.0 if (2 * tp_liver + fp_liver + fn_liver) == 0 else (2 * tp_liver) / (2 * tp_liver + fp_liver + fn_liver),
        "dice_lesion_raw": 0.0 if (2 * tp_lesion + fp_lesion + fn_lesion) == 0 else (2 * tp_lesion) / (2 * tp_lesion + fp_lesion + fn_lesion),
        "hd95_liver_raw": 0.0,
        "hd95_lesion_raw": 0.0,
    }
    if spacing is not None:
        result["spacing"] = list(spacing)
    if file_ending is not None:
        result["file_ending"] = file_ending
    if prediction_file is not None:
        result["prediction_file"] = prediction_file
    if reference_file is not None:
        result["reference_file"] = reference_file

    result.update(compute_confidence_error_metrics(prob=prob, target=target, pred=pred))
    return result


def _compute_binary_dice(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    pred_bool = pred_mask.bool()
    target_bool = target_mask.bool()
    tp = (pred_bool & target_bool).sum().item()
    fp = (pred_bool & (~target_bool)).sum().item()
    fn = ((~pred_bool) & target_bool).sum().item()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def _build_double_threshold_case_record(
    case_id: str,
    prob: torch.Tensor,
    target: torch.Tensor,
    low_thr: float,
    high_thr: float,
) -> Dict[str, Any]:
    lesion_pred = double_threshold_lesion_postprocess(
        prob,
        low_thr=low_thr,
        high_thr=high_thr,
        min_component_size=min_component_size,
        connected_dim=connected_dim,
    )
    lesion_target = (target == 2).to(torch.uint8)
    return {
        "case_id": case_id,
        "dice_lesion_dt": _compute_binary_dice(lesion_pred, lesion_target),
    }


def _build_double_threshold_scan(
    case_cache: List[Dict[str, Any]],
    threshold_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    pairs = []
    best_pair = None

    if threshold_pairs is None:
        threshold_pairs = [(default_low_thr, default_high_thr)]

    for low_thr, high_thr in threshold_pairs:
        if high_thr <= low_thr:
            continue
        case_records = [
            _build_double_threshold_case_record(
                case_id=entry["case_id"],
                prob=entry["prob"],
                target=entry["target"],
                low_thr=low_thr,
                high_thr=high_thr,
            )
            for entry in case_cache
        ]
        dataset_metrics = aggregate_case_profiles(case_records)
        pair_record = {
            "low_thr": low_thr,
            "high_thr": high_thr,
            "dataset_metrics": dataset_metrics,
        }
        pairs.append(pair_record)

        if best_pair is None or dataset_metrics.get("dice_lesion_dt", 0.0) > best_pair["dataset_metrics"].get("dice_lesion_dt", 0.0):
            best_pair = pair_record

    return {
        "default": {
            "low_thr": default_low_thr,
            "high_thr": default_high_thr,
        },
        "pairs": pairs,
        "best_pair_by_lesion_dice": best_pair,
    }


def _iter_case_probability_data(
    pred_folder: str,
    gt_folder: str,
    dataset_json_file: str,
    file_ending: str,
):
    pred_files = subfiles(pred_folder, suffix=file_ending, join=True)
    for pred_file in pred_files:
        case_id = os.path.basename(pred_file)[:-len(file_ending)]
        ref_file = os.path.join(gt_folder, case_id + file_ending)
        if not os.path.isfile(ref_file):
            continue
        pred, target, _ = _load_segmentation_pair(pred_file, ref_file, dataset_json_file)
        target_t = _normalize_spatial_tensor(torch.from_numpy(target.astype(np.int64))).unsqueeze(0)
        prob = _load_probabilities_if_available(pred_file)
        if prob is None:
            pred_t = _normalize_spatial_tensor(torch.from_numpy(pred.astype(np.int64)))
            prob = torch.nn.functional.one_hot(pred_t.long(), num_classes=3).movedim(-1, 0).unsqueeze(0).float()
        yield {
            "case_id": case_id,
            "prob": prob,
            "target": target_t.long(),
        }


def _build_case_probability_cache_from_folders(
    pred_folder: str,
    gt_folder: str,
    dataset_json_file: str,
    file_ending: str,
) -> List[Dict[str, Any]]:
    dataset_json = load_json(dataset_json_file)
    pred_files = subfiles(pred_folder, suffix=file_ending, join=True)
    case_cache: List[Dict[str, Any]] = []
    for pred_file in pred_files:
        case_id = os.path.basename(pred_file)[:-len(file_ending)]
        ref_file = os.path.join(gt_folder, case_id + file_ending)
        if not os.path.isfile(ref_file):
            continue

        pred, target, props = _load_segmentation_pair(pred_file, ref_file, dataset_json_file)
        pred_t = _normalize_spatial_tensor(torch.from_numpy(pred.astype(np.int64)))
        target_t = _normalize_spatial_tensor(torch.from_numpy(target.astype(np.int64)))

        prob = _load_probabilities_if_available(pred_file)
        if prob is None:
            prob = torch.nn.functional.one_hot(pred_t.long(), num_classes=3).movedim(-1, 0).unsqueeze(0).float()

        case_cache.append(
            {
                "case_id": case_id,
                "pred_file": pred_file,
                "ref_file": ref_file,
                "prob": prob,
                "pred_map": pred_t.unsqueeze(0).long(),
                "target": target_t.unsqueeze(0).long(),
                "spacing": props.get("spacing", None),
                "file_ending": dataset_json["file_ending"],
            }
        )
    return case_cache


def _build_double_threshold_scan_from_folders(
    pred_folder: str,
    gt_folder: str,
    dataset_json_file: str,
    file_ending: str,
    threshold_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    case_cache = _build_case_probability_cache_from_folders(
        pred_folder=pred_folder,
        gt_folder=gt_folder,
        dataset_json_file=dataset_json_file,
        file_ending=file_ending,
    )
    return _build_double_threshold_scan(case_cache=case_cache, threshold_pairs=threshold_pairs)


def _build_class_probability_coverage(case_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
    class_probs = {0: [], 1: [], 2: []}
    class_names = {0: "bg", 1: "liver", 2: "lesion"}

    for entry in case_cache:
        target = entry["target"][0]
        prob = entry["prob"][0]
        for cls in [0, 1, 2]:
            mask = target == cls
            if int(mask.sum().item()) > 0:
                class_probs[cls].append(prob[cls][mask].cpu().numpy())

    result: Dict[str, Any] = {}
    for cls in [0, 1, 2]:
        vals = np.concatenate(class_probs[cls]) if class_probs[cls] else np.array([], dtype=np.float32)
        class_result: Dict[str, Any] = {"count": int(len(vals))}
        for cov in coverage_levels:
            if len(vals) == 0:
                tau = 0.0
                empirical_frac = 0.0
            else:
                tau = float(np.quantile(vals, 1 - cov))
                empirical_frac = float((vals >= tau).mean())
            class_result[f"{cov:.2f}"] = {
                "tau": tau,
                "empirical_fraction": empirical_frac,
            }
        result[class_names[cls]] = class_result
    return result


def _build_class_probability_coverage_from_folders(
    pred_folder: str,
    gt_folder: str,
    dataset_json_file: str,
    file_ending: str,
) -> Dict[str, Any]:
    case_cache = _build_case_probability_cache_from_folders(
        pred_folder=pred_folder,
        gt_folder=gt_folder,
        dataset_json_file=dataset_json_file,
        file_ending=file_ending,
    )
    return _build_class_probability_coverage(case_cache)


def _build_high_confidence_domain_purity(case_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
    class_names = {0: "bg", 1: "liver", 2: "lesion"}
    domain_specs = {"liver": 1, "lesion": 2}
    stats = {
        domain_name: {
            tau: {"total": 0, "gt_bg": 0, "gt_liver": 0, "gt_lesion": 0}
            for tau in confidence_thresholds
        }
        for domain_name in domain_specs
    }

    for entry in case_cache:
        target = entry["target"][0]
        prob = entry["prob"][0]
        for domain_name, cls_idx in domain_specs.items():
            for tau in confidence_thresholds:
                mask = prob[cls_idx] >= tau
                record = stats[domain_name][tau]
                record["total"] += int(mask.sum().item())
                record["gt_bg"] += int(((target == 0) & mask).sum().item())
                record["gt_liver"] += int(((target == 1) & mask).sum().item())
                record["gt_lesion"] += int(((target == 2) & mask).sum().item())

    result: Dict[str, Any] = {}
    for domain_name in domain_specs:
        domain_result: Dict[str, Any] = {}
        for tau in confidence_thresholds:
            record = stats[domain_name][tau]
            total = record["total"]
            if total == 0:
                domain_result[f"{tau:.1f}"] = {
                    "total": 0,
                    "gt_bg_fraction": 0.0,
                    "gt_liver_fraction": 0.0,
                    "gt_lesion_fraction": 0.0,
                    f"gt_{domain_name}_fraction": 0.0,
                }
            else:
                domain_result[f"{tau:.1f}"] = {
                    "total": total,
                    "gt_bg_fraction": record["gt_bg"] / total,
                    "gt_liver_fraction": record["gt_liver"] / total,
                    "gt_lesion_fraction": record["gt_lesion"] / total,
                    f"gt_{domain_name}_fraction": record[f"gt_{domain_name}"] / total,
                }
        result[domain_name] = domain_result
    return result


def _finalize_class_probability_coverage_from_arrays(class_probs: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
    class_names = {0: "bg", 1: "liver", 2: "lesion"}
    result: Dict[str, Any] = {}
    for cls in [0, 1, 2]:
        vals = np.concatenate(class_probs[cls]) if class_probs[cls] else np.array([], dtype=np.float32)
        class_result: Dict[str, Any] = {"count": int(len(vals))}
        for cov in coverage_levels:
            if len(vals) == 0:
                tau = 0.0
                empirical_frac = 0.0
            else:
                tau = float(np.quantile(vals, 1 - cov))
                empirical_frac = float((vals >= tau).mean())
            class_result[f"{cov:.2f}"] = {
                "tau": tau,
                "empirical_fraction": empirical_frac,
            }
        result[class_names[cls]] = class_result
    return result


def _finalize_high_confidence_domain_purity_from_stats(stats: Dict[str, Dict[float, Dict[str, int]]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for domain_name, by_tau in stats.items():
        domain_result: Dict[str, Any] = {}
        for tau in confidence_thresholds:
            record = by_tau[tau]
            total = record["total"]
            if total == 0:
                domain_result[f"{tau:.1f}"] = {
                    "total": 0,
                    "gt_bg_fraction": 0.0,
                    "gt_liver_fraction": 0.0,
                    "gt_lesion_fraction": 0.0,
                    f"gt_{domain_name}_fraction": 0.0,
                }
            else:
                domain_result[f"{tau:.1f}"] = {
                    "total": total,
                    "gt_bg_fraction": record["gt_bg"] / total,
                    "gt_liver_fraction": record["gt_liver"] / total,
                    "gt_lesion_fraction": record["gt_lesion"] / total,
                    f"gt_{domain_name}_fraction": record[f"gt_{domain_name}"] / total,
                }
        result[domain_name] = domain_result
    return result


def _finalize_double_threshold_scan_from_case_records(
    pair_case_records: Dict[Tuple[float, float], List[Dict[str, Any]]]
) -> Dict[str, Any]:
    pairs = []
    best_pair = None
    for (low_thr, high_thr), case_records in pair_case_records.items():
        dataset_metrics = aggregate_case_profiles(case_records)
        pair_record = {
            "low_thr": low_thr,
            "high_thr": high_thr,
            "dataset_metrics": dataset_metrics,
        }
        pairs.append(pair_record)
        if best_pair is None or dataset_metrics.get("dice_lesion_dt", 0.0) > best_pair["dataset_metrics"].get("dice_lesion_dt", 0.0):
            best_pair = pair_record

    return {
        "default": {
            "low_thr": default_low_thr,
            "high_thr": default_high_thr,
        },
        "pairs": pairs,
        "best_pair_by_lesion_dice": best_pair,
    }


def _build_high_confidence_domain_purity_from_folders(
    pred_folder: str,
    gt_folder: str,
    dataset_json_file: str,
    file_ending: str,
) -> Dict[str, Any]:
    case_cache = _build_case_probability_cache_from_folders(
        pred_folder=pred_folder,
        gt_folder=gt_folder,
        dataset_json_file=dataset_json_file,
        file_ending=file_ending,
    )
    return _build_high_confidence_domain_purity(case_cache)


def aggregate_case_profiles(case_profiles: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(case_profiles) == 0:
        return {}
    numeric_keys = []
    for key, value in case_profiles[0].items():
        if key == "case_id":
            continue
        if isinstance(value, (int, float)):
            numeric_keys.append(key)
    result = {}
    for key in numeric_keys:
        values = [float(profile[key]) for profile in case_profiles if isinstance(profile.get(key), (int, float))]
        if len(values) == 0:
            continue
        result[key] = float(np.mean(values))
    return result


def build_confidence_profile(case_profiles: List[Dict[str, Any]], config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = {
        "confidence_thresholds": confidence_thresholds,
        "lesion_low_thresholds": lesion_low_thresholds,
        "lesion_high_thresholds": lesion_high_thresholds,
        "default_low_thr": default_low_thr,
        "default_high_thr": default_high_thr,
        "min_component_size": min_component_size,
        "connected_dim": connected_dim,
        "include_class_probability_coverage": True,
        "include_high_confidence_domain_purity": True,
        "include_double_threshold_scan": True,
        "include_hd95_metrics": True,
    }
    if config_override:
        config.update(config_override)
    return {
        "config": config,
        "dataset_metrics": aggregate_case_profiles(case_profiles),
        "cases": case_profiles,
        "class_probability_coverage": {},
        "high_confidence_domain_purity": {},
        "double_threshold_scan": {
            "default": {
                "low_thr": config["default_low_thr"],
                "high_thr": config["default_high_thr"],
            },
            "pairs": [],
        },
    }


def save_confidence_profile_json(case_profiles: List[Dict[str, Any]], output_file: str | Path, config_override: Optional[Dict[str, Any]] = None) -> None:
    output_file = Path(output_file)
    payload = build_confidence_profile(case_profiles, config_override=config_override)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_segmentation_pair(
    pred_file: str,
    ref_file: str,
    dataset_json_file: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    dataset_json = load_json(dataset_json_file)
    rw = determine_reader_writer_from_dataset_json(dataset_json, ref_file)()
    pred, pred_props = rw.read_seg(pred_file)
    ref, _ = rw.read_seg(ref_file)
    return pred, ref, pred_props


def _normalize_spatial_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.ndim > 0 and t.shape[0] == 1:
        return t[0]
    return t


def _load_probabilities_if_available(pred_file: str) -> Optional[torch.Tensor]:
    npz_file = os.path.splitext(os.path.splitext(pred_file)[0])[0] + '.npz'
    if not os.path.isfile(npz_file):
        return None

    payload = np.load(npz_file)
    if 'probabilities' not in payload.files:
        return None

    prob = torch.from_numpy(payload['probabilities']).float()
    if prob.ndim < 2:
        return None
    return prob.unsqueeze(0)


def _profile_case_from_segmentation_files(
    case_id: str,
    pred_file: str,
    ref_file: str,
    dataset_json_file: str,
) -> Dict[str, Any]:
    pred, target, props = _load_segmentation_pair(pred_file, ref_file, dataset_json_file)

    pred_t = _normalize_spatial_tensor(torch.from_numpy(pred.astype(np.int64)))
    target_t = _normalize_spatial_tensor(torch.from_numpy(target.astype(np.int64)))

    prob = _load_probabilities_if_available(pred_file)
    if prob is None:
        prob = torch.nn.functional.one_hot(pred_t.long(), num_classes=3).movedim(-1, 0).unsqueeze(0).float()
    pred_map = pred_t.unsqueeze(0)
    target_map = target_t.unsqueeze(0)

    spacing = props.get("spacing", None)

    profile = profile_case_from_probabilities(
        case_id=case_id,
        prob=prob,
        target=target_map.long(),
        pred=pred_map.long(),
        spacing=spacing,
        prediction_file=pred_file,
        reference_file=ref_file,
        file_ending=load_json(dataset_json_file)["file_ending"],
    )

    pred_liver = pred_map == 1
    pred_lesion = pred_map == 2
    tgt_liver = target_map == 1
    tgt_lesion = target_map == 2

    try:
        if int(pred_liver.sum().item()) > 0 and int(tgt_liver.sum().item()) > 0:
            profile["hd95_liver_raw"] = float(hd95(pred_liver.cpu().numpy().astype(bool), tgt_liver.cpu().numpy().astype(bool), voxelspacing=spacing))
    except Exception:
        profile["hd95_liver_raw"] = 0.0
    try:
        if int(pred_lesion.sum().item()) > 0 and int(tgt_lesion.sum().item()) > 0:
            profile["hd95_lesion_raw"] = float(hd95(pred_lesion.cpu().numpy().astype(bool), tgt_lesion.cpu().numpy().astype(bool), voxelspacing=spacing))
    except Exception:
        profile["hd95_lesion_raw"] = 0.0
    return profile


def generate_confidence_profile_for_folder(
    pred_folder: str,
    gt_folder: str,
    dataset_json_file: str,
    plans_file: str,
    output_file: Optional[str] = None,
    double_threshold_pairs: Optional[List[Tuple[float, float]]] = None,
    include_class_probability_coverage: bool = True,
    include_high_confidence_domain_purity: bool = True,
    include_double_threshold_scan: bool = True,
    include_hd95_metrics: bool = True,
) -> Dict[str, Any]:
    del plans_file  # kept in signature for stable integration shape
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json["file_ending"]

    if double_threshold_pairs is None:
        double_threshold_pairs = [(default_low_thr, default_high_thr)]

    pred_files = subfiles(pred_folder, suffix=file_ending, join=True)
    case_profiles = []
    class_probs = {0: [], 1: [], 2: []} if include_class_probability_coverage else None
    purity_stats = (
        {
            domain_name: {
                tau: {"total": 0, "gt_bg": 0, "gt_liver": 0, "gt_lesion": 0}
                for tau in confidence_thresholds
            }
            for domain_name in ("liver", "lesion")
        }
        if include_high_confidence_domain_purity
        else None
    )
    valid_pairs = (
        [(low_thr, high_thr) for low_thr, high_thr in double_threshold_pairs if high_thr > low_thr]
        if include_double_threshold_scan
        else []
    )
    pair_case_records: Dict[Tuple[float, float], List[Dict[str, Any]]] = (
        {pair: [] for pair in valid_pairs}
        if include_double_threshold_scan
        else {}
    )

    for pred_file in pred_files:
        case_id = os.path.basename(pred_file)[:-len(file_ending)]
        ref_file = os.path.join(gt_folder, case_id + file_ending)
        if not os.path.isfile(ref_file):
            continue

        pred, target, props = _load_segmentation_pair(pred_file, ref_file, dataset_json_file)
        pred_t = _normalize_spatial_tensor(torch.from_numpy(pred.astype(np.int64)))
        target_t = _normalize_spatial_tensor(torch.from_numpy(target.astype(np.int64)))
        pred_map = pred_t.unsqueeze(0).long()
        target_map = target_t.unsqueeze(0).long()

        prob = _load_probabilities_if_available(pred_file)
        if prob is None:
            prob = torch.nn.functional.one_hot(pred_t.long(), num_classes=3).movedim(-1, 0).unsqueeze(0).float()

        profile = profile_case_from_probabilities(
            case_id=case_id,
            prob=prob,
            target=target_map,
            pred=pred_map,
            spacing=props.get("spacing", None),
            prediction_file=pred_file,
            reference_file=ref_file,
            file_ending=file_ending,
        )

        pred_liver = pred_map == 1
        pred_lesion = pred_map == 2
        tgt_liver = target_map == 1
        tgt_lesion = target_map == 2

        if include_hd95_metrics:
            try:
                if int(pred_liver.sum().item()) > 0 and int(tgt_liver.sum().item()) > 0:
                    profile["hd95_liver_raw"] = float(hd95(pred_liver.cpu().numpy().astype(bool), tgt_liver.cpu().numpy().astype(bool), voxelspacing=props.get("spacing", None)))
            except Exception:
                profile["hd95_liver_raw"] = 0.0
            try:
                if int(pred_lesion.sum().item()) > 0 and int(tgt_lesion.sum().item()) > 0:
                    profile["hd95_lesion_raw"] = float(hd95(pred_lesion.cpu().numpy().astype(bool), tgt_lesion.cpu().numpy().astype(bool), voxelspacing=props.get("spacing", None)))
            except Exception:
                profile["hd95_lesion_raw"] = 0.0
        case_profiles.append(profile)

        if include_class_probability_coverage or include_high_confidence_domain_purity or include_double_threshold_scan:
            target0 = target_map[0]
            prob0 = prob[0]

            if include_class_probability_coverage:
                for cls in [0, 1, 2]:
                    mask = target0 == cls
                    if int(mask.sum().item()) > 0:
                        class_probs[cls].append(prob0[cls][mask].cpu().numpy())

            if include_high_confidence_domain_purity:
                for domain_name, cls_idx in {"liver": 1, "lesion": 2}.items():
                    for tau in confidence_thresholds:
                        mask = prob0[cls_idx] >= tau
                        record = purity_stats[domain_name][tau]
                        record["total"] += int(mask.sum().item())
                        record["gt_bg"] += int(((target0 == 0) & mask).sum().item())
                        record["gt_liver"] += int(((target0 == 1) & mask).sum().item())
                        record["gt_lesion"] += int(((target0 == 2) & mask).sum().item())

            if include_double_threshold_scan:
                for low_thr, high_thr in valid_pairs:
                    pair_case_records[(low_thr, high_thr)].append(
                        _build_double_threshold_case_record(
                            case_id=case_id,
                            prob=prob,
                            target=target_map,
                            low_thr=low_thr,
                            high_thr=high_thr,
                        )
                    )

    payload = build_confidence_profile(
        case_profiles,
        config_override={
            "include_class_probability_coverage": include_class_probability_coverage,
            "include_high_confidence_domain_purity": include_high_confidence_domain_purity,
            "include_double_threshold_scan": include_double_threshold_scan,
            "include_hd95_metrics": include_hd95_metrics,
        },
    )
    if include_class_probability_coverage:
        payload["class_probability_coverage"] = _finalize_class_probability_coverage_from_arrays(class_probs)
    if include_high_confidence_domain_purity:
        payload["high_confidence_domain_purity"] = _finalize_high_confidence_domain_purity_from_stats(purity_stats)
    if include_double_threshold_scan:
        payload["double_threshold_scan"] = _finalize_double_threshold_scan_from_case_records(pair_case_records)
    if output_file is None:
        output_file = os.path.join(pred_folder, "confidence_profile.json")
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload
