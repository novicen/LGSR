from __future__ import annotations

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
from lgsr.losses.lgsr_loss import LGSRSelectiveLoss


class nnUNetTrainer_lgsr(nnUNetTrainer):
    """LGSR primary configuration: K=(1,3,3), identity covariance, Dice enabled."""

    gmm_components_per_class = (1, 3, 3)
    gmm_feature_dim = 32
    gmm_nll_weight = 0.01
    safe_ball_rates = (1.0, 1.0, 1.0)
    require_safe_prediction_match = False
    num_epochs = 200
    num_iterations_per_epoch = 250
    num_val_iterations_per_epoch = 50

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["architecture"]["network_class_name"] = (
            "lgsr.models.unet.LGSRUNet"
        )
        arch = self.configuration_manager.configuration["architecture"]["arch_kwargs"]
        arch["mode"] = "gmm_posterior"
        arch["gmm_components_per_class"] = tuple(self.gmm_components_per_class)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
        num_input_channels, num_output_channels, enable_deep_supervision=True,
    ):
        kwargs = dict(arch_init_kwargs)
        kwargs["gmm_components_per_class"] = (1, 3, 3)
        return nnUNetTrainer.build_network_architecture(
            "lgsr.models.unet.LGSRUNet",
            kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False,
        )

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("LGSR requires class-label segmentation")
        return LGSRSelectiveLoss(
            use_gmm_nll=True,
            gmm_nll_weight=self.gmm_nll_weight,
            gmm_nll_per_dim=False,
            ignore_label=self.label_manager.ignore_label,
            use_strict_safe_ball=True,
            safe_ball_rates=self.safe_ball_rates,
            require_safe_prediction_match=self.require_safe_prediction_match,
            feature_dim=self.gmm_feature_dim,
            use_dice=True,
            dice_weight=1.0,
            dice_batch_dice=True,
        )

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = target[0] if isinstance(target, (list, tuple)) else target
        target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            payload = self.network(data, mode="train")
            loss = self.loss(payload, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        result = {"loss": loss.detach().cpu().numpy()}
        for key, value in self.loss.last_debug.items():
            if torch.is_tensor(value) and value.ndim == 0:
                result[key] = value.detach().cpu().numpy()
        return result

    def validation_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = target[0] if isinstance(target, (list, tuple)) else target
        target = target.to(self.device, non_blocking=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            payload = self.network(data, mode="train")
            loss = self.loss(payload, target)
            output = payload["logits"]
        pred = output.argmax(1)[:, None]
        pred_onehot = torch.zeros_like(output, dtype=torch.float32)
        pred_onehot.scatter_(1, pred, 1)
        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()
            target[target == self.label_manager.ignore_label] = 0
        else:
            mask = None
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target, axes=axes, mask=mask)
        result = {"loss": loss.detach().cpu().numpy(), "tp_hard": tp.detach().cpu().numpy()[1:], "fp_hard": fp.detach().cpu().numpy()[1:], "fn_hard": fn.detach().cpu().numpy()[1:]}
        for key, value in self.loss.last_debug.items():
            if torch.is_tensor(value) and value.ndim == 0:
                result[key] = value.detach().cpu().numpy()
        return result

