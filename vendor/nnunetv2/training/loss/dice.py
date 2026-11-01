from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc


class MemoryEfficientRelaxMaskDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        super(MemoryEfficientRelaxMaskDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.last_debug = {}

    @staticmethod
    def _ensure_mask_shape(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if mask.ndim == reference.ndim - 1:
            mask = mask.unsqueeze(1)
        if mask.ndim != reference.ndim:
            raise ValueError(f"Mask ndim {mask.ndim} does not match reference ndim {reference.ndim}")
        return mask.bool()

    def forward(self, x, y, loss_mask=None, inactive_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if loss_mask is None:
                active_mask = torch.ones((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=torch.bool)
            else:
                active_mask = self._ensure_mask_shape(loss_mask, y)

            if inactive_mask is None:
                inactive_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            else:
                inactive_mask = self._ensure_mask_shape(inactive_mask, y)

            inactive_mask = inactive_mask & (~active_mask)
            covered_mask = active_mask | inactive_mask

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = (y_onehot * covered_mask).sum(axes)
            num_active = active_mask.sum().float()
            num_inactive = inactive_mask.sum().float()
            num_total = covered_mask.sum().float()
            active_ratio = num_active / torch.clamp(num_total, min=1.0)
            mean_prior = inactive_mask.float().sum() / torch.clamp(num_total, min=1.0)

        if not self.do_bg:
            x = x[:, 1:]

        active_mask = active_mask.to(dtype=x.dtype)
        inactive_mask = inactive_mask.to(dtype=x.dtype)
        prior_x = x.detach()

        intersect = (x * y_onehot * active_mask).sum(axes)
        sum_pred = (x * active_mask).sum(axes)

        intersect = intersect + (prior_x * y_onehot * inactive_mask).sum(axes)
        sum_pred = sum_pred + (prior_x * inactive_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        dc = dc.mean()

        self.last_debug = {
            'num_active': num_active.detach(),
            'num_inactive': num_inactive.detach(),
            'active_ratio': active_ratio.detach(),
            'mean_prior': mean_prior.detach(),
        }
        return -dc


class MemoryEfficientSoftFocalDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, focal_gamma: float = 2.0):  # <--- 加 gamma 参数
        super(MemoryEfficientSoftFocalDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.focal_gamma = focal_gamma  # <--- 保存

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        focal_dice = (1 - dc) ** self.focal_gamma
        
        dc_loss = focal_dice.mean()

        return dc_loss


class MemoryEfficientTverskyLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
    ):
        super(MemoryEfficientTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes)
            fp = (x * (~y_onehot)).sum(axes)
            fn = ((1 - x) * y_onehot).sum(axes)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes)
            fp = (x * (~y_onehot) * loss_mask).sum(axes)
            fn = ((1 - x) * y_onehot * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)

            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        tversky = (tp + self.smooth) / torch.clamp(
            tp + self.alpha * fn + self.beta * fp + self.smooth,
            min=1e-8,
        )
        return ((1.0 - tversky) ** self.gamma).mean()


class AllClassActiveMixTverskyLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = True,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        mix_eta: float = 0.8,
        ignore_index: int = -100,
    ):
        super().__init__()
        if not 0.0 <= mix_eta <= 1.0:
            raise ValueError(f"mix_eta must be in [0, 1], got {mix_eta}")
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = float(smooth)
        self.ddp = ddp
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.mix_eta = float(mix_eta)
        self.ignore_index = int(ignore_index)
        self.last_debug = {}

    @staticmethod
    def _squeeze_spatial_mask(mask: torch.Tensor, reference_ndim: int) -> torch.Tensor:
        if mask.ndim == reference_ndim:
            if mask.shape[1] != 1:
                raise ValueError(f"Expected mask channel dimension to be 1, got shape {tuple(mask.shape)}")
            mask = mask[:, 0]
        return mask.bool()

    @staticmethod
    def _mean_per_class_debug(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.detach()
        return x.detach().mean(0)

    def forward(self, logits, target, active_mask, valid_mask):
        if logits.ndim < 4:
            raise ValueError(f"AllClassActiveMixTverskyLoss expects logits [B,C,...], got {tuple(logits.shape)}")

        if target.ndim == logits.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        active_mask = self._squeeze_spatial_mask(active_mask, logits.ndim).detach()
        valid_mask = self._squeeze_spatial_mask(valid_mask, logits.ndim)

        num_classes = logits.shape[1]
        valid = (
            valid_mask
            & (target >= 0)
            & (target < num_classes)
            & (target != self.ignore_index)
        )

        if not torch.any(valid):
            self.last_debug = {
                "mix_eta": self.mix_eta,
                "tversky_alpha": self.alpha,
                "tversky_beta": self.beta,
                "tversky_gamma": self.gamma,
            }
            return logits.sum() * 0.0

        target_safe = target.clone()
        target_safe[~valid] = 0

        probs = self.apply_nonlin(logits) if self.apply_nonlin is not None else torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target_safe, num_classes=num_classes).permute(
            0, target_safe.ndim, *range(1, target_safe.ndim)
        ).float()
        target_oh = target_oh.to(device=logits.device, dtype=probs.dtype)

        if not self.do_bg:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        active = active_mask & valid
        valid_f = valid[:, None].to(dtype=probs.dtype)
        active_f = active[:, None].to(dtype=probs.dtype)

        reduce_axes = tuple(range(2, probs.ndim))

        tp_active = (probs * target_oh * active_f).sum(reduce_axes)
        fp_active = (probs * (1.0 - target_oh) * active_f).sum(reduce_axes)
        fn_active = ((1.0 - probs) * target_oh * active_f).sum(reduce_axes)

        tp_full = (probs * target_oh * valid_f).sum(reduce_axes)
        fp_full = (probs * (1.0 - target_oh) * valid_f).sum(reduce_axes)
        fn_full = ((1.0 - probs) * target_oh * valid_f).sum(reduce_axes)

        active_den = tp_active + self.alpha * fn_active + self.beta * fp_active
        full_den = tp_full + self.alpha * fn_full + self.beta * fp_full
        mixed_den = self.mix_eta * active_den + (1.0 - self.mix_eta) * full_den

        if self.batch_dice:
            if self.ddp:
                tp_active = AllGatherGrad.apply(tp_active).sum(0)
                fp_active = AllGatherGrad.apply(fp_active).sum(0)
                fn_active = AllGatherGrad.apply(fn_active).sum(0)
                tp_full = AllGatherGrad.apply(tp_full).sum(0)
                fp_full = AllGatherGrad.apply(fp_full).sum(0)
                fn_full = AllGatherGrad.apply(fn_full).sum(0)
                active_den = AllGatherGrad.apply(active_den).sum(0)
                full_den = AllGatherGrad.apply(full_den).sum(0)
                mixed_den = AllGatherGrad.apply(mixed_den).sum(0)

            tp_active = tp_active.sum(0)
            fp_active = fp_active.sum(0)
            fn_active = fn_active.sum(0)
            tp_full = tp_full.sum(0)
            fp_full = fp_full.sum(0)
            fn_full = fn_full.sum(0)
            active_den = active_den.sum(0)
            full_den = full_den.sum(0)
            mixed_den = mixed_den.sum(0)

        active_tversky = (tp_active + self.smooth) / torch.clamp(mixed_den + self.smooth, min=1e-8)
        loss = ((1.0 - active_tversky) ** self.gamma).mean()

        safe_count = (valid & (~active)).sum().to(dtype=probs.dtype)
        valid_count = valid.sum().to(dtype=probs.dtype)
        active_count = active.sum().to(dtype=probs.dtype)
        safe_valid_count = torch.clamp(valid_count, min=1.0)

        self.last_debug = {
            "active_tp_per_class": self._mean_per_class_debug(tp_active),
            "active_fp_per_class": self._mean_per_class_debug(fp_active),
            "active_fn_per_class": self._mean_per_class_debug(fn_active),
            "full_tp_per_class": self._mean_per_class_debug(tp_full),
            "full_fp_per_class": self._mean_per_class_debug(fp_full),
            "full_fn_per_class": self._mean_per_class_debug(fn_full),
            "active_denominator_per_class": self._mean_per_class_debug(active_den),
            "full_denominator_per_class": self._mean_per_class_debug(full_den),
            "mixed_denominator_per_class": self._mean_per_class_debug(mixed_den),
            "active_tversky_per_class": self._mean_per_class_debug(active_tversky),
            "active_ratio": (active_count / safe_valid_count).detach(),
            "safe_ratio": (safe_count / safe_valid_count).detach(),
            "mix_eta": self.mix_eta,
            "tversky_alpha": self.alpha,
            "tversky_beta": self.beta,
            "tversky_gamma": self.gamma,
        }
        return loss


class AllClassActiveMixDiceLoss(AllClassActiveMixTverskyLoss):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1e-5,
        ddp: bool = True,
        mix_eta: float = 0.8,
        ignore_index: int = -100,
    ):
        super().__init__(
            apply_nonlin=apply_nonlin,
            batch_dice=batch_dice,
            do_bg=do_bg,
            smooth=smooth,
            ddp=ddp,
            alpha=0.5,
            beta=0.5,
            mix_eta=mix_eta,
            ignore_index=ignore_index,
        )

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
