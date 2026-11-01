import torch
from nnunetv2.training.loss.dice import (
    SoftDiceLoss,
    MemoryEfficientSoftDiceLoss,
    MemoryEfficientRelaxMaskDiceLoss,
    AllClassActiveMixDiceLoss,
    AllClassActiveMixTverskyLoss,
)
from nnunetv2.training.loss.robust_ce_loss import (
    RobustCrossEntropyLoss,
    HardL1ACELoss,
    SoftL1ACELoss,
    MEEPLoss,
    MultiAnchorSafeRegionCELoss,
    RobustLogitNormCrossEntropyLoss,
    RobustMbLSCrossEntropyLoss,
    RobustSVLSCrossEntropyLoss,
    TopKLoss,
    RobustFocalLoss,
    RobustRelaxLoss,
)
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import dist, nn
from torch.nn import functional as F

from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicBoundaryRemedyLoss(nn.Module):
    """
    改进版：局部二元不确定性校准损失。
    当真实标签为 t，模型高置信度误分为 p 时：
    强行将该像素的预测概率向 [t: 0.5, p: 0.5, 其它: 0.0] 对齐，消除第三个无关类的干扰。
    """
    def __init__(self, confidence_threshold=0.7, ignore_index=-100, weight=1.0):
        super(DynamicBoundaryRemedyLoss, self).__init__()
        self.confidence_threshold = confidence_threshold
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits, targets):
        """
        logits: [B, 3, H, W] (三分类: 0, 1, 2)
        targets: [B, H, W]
        """
        if targets.ndim == 4:
            targets = targets[:,0]
        B, K, H, W = logits.size()
        probs = F.softmax(logits, dim=1)
        
        # 1. 动态获取当前预测类别 [B, H, W] 与置信度
        confidences, preds = torch.max(probs, dim=1)
        
        # 2. 筛选高置信度误分类像素的掩码
        valid_mask = (targets != self.ignore_index)
        error_mask = (preds != targets) & valid_mask
        overconfident_mask = (confidences > self.confidence_threshold) & error_mask
        
        num_hard_pixels = overconfident_mask.sum()
        if num_hard_pixels == 0:
            # 安全返回，避免无硬像素时梯度报错
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # 3. 提取这些硬像素对应的预测值、真实标签和 log_softmax 值
        # 形状: [N]
        flat_preds = preds[overconfident_mask].unsqueeze(1)    # 预测类 p
        flat_targets = targets[overconfident_mask].unsqueeze(1).to(torch.int64) # 真实类 t
        
        # 形状: [N, 3]
        flat_logits = logits.permute(0, 2, 3, 1)[overconfident_mask]
        log_probs = F.log_softmax(flat_logits, dim=-1)
        
        # 4. 提取 log P(t) 和 log P(p)
        # 形状: [N, 1]
        log_p_true = torch.gather(log_probs, dim=1, index=flat_targets)
        log_p_pred = torch.gather(log_probs, dim=1, index=flat_preds)
        
        # 5. 精准拉拢：只让真实类与预测类的概率对齐到 0.5 (即 Logit 平等拉扯)
        # 根据 KL 散度公式展开：-0.5 * log P(t) - 0.5 * log P(p)
        # 这会迫使无关类的概率压低到 0，而 t 和 p 均分概率
        pixel_losses = -0.5 * (log_p_true + log_p_pred)
        loss = pixel_losses.mean() * self.weight
        # 返回当前 Batch 内硬像素的均值损失
        return loss



class HDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_distance_map(self, gt_bin_np: np.ndarray):
        """
        gt_bin_np: [B,H,W] 0/1 numpy
        return dist_map [B,H,W]: 背景点→最近前景边界距离
        """
        B = gt_bin_np.shape[0]
        dists = []
        for b in range(B):
            # 1-mask：背景区域，计算背景到病灶的距离
            dist = distance_transform_edt(1 - gt_bin_np[b])
            dists.append(dist)
        return np.stack(dists, axis=0)

    def forward(self, logits: torch.Tensor, gt: torch.Tensor, target_cls=2):
        """
        logits: [B, C, H, W] 网络原始输出(未softmax)
        gt:     [B, H, W] 标签 0/1/2
        target_cls: 只对肿瘤2施加HD约束（肝脏1不加）
        """
        if gt.ndim == 4:
            gt = gt[:,0]
        B, C, H, W = logits.shape
        # 当前类别概率
        prob = F.softmax(logits, dim=1)[:, target_cls]  # [B,H,W]
        # 生成二值GT
        gt_bin = (gt == target_cls).float()

        # CPU生成距离图（不参与梯度）
        gt_np = gt_bin.detach().cpu().numpy().astype(np.uint8)
        dist_np = self.get_distance_map(gt_np)
        dist_map = torch.from_numpy(dist_np).to(prob.device).float()

        # 核心loss：在GT外面(dist>0)预测越高，loss越大
        loss = (prob * dist_map).mean()
        return loss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_CE_LogitNorm_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustLogitNormCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_LogitNorm_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else net_output.sum() * 0.0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class DC_and_CE_SVLS_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustSVLSCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_SVLS_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else net_output.sum() * 0.0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class DC_and_CE_HardL1ACE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, ace_kwargs, weight_ce=1, weight_dice=1, loss_weight=0.1,
                 ignore_label=None, dice_class=SoftDiceLoss):
        super().__init__()
        self.main_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )

        if ignore_label is not None:
            ace_kwargs = dict(ace_kwargs)
            ace_kwargs['ignore_index'] = ignore_label

        self.ace = HardL1ACELoss(**ace_kwargs)
        self.loss_weight = float(loss_weight)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        main_loss = self.main_loss(net_output, target)
        ace_loss = self.ace(net_output, target[:, 0])
        return main_loss + self.loss_weight * ace_loss


class DC_and_CE_SoftL1ACE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, ace_kwargs, weight_ce=1, weight_dice=1, loss_weight=0.1,
                 ignore_label=None, dice_class=SoftDiceLoss):
        super().__init__()
        self.main_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )

        if ignore_label is not None:
            ace_kwargs = dict(ace_kwargs)
            ace_kwargs['ignore_index'] = ignore_label

        self.ace = SoftL1ACELoss(**ace_kwargs)
        self.loss_weight = float(loss_weight)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        main_loss = self.main_loss(net_output, target)
        ace_loss = self.ace(net_output, target[:, 0])
        return main_loss + self.loss_weight * ace_loss


class DC_and_CE_MEEP_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, meep_kwargs, weight_ce=1, weight_dice=1, loss_weight=0.1,
                 ignore_label=None, dice_class=SoftDiceLoss):
        super().__init__()
        self.main_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )

        if ignore_label is not None:
            meep_kwargs = dict(meep_kwargs)
            meep_kwargs['ignore_index'] = ignore_label

        self.meep = MEEPLoss(**meep_kwargs)
        self.loss_weight = float(loss_weight)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        main_loss = self.main_loss(net_output, target)
        meep_loss = self.meep(net_output, target[:, 0])
        return main_loss + self.loss_weight * meep_loss


class DC_and_CE_multi_anchor_relax_loss(nn.Module):
    def __init__(self, safe_ce_kwargs, weight_ce=1, weight_dice=0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = MultiAnchorSafeRegionCELoss(**safe_ce_kwargs)

    def forward(
        self,
        class_logits: torch.Tensor,
        anchor_scores: torch.Tensor,
        target: torch.Tensor,
        anchors: torch.Tensor,
        bias: torch.Tensor,
        anchor_classes: torch.Tensor,
    ):
        ce_loss = self.ce(class_logits, anchor_scores, target, anchors, bias, anchor_classes) \
            if self.weight_ce != 0 else class_logits.sum() * 0.0
        return self.weight_ce * ce_loss


class DC_and_CE_MbLS_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustMbLSCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_MbLS_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else net_output.sum() * 0.0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class CE_and_ConfFP_Tversky_loss(nn.Module):
    def __init__(self, soft_tversky_kwargs, ce_kwargs, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = float(soft_tversky_kwargs.get("alpha", 0.3))
        self.beta = float(soft_tversky_kwargs.get("beta", 0.7))
        self.rho = float(soft_tversky_kwargs.get("rho", 2.0))
        self.gamma_weight = float(soft_tversky_kwargs.get("gamma_weight", 2.0))
        self.w_max = float(soft_tversky_kwargs.get("w_max", 3.0))
        self.eps = float(soft_tversky_kwargs.get("eps", 1e-6))
        self.last_debug = {}

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "ignore label expects target shape [B,1,...]"
            mask = target != self.ignore_label
            target_masked = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            mask = None
            target_masked = target
            num_fg = None

        ce_loss = self.ce(net_output, target[:, 0]) if (self.ignore_label is None or num_fg > 0) else net_output.new_zeros(())

        probs = F.softmax(net_output.float(), dim=1)
        target_long = target_masked[:, 0].long()
        y_onehot = F.one_hot(target_long, num_classes=net_output.shape[1]).movedim(-1, 1).float()

        if mask is not None:
            valid_mask = mask.float()
            probs = probs * valid_mask
            y_onehot = y_onehot * valid_mask
        else:
            valid_mask = None

        reduce_dims = tuple(i for i in range(net_output.ndim) if i != 1)
        fp_weight = 1.0 + self.rho * (probs.detach() ** self.gamma_weight)
        fp_weight = torch.clamp(fp_weight, max=self.w_max)

        tp = torch.sum(y_onehot * probs, dim=reduce_dims)
        fn = torch.sum(y_onehot * (1.0 - probs), dim=reduce_dims)
        fp = torch.sum((1.0 - y_onehot) * fp_weight * probs, dim=reduce_dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        loss_conf_fp_tversky = torch.mean(1.0 - tversky)
        total_loss = ce_loss + loss_conf_fp_tversky

        if valid_mask is not None:
            valid_weights = fp_weight[valid_mask.expand_as(fp_weight) > 0]
            mean_fp_weight = valid_weights.mean() if valid_weights.numel() > 0 else net_output.new_zeros(())
            max_fp_weight = valid_weights.max() if valid_weights.numel() > 0 else net_output.new_zeros(())
        else:
            mean_fp_weight = fp_weight.mean()
            max_fp_weight = fp_weight.max()

        self.last_debug = {
            "loss_ce": ce_loss.detach(),
            "loss_conf_fp_tversky": loss_conf_fp_tversky.detach(),
            "mean_fp_weight": mean_fp_weight.detach(),
            "max_fp_weight": max_fp_weight.detach(),
        }
        return total_loss
    

class ETFRegularizer(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, loss_weight: float = 0.75):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.loss_weight = loss_weight
        self.register_buffer('W_etf', self._generate_simplex_etf())
        
    def _generate_simplex_etf(self) -> torch.Tensor:
        K = self.num_classes
        D = self.embed_dim
        I = torch.eye(K)
        ones = torch.ones(K, K)
        M = torch.sqrt(torch.tensor(K / (K - 1))) * (I - 1.0 / K * ones)
        if D >= K:
            rand = torch.randn(D, K)
            q, _ = torch.qr(rand)
            W_etf = q @ M
        else:
            W_etf = M[:D, :]
        W_etf = F.normalize(W_etf, p=2, dim=0)
        return W_etf.detach()

    @torch.no_grad()
    def compute_class_centers(self, feats, labels):
        K = self.num_classes
        D = feats.size(1)
        centers = torch.zeros(K, D, device=feats.device)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = feats[mask].mean(dim=0)
        return centers

    def forward(self, feat_map, seg_label):
        # feat_map: B, C, H, W  (来自 UNet decoder 最后一层)
        # seg_label: B, 1, H, W
        B, C, H, W = feat_map.shape
        feats = feat_map.permute(0,2,3,1).reshape(-1, C)  # (B*H*W, C)
        labels = seg_label.reshape(-1)                     # (B*H*W)
        centers = self.compute_class_centers(feats, labels)
        logits_etf = centers @ self.W_etf
        target = torch.arange(self.num_classes, device=feats.device)
        loss = F.cross_entropy(logits_etf, target)
        return self.loss_weight * loss
    



class CrossImagePrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, min_pixels=50, loss_weight=1.0, num_classes=3, metric='cosine'):
        super().__init__()
        self.temperature = temperature
        self.min_pixels = min_pixels
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.metric = metric.lower()

    def forward(self, features, labels):
        B, C, H, W = features.shape
        if labels.ndim == 4:
            labels = labels[:, 0]

        protos =[]
        proto_labels = []

        # 1. 提取每个类的原型
        for b in range(B):
            feat_b = features[b].view(C, -1)
            label_b = labels[b].view(-1)
            for cls_id in range(self.num_classes):
                mask = (label_b == cls_id)
                if mask.sum() > self.min_pixels:
                    proto = feat_b[:, mask].mean(dim=1)
                    protos.append(proto)
                    proto_labels.append(cls_id)

        if len(protos) < 2:
            return torch.tensor(0.0, device=features.device)

        Z = torch.stack(protos, dim=0)  # [M, C]
        Y = torch.tensor(proto_labels, device=Z.device)
        M = Z.size(0)

        # 2. 相似度计算 (修复归一化 Bug)
        if self.metric == 'cosine':
            # 👑 必须先进行 L2 归一化，否则 matmul 只是点积！
            Z_norm = F.normalize(Z, p=2, dim=1)
            sim = torch.matmul(Z_norm, Z_norm.t()) / self.temperature
        else:
            # 欧氏距离模式
            dist = torch.sum((Z.unsqueeze(1) - Z.unsqueeze(0)) ** 2, dim=-1)
            sim = -dist / self.temperature

        # 3. 掩码逻辑 (你的完美逻辑，保持不变)
        pos_mask = (Y.unsqueeze(0) == Y.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # 自己不算

        lesion_mask = (Y == 2)
        pos_mask[lesion_mask] = 0.0
        pos_mask[:, lesion_mask] = 0.0

        # 负样本：除了自己和正样本之外的所有人
        neg_mask = 1.0 - pos_mask - torch.eye(M, device=Z.device)
        
        # 有效样本掩码：分母只算正样本和负样本 (剔除自己)
        valid_mask = pos_mask + neg_mask

        # =========================================================
        # 👑 4. 标准 SupCon InfoNCE (解决 NaN 爆炸 与 梯度偷懒)
        # =========================================================
        # 为了数值稳定性，减去每一行的最大值 (LogSumExp Trick)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_shifted = sim - sim_max.detach() # detach 很关键，防梯度阻断

        # 计算 exp
        exp_sim = torch.exp(sim_shifted) * valid_mask

        # 计算 log_prob: log( exp(sim_pos) / sum(exp_sim) )
        # = sim_pos - log(sum(exp_sim))
        # 加上 1e-8 防止除以 0 (当某行 valid_mask 全为 0 时)
        log_prob = sim_shifted - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 提取正样本的 log_prob
        pos_log_prob = log_prob * pos_mask
        
        # 每行的正样本个数
        num_pos = pos_mask.sum(dim=1)
        valid_rows = (num_pos > 0)

        if not valid_rows.any():
            return torch.tensor(0.0, device=Z.device)

        # SupCon Loss: 对所有正样本的 log_prob 取平均，然后再按行取平均
        # 公式: Loss_i = - 1/|Pos| * sum(log_prob_pos)
        loss = - (pos_log_prob[valid_rows].sum(dim=1) / num_pos[valid_rows]).mean()

        return loss * self.loss_weight
    


class GlobalPrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_classes=3, loss_weight=1.0, min_pixels=50, valid_lesion=False):
        """
        temperature: InfoNCE 的温度缩放系数，控制 Softmax 锐度 (0.07~0.1 最佳)
        num_classes: 类别总数
        lesion_classes: 病灶在标签中的 index 列表。
                        【神级先验】：病灶形态千奇百怪，绝不能强迫它们聚成一点！
                        所以遇到这些类别，我们只推远它们与正常组织的距离，不产生向心拉力！
        """
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.min_pixels = min_pixels
        self.valid_lesion = valid_lesion

    def forward(self, features, labels, global_protos, valid_mask=None):
        """
        features: [B, C, H, W] 当前批次提取的密集特征 (建议归一化前)
        labels: [B, H, W] 当前批次的像素级细化标签 (如 0~6)
        global_protos: [K, C] 你通过 EMA 维护的全局 K 个类原型
        valid_mask: [B, H, W] 可选，核心像素掩码。建议只让高置信度的像素参与对比，防噪。
        """
        if labels.ndim == 4:
            labels = labels[:, 0]
        B, C, H, W = features.shape
                
        protos =[]
        proto_labels =[]
        for b in range(B):
            feat_b = features[b].view(C, -1)
            label_b = labels[b].view(-1)
            
            if valid_mask is not None:
                v_mask = valid_mask[b].view(-1)
            else:
                v_mask = torch.ones_like(label_b, dtype=torch.bool)

            for cls_id in range(self.num_classes):
                mask = (label_b == cls_id) & v_mask
                if mask.sum() > self.min_pixels:
                    # 👑 对该切片的该类像素求均值，得到宏观特征
                    proto = feat_b[:, mask].mean(dim=1)
                    protos.append(proto)
                    proto_labels.append(cls_id)
        if len(protos) == 0:
            return torch.tensor(0.0, device=features.device)
        Z = torch.stack(protos, dim=0).to(torch.float32)  # [M, C]
        Y = torch.tensor(proto_labels, device=Z.device, dtype=torch.long)
        P = global_protos.to(torch.float32).detach()      #[K, C] 固定靶

        Z = F.normalize(Z, p=2, dim=1)    # [M, C]
        P = F.normalize(P, p=2, dim=1)    # [K, C]

        # [N, C] @ [C, K] -> [N, K] 
        sim_matrix = torch.matmul(Z, P.t()) / self.temperature

        # 正样本掩码：当前像素标签对应的那个原型
        # one_hot 会生成[N, K] 的 0/1 矩阵
        pos_mask = F.one_hot(Y, num_classes=self.num_classes).float()
        
        if not self.valid_lesion:
            le_cls = 2
            is_lesion = (Y == le_cls)
            pos_mask[is_lesion, :] = 0.0

        # 找每行的最大值，防止 exp(x) 溢出
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_shifted = sim_matrix - sim_max.detach()

        # 分母：当前像素到所有 K 个原型的 exp 相似度之和
        exp_sim = torch.exp(sim_shifted)
        denominator = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        
        # log 概率：sim - log(sum(exp))
        log_prob = sim_shifted - torch.log(denominator)

        # 只提取正样本处的 log 概率
        pos_log_prob = log_prob * pos_mask
        
        # 计算每一行的正样本数 (病灶行的 num_pos 为 0)
        num_pos = pos_mask.sum(dim=1)
        valid_rows = num_pos > 0

        if not valid_rows.any():
            return torch.tensor(0.0, device=features.device)

        # 最终 Loss：只对有正样本的行（正常肝脏和背景）计算 Pull 梯度
        # 但所有的像素（包括病灶）在上面算 denominator 时，都产生了强大的 Push 梯度！
        loss = - (pos_log_prob[valid_rows].sum(dim=1) / num_pos[valid_rows]).mean()

        return loss * self.loss_weight

class SliceGlobalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, lesion_classes=[2], loss_weight=1.0, min_pixels=10):
        super().__init__()
        self.temperature = temperature
        self.lesion_classes = lesion_classes
        self.loss_weight = loss_weight
        self.min_pixels = min_pixels # 降低阈值，因为我们只算均值

    def forward(self, features, labels, global_protos, valid_mask=None):
        if labels.ndim == 4:
            labels = labels[:, 0]
            
        B, C, H, W = features.shape

        protos =[]
        proto_labels =[]

        # ====================================================
        # 1. 核心纠正：执行【切片级】的特征池化 (Slice-level Pooling)
        # 抹平像素级高频噪声，提取纯正的宏观解剖/域特征
        # ====================================================
        for b in range(B):
            feat_b = features[b].view(C, -1)
            label_b = labels[b].view(-1)
            
            # 如果有形态学的 Core Mask，可以进一步提纯
            if valid_mask is not None:
                v_mask = valid_mask[b].view(-1)
            else:
                v_mask = torch.ones_like(label_b, dtype=torch.bool)

            K = global_protos.shape[0]
            for cls_id in range(K):
                mask = (label_b == cls_id) & v_mask
                if mask.sum() > self.min_pixels:
                    # 👑 对该切片的该类像素求均值，得到宏观特征
                    proto = feat_b[:, mask].mean(dim=1)
                    protos.append(proto)
                    proto_labels.append(cls_id)

        if len(protos) == 0:
            return torch.tensor(0.0, device=features.device)

        # M 是当前 Batch 内有效提取出的切片级原型数量 (M <= B * K)
        Z = torch.stack(protos, dim=0).to(torch.float32)  # [M, C]
        Y = torch.tensor(proto_labels, device=Z.device, dtype=torch.long)
        P = global_protos.to(torch.float32).detach()      #[K, C] 固定靶
        M = Z.shape[0]

        # 2. 超球面投影 (消除绝对光照和对比度的干扰)
        Z = F.normalize(Z, p=2, dim=1)
        P = F.normalize(P, p=2, dim=1)

        # 3. 计算切片级特征 到 全局原型 的相似度矩阵
        sim_matrix = torch.matmul(Z, P.t()) / self.temperature # [M, K]

        # 4. 将提取出的原型按医学先验分轨
        is_lesion = torch.zeros(M, dtype=torch.bool, device=Z.device)
        for le_cls in self.lesion_classes:
            is_lesion |= (Y == le_cls)
        is_normal = ~is_lesion

        loss = 0.0
        valid_loss_terms = 0

        # ====================================================
        # 轨道 A：正常类 InfoNCE (找自己的全局，排斥其他人)
        # ====================================================
        if is_normal.sum() > 0:
            sim_normal = sim_matrix[is_normal] #[M_norm, K]
            Y_normal = Y[is_normal]
            
            pos_mask = F.one_hot(Y_normal, num_classes=K).float()
            
            # LogSumExp 技巧防 NaN
            sim_max, _ = torch.max(sim_normal, dim=1, keepdim=True)
            sim_shifted = sim_normal - sim_max.detach()
            
            exp_sim = torch.exp(sim_shifted)
            denominator = exp_sim.sum(dim=1, keepdim=True) + 1e-8
            
            log_prob = sim_shifted - torch.log(denominator)
            pos_log_prob = log_prob * pos_mask
            
            loss_normal = -pos_log_prob.sum(dim=1).mean()
            loss += loss_normal
            valid_loss_terms += 1

        # ====================================================
        # 轨道 B：病灶类 纯粹排斥 Loss (The Mathematical Fix)
        # ====================================================
        if is_lesion.sum() > 0:
            sim_lesion = sim_matrix[is_lesion] # [M_les, K]
            
            # 找出所有的正常类的索引
            normal_classes =[c for c in range(K) if c not in self.lesion_classes]
            
            if len(normal_classes) > 0:
                # 取出病灶到正常原型的相似度
                sim_lesion_to_normals = sim_lesion[:, normal_classes] # [M_les, K_norm]
                
                # 👑 神级排斥法则：直接最小化相似度的 LogSumExp
                # 迫使网络将病灶特征推到离所有正常原型越远越好的死角！
                sim_max_le, _ = torch.max(sim_lesion_to_normals, dim=1, keepdim=True)
                sim_shifted_le = sim_lesion_to_normals - sim_max_le.detach()
                
                # LogSumExp 等价于平滑的 Max。最小化它，就是最小化病灶与最像的正常类的相似度
                loss_lesion = (torch.log(torch.exp(sim_shifted_le).sum(dim=1) + 1e-8) + sim_max_le.squeeze(1)).mean()
                
                loss += loss_lesion
                valid_loss_terms += 1

        if valid_loss_terms > 0:
            return (loss / valid_loss_terms) * self.loss_weight
        else:
            return torch.tensor(0.0, device=features.device)

class MSELoss(nn.Module):
    def __init__(self, temperature=0.1, min_pixels=50, weight_mse=1.0, lesion_classes=[2], valid_lesion=False):
        """
        temperature: 控制 InfoNCE 梯度的尖锐度 (推荐 0.07~0.1)
        lesion_classes: 病灶类别列表。遇到病灶，不计算 MSE，在 InfoNCE 中也只推远不拉近。
        weight_nce: 对比学习 Loss 的权重 (约束相对方向/流形骨架)
        weight_mse: MSE Loss 的权重 (约束绝对物理幅度/亮度对齐)
        """
        super().__init__()
        self.temperature = temperature
        self.min_pixels = min_pixels
        self.weight_mse = weight_mse
        self.num_classes = 3 # 假设为 3 分类
        self.lesion_classes = lesion_classes
        self.valid_lesion = valid_lesion

    def forward(self, features, labels, global_protos):
        """
        features:[B, C, H, W] - 网络提取的特征
        labels: [B, (1), H, W] - 真实标签
        global_protos: [K, C] - 后台 EMA 维护的全局原型，或者外部传入的标准锚点
        """
        B, C, H, W = features.shape
        if labels.ndim == 4:
            labels = labels[:, 0]

        protos = []
        proto_labels =[]

        # ====================================================
        # 1. 切片级池化 (Slice-level Pooling) —— 提取纯净宏观底色
        # ====================================================
        for b in range(B):
            feat_b = features[b].view(C, -1)
            label_b = labels[b].view(-1)
            for cls_id in range(self.num_classes):
                mask = (label_b == cls_id)
                if mask.sum() > self.min_pixels:
                    # 求当前切片、当前类的平均特征
                    proto = feat_b[:, mask].mean(dim=1)
                    protos.append(proto)
                    proto_labels.append(cls_id)

        # 如果当前 Batch 连 1 个有效切片特征都没提取出来，直接返回 0
        if len(protos) == 0:
            return torch.tensor(0.0, device=features.device), torch.tensor(0.0, device=features.device)

        # Z 是当前 Batch 内提取出的所有切片级特征
        Z = torch.stack(protos, dim=0)  # [M, C]
        Y = torch.tensor(proto_labels, device=Z.device) # [M]
        M = Z.size(0)
        pos_mask = F.one_hot(Y, num_classes=self.num_classes).float()
        # 🚨 医学先验：病灶绝对不拉向中心，只负责推开正常原型！
        for le_cls in self.lesion_classes:
            is_lesion = (Y == le_cls)
            pos_mask[is_lesion, :] = 0.0
        num_pos = pos_mask.sum(dim=1)
        valid_rows = num_pos > 0

        # 👑 锁定全局原型作为“绝对真理靶子”，绝不在 Loss 里接收梯度！
        P = global_protos.detach() # [K, C]
        loss_mse = torch.tensor(0.0, device=Z.device)
        if self.weight_mse > 0:
            # 只有正常组织（有正样本的行）才允许进行绝对数值的对齐！
            # 病灶的绝对特征千奇百怪，强行 MSE 会导致特征撕裂。
            if self.valid_lesion:
                valid_rows = torch.ones_like(valid_rows, dtype=torch.bool) # 如果病灶也要对齐，就全都算
            if valid_rows.any():
                # 取出当前 Batch 里所有的正常肝脏和背景切片特征
                Z_normal = Z[valid_rows] 
                Y_normal = Y[valid_rows]
                
                # 取出它们对应的全局绝对原型
                P_target = P[Y_normal]
                if P_target.ndim == 3:
                    #(M, L, C)
                    dist = torch.norm(Z_normal.unsqueeze(1) - P_target, dim=2) #(M, L)
                    _, min_idx = dist.min(dim=1)
                    P_target = P_target[torch.arange(P_target.size(0)), min_idx]
                
                # 👑 暴力对齐绝对数值：不管你在哪个机器上拍的，
                # 你的切片均值必须死死锚定在全局原型的绝对位置上！
                # dist = torch.norm(Z_normal - P_target, dim=1)
                # loss_mse = dist.mean()
                loss_mse = F.mse_loss(Z_normal, P_target)

        # ====================================================
        # 4. 融合返回
        # ====================================================
        total_loss = self.weight_mse * loss_mse
        return total_loss

class GlobalPrototypeContrastiveAndMSELoss(nn.Module):
    def __init__(self, temperature=0.1, min_pixels=50, weight_nce=1.0, weight_mse=1.0, lesion_classes=[2], valid_lesion=False):
        """
        temperature: 控制 InfoNCE 梯度的尖锐度 (推荐 0.07~0.1)
        lesion_classes: 病灶类别列表。遇到病灶，不计算 MSE，在 InfoNCE 中也只推远不拉近。
        weight_nce: 对比学习 Loss 的权重 (约束相对方向/流形骨架)
        weight_mse: MSE Loss 的权重 (约束绝对物理幅度/亮度对齐)
        """
        super().__init__()
        self.temperature = temperature
        self.min_pixels = min_pixels
        self.weight_nce = weight_nce
        self.weight_mse = weight_mse
        self.num_classes = 3 # 假设为 3 分类
        self.lesion_classes = lesion_classes
        self.valid_lesion = valid_lesion

    def forward(self, features, labels, global_protos):
        """
        features:[B, C, H, W] - 网络提取的特征
        labels: [B, (1), H, W] - 真实标签
        global_protos: [K, C] - 后台 EMA 维护的全局原型，或者外部传入的标准锚点
        """
        B, C, H, W = features.shape
        if labels.ndim == 4:
            labels = labels[:, 0]

        protos = []
        proto_labels =[]

        # ====================================================
        # 1. 切片级池化 (Slice-level Pooling) —— 提取纯净宏观底色
        # ====================================================
        for b in range(B):
            feat_b = features[b].view(C, -1)
            label_b = labels[b].view(-1)
            for cls_id in range(self.num_classes):
                mask = (label_b == cls_id)
                if mask.sum() > self.min_pixels:
                    # 求当前切片、当前类的平均特征
                    proto = feat_b[:, mask].mean(dim=1)
                    protos.append(proto)
                    proto_labels.append(cls_id)

        # 如果当前 Batch 连 1 个有效切片特征都没提取出来，直接返回 0
        if len(protos) == 0:
            return torch.tensor(0.0, device=features.device), torch.tensor(0.0, device=features.device)

        # Z 是当前 Batch 内提取出的所有切片级特征
        Z = torch.stack(protos, dim=0)  # [M, C]
        Y = torch.tensor(proto_labels, device=Z.device) # [M]
        M = Z.size(0)

        # 👑 锁定全局原型作为“绝对真理靶子”，绝不在 Loss 里接收梯度！
        P = global_protos.detach() # [K, C]
        K = P.size(0)

        # ====================================================
        # 2. 全局 InfoNCE Loss (约束特征流形的相对拓扑与角度)
        # ====================================================
        # 正样本：Z[i] 对应的真实类别全局原型 P[Y[i]]
        pos_mask = F.one_hot(Y, num_classes=K).float()
        
        # 🚨 医学先验：病灶绝对不拉向中心，只负责推开正常原型！
        for le_cls in self.lesion_classes:
            is_lesion = (Y == le_cls)
            pos_mask[is_lesion, :] = 0.0
        num_pos = pos_mask.sum(dim=1)
        valid_rows = num_pos > 0

        loss_nce = torch.tensor(0.0, device=Z.device)
        if self.weight_nce > 0:
            Z_norm = F.normalize(Z, p=2, dim=1)
            P_norm = F.normalize(P, p=2, dim=1)
            # [M, C] @ [C, K] -> [M, K]
            sim = torch.matmul(Z_norm, P_norm.t()) / self.temperature

            # LogSumExp 稳定计算
            sim_max, _ = torch.max(sim, dim=1, keepdim=True)
            sim_shifted = sim - sim_max.detach()

            exp_sim = torch.exp(sim_shifted)
            # 分母：计算当前切片特征与 所有 K 个全局原型 的相似度总和
            denominator = exp_sim.sum(dim=1, keepdim=True) + 1e-8
            
            log_prob = sim_shifted - torch.log(denominator)
            pos_log_prob = log_prob * pos_mask
            
            if valid_rows.any():
                loss_nce = - (pos_log_prob[valid_rows].sum(dim=1) / num_pos[valid_rows]).mean()
            else:
                loss_nce = torch.tensor(0.0, device=Z.device)

        # ====================================================
        # 3. 绝对 MSE Loss (对齐亮度和幅度的绝对物理标尺)
        # ====================================================
        loss_mse = torch.tensor(0.0, device=Z.device)
        if self.weight_mse > 0:
            # 只有正常组织（有正样本的行）才允许进行绝对数值的对齐！
            # 病灶的绝对特征千奇百怪，强行 MSE 会导致特征撕裂。
            if self.valid_lesion:
                valid_rows = torch.ones_like(valid_rows, dtype=torch.bool) # 如果病灶也要对齐，就全都算
            if valid_rows.any():
                # 取出当前 Batch 里所有的正常肝脏和背景切片特征
                Z_normal = Z[valid_rows] 
                Y_normal = Y[valid_rows]
                
                # 取出它们对应的全局绝对原型
                P_target = P[Y_normal]
                if P_target.ndim == 3:
                    #(M, L, C)
                    dist = torch.norm(Z_normal.unsqueeze(1) - P_target, dim=2)
                    _, min_idx = dist.min(dim=1)
                    P_target = P_target[torch.arange(P_target.size(0)), min_idx]
                
                # 👑 暴力对齐绝对数值：不管你在哪个机器上拍的，
                # 你的切片均值必须死死锚定在全局原型的绝对位置上！
                # dist = torch.norm(Z_normal - P_target, dim=1)
                # loss_mse = dist.mean()
                loss_mse = F.mse_loss(Z_normal, P_target)

        # ====================================================
        # 4. 融合返回
        # ====================================================
        total_loss = self.weight_nce * loss_nce + self.weight_mse * loss_mse
        return total_loss
    

class ClassAdaptivePooledLoss(nn.Module):
    def __init__(self, pool_sizes_dict, ignore_label=255):
        """
        pool_sizes_dict: 字典形式，控制每一类的池化程度。
                         例如: {0: 4, 1: 2, 2: 1} 
                         代表背景 4x4 融为一票，肝脏 2x2 融为一票，病灶 1x1 (不池化，保留每票)
        """
        super().__init__()
        self.pool_sizes = pool_sizes_dict
        self.ignore_label = ignore_label

    def forward(self, logits, targets, valid_mask=None):
        """
        logits: [B, C, H, W] - 网络的原始输出
        targets:[B, H, W]    - 真实标签
        valid_mask:[B, H, W] - 你计算出的及格掩码 (难样本为 1，进圈的易样本为 0)
        """
        B, C, H, W = logits.shape
        if targets.ndim == 4:
            targets = targets[:, 0]
        if valid_mask is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        # 用来收集所有类别的最终“选票”
        all_pooled_logits =[]
        all_pooled_targets =[]
        
        for cls_id, p_size in self.pool_sizes.items():
            # 1. 构造当前类别的专属严苛掩码
            # 必须同时满足：是这个类 + 是难样本(valid) + 不是 ignore 边缘
            cls_mask = (targets == cls_id) & valid_mask & (targets != self.ignore_label)
            cls_mask_float = cls_mask.float().unsqueeze(1) # [B, 1, H, W]
            
            if cls_mask_float.sum() == 0:
                continue # 这个 Batch 里没有该类的难样本，直接跳过
                
            # 2. 如果池化尺寸为 1，说明不需要池化（如病灶），保留像素级极致微操！
            if p_size == 1:
                # [B, C, H, W] ->[B, H, W, C] -> 提取 Mask 处的特征 -> [N, C]
                valid_logits = logits.permute(0, 2, 3, 1)[cls_mask]
                valid_targets = torch.full((valid_logits.shape[0],), cls_id, dtype=torch.long, device=logits.device)
                
                all_pooled_logits.append(valid_logits)
                all_pooled_targets.append(valid_targets)
                
            # 3. 👑 如果需要池化（如背景、肝脏），执行神级 Mask-Aware Pooling！
            else:
                # 分子: Logit * Mask 的均值 (无效区域被当成 0)
                num = F.avg_pool2d(logits * cls_mask_float, kernel_size=p_size, stride=p_size) # [B, C, H', W']
                # 分母: Mask 的均值 (即池化块内有效像素的比例)
                den = F.avg_pool2d(cls_mask_float, kernel_size=p_size, stride=p_size)          #[B, 1, H', W']
                
                # 找出哪些池化块里面，至少包含了 1 个当前类的有效像素
                valid_windows = (den > 0).squeeze(1) # [B, H', W']
                
                if valid_windows.sum() > 0:
                    # 执行除法，还原真实的有效均值 (加上 clamp 防止浮点误差除以 0)
                    pooled_logits = num / torch.clamp(den, min=1e-7) # [B, C, H', W']
                    
                    # 把这些计算好的宏观 Logit 提取出来
                    valid_pooled_logits = pooled_logits.permute(0, 2, 3, 1)[valid_windows] #[M, C]
                    valid_pooled_targets = torch.full((valid_pooled_logits.shape[0],), cls_id, dtype=torch.long, device=logits.device)
                    
                    all_pooled_logits.append(valid_pooled_logits)
                    all_pooled_targets.append(valid_pooled_targets)

        # 4. 如果全图没有任何需要计算的特征
        if len(all_pooled_logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            

        final_logits = torch.cat(all_pooled_logits, dim=0)   # [Total_Votes, C]
        final_targets = torch.cat(all_pooled_targets, dim=0) # [Total_Votes]
        
        loss = F.cross_entropy(final_logits, final_targets, reduction='mean')
        
        return loss


class CrossSliceContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, min_pixels=50, num_classes=3, lesion_classes=[2]):
        super().__init__()
        self.temperature = temperature
        self.min_pixels = min_pixels
        self.lesion_classes = lesion_classes # 医学神级先验：病灶绝对不拉近！
        self.num_classes = num_classes

    def forward(self, features, labels, memory_bank):
        """
        features: [B, C, H, W] - 当前 Batch 的高维特征图
        labels: [B, H, W]      - 真实标签
        memory_bank: PrototypeMemoryBank 实例
        """
        if labels.ndim == 4:
            labels = labels[:, 0]
        B, C, H, W = features.shape
        
        # 1. 提取当前 Batch 每张切片的【类池化特征】(Region Pooling)
        curr_protos = []
        curr_labels = []
        
        for b in range(B):
            feat_b = features[b].view(C, -1)
            label_b = labels[b].view(-1)
            
            for cls_id in range(self.num_classes):
                mask = (label_b == cls_id)
                if mask.sum() > self.min_pixels:
                    proto = feat_b[:, mask].mean(dim=1) # [C]
                    curr_protos.append(proto)
                    curr_labels.append(cls_id)
                    
        if len(curr_protos) == 0:
            return torch.tensor(0.0, device=features.device)
            
        Z_curr = torch.stack(curr_protos, dim=0) # [M_curr, C]
        Y_curr = torch.tensor(curr_labels, device=features.device)
        
        # 2. 从记忆库中请出“历史大军”
        Z_mem, Y_mem = memory_bank.get_memory()
        
        if Z_mem is None:
            # 记忆库为空（训练刚开始），只更新库，不算 Loss
            memory_bank.enqueue(Z_curr, Y_curr)
            return torch.tensor(0.0, device=features.device)
            
        # 3. 👑 投射到超球面 (极度重要：消除亮度和绝对幅度的剧烈波动)
        Z_curr = F.normalize(Z_curr, p=2, dim=1)
        Z_mem = F.normalize(Z_mem, p=2, dim=1)
        
        # 4. 算跨时空余弦相似度矩阵 [M_curr, M_mem]
        # 当前切片 vs 历史成百上千张切片！
        sim_matrix = torch.matmul(Z_curr, Z_mem.t()) / self.temperature
        
        # 5. 构造正负样本 Mask
        pos_mask = (Y_curr.unsqueeze(1) == Y_mem.unsqueeze(0)).float()
        
        # 👑 医学先验防御：病灶之间形态各异，绝对禁止类内拉近！
        # 强行拉近不同病人的病灶，会导致特征空间崩塌。
        for le_cls in self.lesion_classes:
            is_lesion_curr = (Y_curr == le_cls)
            pos_mask[is_lesion_curr, :] = 0.0 # 病灶不拥抱任何历史病灶，只排斥历史正常组织！

        # 负样本掩码：不同类别的就是敌人
        neg_mask = (Y_curr.unsqueeze(1) != Y_mem.unsqueeze(0)).float()
        
        # 6. 标准 InfoNCE 稳定计算 (LogSumExp Trick)
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_shifted = sim_matrix - sim_max.detach()
        
        exp_sim = torch.exp(sim_shifted)
        # 分母：自己同类的历史样本 + 所有异类的历史样本
        denominator = (exp_sim * (pos_mask + neg_mask)).sum(dim=1, keepdim=True) + 1e-8
        
        log_prob = sim_shifted - torch.log(denominator)
        
        pos_log_prob = log_prob * pos_mask
        num_pos = pos_mask.sum(dim=1)
        
        valid_rows = num_pos > 0
        if not valid_rows.any():
            # 顺手把当前特征送进记忆库
            memory_bank.enqueue(torch.stack(curr_protos, dim=0), Y_curr)
            return torch.tensor(0.0, device=features.device)
            
        # 最终的跨切片对比 Loss
        loss_nce = - (pos_log_prob[valid_rows].sum(dim=1) / num_pos[valid_rows]).mean()
        
        # 7. 结算完毕，把当前特征（归一化前的）存入历史记忆库，供下一个 Batch 使用
        memory_bank.enqueue(torch.stack(curr_protos, dim=0), Y_curr)
        
        return loss_nce


class IntraSliceCollapseLoss(nn.Module):
    def __init__(self, r_in=0.5, permit_classes = [], weights = []):
        super().__init__()
        self.r_in = r_in # 👑 及格万岁：给血管纹理留出的自由度
        assert len(permit_classes) == len(weights), "length of permit_classes must equal length od weights"
        self.permit_classes = permit_classes
        self.weights = weights

    def forward(self, feat, labels):
        """
        feat: [B, C, H, W] - 必须是未做 Batch 间融合的原始特征
        labels: [B, H, W]  - 真实标签 (0:Bg, 1:Liver, 2:Lesion)
        valid_mask: [B, H, W] - 之前说的核心区 Mask (只拉置信度高的)
        """
        if labels.ndim == 4:
            labels = labels[:,0]
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).transpose(1, 2) # [B, N, C]
        labels_flat = labels.view(B, -1)                # [B, N]
        
        total_loss = []
        # 👑 核心：严格以 Batch 中的“单张切片 (b)”为单位进行隔离操作！
        for c in self.permit_classes:
            valid_slices = 0
            cur_loss = 0.0
            for b in range(B):
                # 取出当前切片的 肝脏核心像素
                cls_mask = (labels_flat[b] == c)
                
                if cls_mask.sum() > 10:
                    # 1. 现场计算【当前切片的专属肝脏中心】
                    slice_center = feat_flat[b, cls_mask, :].mean(dim=0).detach() # [C] 冻结梯度，作为绝对锚点
                    
                    # 2. 提取当前切片的所有肝脏特征
                    slice_feats = feat_flat[b, labels_flat[b] == 1, :] # [M, C]
                    
                    # 3. 算距离 (只算自己跟自己的距离)
                    dist_to_own_center = torch.norm(slice_feats - slice_center.unsqueeze(0), dim=1)
                    
                    # 4. 切片级坍缩 (带 Margin，保纹理)
                    cur_loss += F.relu(dist_to_own_center - self.r_in).mean()
                    valid_slices += 1
            total_loss.append(cur_loss / valid_slices if valid_slices > 0 else 0.0)
        total_loss = torch.tensor(total_loss, device=feat.device)
        weights = torch.tensor(self.weights, device=feat.device).float()
        if len(self.permit_classes) > 0:
            return torch.dot(total_loss, weights)
        else:
            return torch.tensor(0.0, device=feat.device)

class DC_and_CE_relax_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_relax_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustRelaxLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.last_debug = {}

    def _build_relax_supervision_masks(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor,
                                       dismiss_label: list, permit_label: list):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            base_mask = target != self.ignore_label
            target_dice = torch.where(base_mask, target, 0)
            num_fg = base_mask.sum()
        else:
            target_dice = target
            base_mask = torch.ones_like(target, dtype=torch.bool)
            num_fg = None

        dismiss_mask = torch.ones_like(target_dice, dtype=torch.bool)
        for lbl in dismiss_label:
            dismiss_mask &= (target_dice != lbl)

        permit_mask = torch.zeros_like(target_dice, dtype=torch.bool)
        for lbl in permit_label:
            permit_mask |= (target_dice == lbl)

        target_dice = torch.where(base_mask & dismiss_mask, target_dice, 0)
        target_squeeze = target_dice[:, 0].long()
        true_logit = torch.gather(net_output, dim=1, index=target_squeeze.unsqueeze(1)).squeeze(1)
        true_thresh = thresh[target_squeeze]
        relax_mask = (true_logit <= true_thresh).unsqueeze(1)
        relax_mask[permit_mask] = 1

        active_mask = base_mask & dismiss_mask & relax_mask
        inactive_mask = base_mask & dismiss_mask & (~relax_mask)
        return target_dice, base_mask, active_mask, inactive_mask, num_fg

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor, dismiss_label: list=[], permit_label: list=[]):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        target_dice, base_mask, mask, _, num_fg = self._build_relax_supervision_masks(
            net_output, target, thresh, dismiss_label, permit_label
        )
        dc_loss = self.dc(net_output, target_dice, loss_mask=base_mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            **ce_debug,
        }
        return result


class DC_and_CE_relax_prior_dice_loss(DC_and_CE_relax_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientRelaxMaskDiceLoss):
        super(DC_and_CE_relax_prior_dice_loss, self).__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor, dismiss_label: list=[], permit_label: list=[]):
        target_dice, base_mask, active_mask, inactive_mask, num_fg = self._build_relax_supervision_masks(
            net_output, target, thresh, dismiss_label, permit_label
        )

        dc_loss = self.dc(
            net_output,
            target_dice,
            loss_mask=active_mask,
            inactive_mask=inactive_mask,
        ) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        dice_debug = getattr(self.dc, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            **ce_debug,
            **{f"dice_{k}": v for k, v in dice_debug.items()},
            **dice_debug,
        }
        return result


class DC_and_CE_relax_correct_bg_suppression_loss(DC_and_CE_relax_prior_dice_loss):
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor, dismiss_label: list=[], permit_label: list=[]):
        target_dice, base_mask, active_mask, inactive_mask, num_fg = self._build_relax_supervision_masks(
            net_output, target, thresh, dismiss_label, permit_label
        )

        target_labels = target_dice[:, 0].long()
        pred_labels = net_output.argmax(dim=1)
        correct_bg = (target_labels == 0) & (pred_labels == 0) & base_mask[:, 0].bool()
        correct_bg_mask = correct_bg[:, None]

        base_mask = base_mask & (~correct_bg_mask)
        active_mask = active_mask & (~correct_bg_mask)
        inactive_mask = inactive_mask & (~correct_bg_mask)

        dc_loss = self.dc(
            net_output,
            target_dice,
            loss_mask=active_mask,
            inactive_mask=inactive_mask,
        ) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        dice_debug = getattr(self.dc, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            "num_correct_bg": correct_bg.sum().detach(),
            **ce_debug,
            **{f"dice_{k}": v for k, v in dice_debug.items()},
            **dice_debug,
        }
        return result


def _compute_component_radius_np(component_mask: np.ndarray, min_core_ratio: float):
    structure = ndimage.generate_binary_structure(component_mask.ndim, 1)
    total = int(component_mask.sum())
    if total == 0:
        return 0, 0.0, component_mask.copy()

    best_radius = 0
    best_core = component_mask.copy()
    best_ratio = 1.0
    current = component_mask.copy()

    while True:
        eroded = ndimage.binary_erosion(current, structure=structure, iterations=1, border_value=0)
        eroded_count = int(eroded.sum())
        if eroded_count == 0:
            break
        ratio = eroded_count / total
        if ratio < min_core_ratio:
            break
        best_radius += 1
        best_core = eroded.copy()
        best_ratio = ratio
        current = eroded

    return best_radius, best_ratio, best_core


def _make_band_from_component_np(component_mask: np.ndarray, radius: int):
    structure = ndimage.generate_binary_structure(component_mask.ndim, 1)
    band_radius = max(int(radius), 1)
    dilated = ndimage.binary_dilation(component_mask, structure=structure, iterations=band_radius)
    eroded = ndimage.binary_erosion(component_mask, structure=structure, iterations=band_radius, border_value=0)
    inner_shell = np.logical_and(component_mask, np.logical_not(eroded))
    outer_shell = np.logical_and(dilated, np.logical_not(component_mask))
    return np.logical_or(inner_shell, outer_shell), band_radius


def _assign_band_to_classes_np(band_mask: np.ndarray, label: np.ndarray):
    return {
        "background": np.logical_and(band_mask, label == 0),
        "liver": np.logical_and(band_mask, label == 1),
        "lesion": np.logical_and(band_mask, label == 2),
    }


def _compute_relax_boundary_prior_mask(
    target: torch.Tensor,
    lesion_core_ratio_min: float = 0.6,
    liver_outer_width: int = 3,
    boundary_generation_mode: str = "2d",
    ignore_label: int | None = None,
    lesion_boundary_target_ratio: float | None = None,
    lesion_boundary_fill_fraction: float = 0.5,
):
    if target.ndim == 4:
        assert target.shape[1] == 1
        target = target[:, 0]
    target_np = target.detach().cpu().numpy().astype(np.int16)
    valid_np = np.ones_like(target_np, dtype=bool) if ignore_label is None else (target_np != ignore_label)
    safe_target_np = target_np.copy()
    safe_target_np[~valid_np] = 0

    boundary_np = np.zeros_like(safe_target_np, dtype=bool)

    if boundary_generation_mode not in {"2d", "2d_image", "3d"}:
        raise ValueError(f"Unsupported boundary_generation_mode: {boundary_generation_mode}")

    def _compute_single(label_arr: np.ndarray):
        lesion_mask = label_arr == 2
        liver_mask = label_arr == 1
        lesion_generated = {
            "lesion": np.zeros_like(lesion_mask, dtype=bool),
            "liver": np.zeros_like(lesion_mask, dtype=bool),
            "background": np.zeros_like(lesion_mask, dtype=bool),
        }
        liver_generated = {
            "lesion": np.zeros_like(lesion_mask, dtype=bool),
            "liver": np.zeros_like(lesion_mask, dtype=bool),
            "background": np.zeros_like(lesion_mask, dtype=bool),
        }

        lesion_components, num_components = ndimage.label(
            lesion_mask.astype(np.uint8),
            structure=ndimage.generate_binary_structure(lesion_mask.ndim, 1),
        )
        for comp_idx in range(1, num_components + 1):
            comp_mask = lesion_components == comp_idx
            radius, _, _ = _compute_component_radius_np(comp_mask, lesion_core_ratio_min)
            band_mask, band_radius = _make_band_from_component_np(comp_mask, radius)

            if lesion_boundary_target_ratio is not None:
                comp_total = int(comp_mask.sum())
                if comp_total > 0:
                    target_ratio = float(np.clip(lesion_boundary_target_ratio, 0.0, 1.0))
                    fill_fraction = float(np.clip(lesion_boundary_fill_fraction, 0.0, 1.0))
                    current_count = int(np.logical_and(band_mask, comp_mask).sum())
                    target_count_float = target_ratio * comp_total
                    max_target_count = int(np.floor(target_count_float + 1e-6))
                    if current_count < max_target_count:
                        desired_count = int(np.ceil(current_count + fill_fraction * (target_count_float - current_count)))
                        desired_count = min(desired_count, max_target_count)
                    else:
                        desired_count = current_count
                    adaptive_radius = int(band_radius)
                    while current_count < desired_count:
                        adaptive_radius += 1
                        candidate_band, _ = _make_band_from_component_np(comp_mask, adaptive_radius)
                        candidate_lesion = np.logical_and(candidate_band, comp_mask)
                        additional = np.logical_and(candidate_lesion, np.logical_not(band_mask))
                        additional_indices = np.argwhere(additional)
                        if additional_indices.size == 0:
                            break
                        needed = min(desired_count - current_count, max_target_count - current_count)
                        if needed <= 0:
                            break
                        selected = additional_indices[:needed]
                        band_mask[tuple(selected.T)] = True
                        current_count += selected.shape[0]

            class_masks = _assign_band_to_classes_np(band_mask, label_arr)
            lesion_generated["lesion"] |= class_masks["lesion"]
            lesion_generated["liver"] |= class_masks["liver"]
            lesion_generated["background"] |= class_masks["background"]

        if liver_outer_width > 0:
            structure = ndimage.generate_binary_structure(liver_mask.ndim, 1)
            liver_outer = np.logical_and(
                ndimage.binary_dilation(liver_mask, structure=structure, iterations=liver_outer_width),
                np.logical_not(liver_mask),
            )
            liver_inner = np.logical_and(
                liver_mask,
                np.logical_not(
                    ndimage.binary_erosion(liver_mask, structure=structure, iterations=liver_outer_width, border_value=0)
                ),
            )
            liver_band = np.logical_or(liver_outer, liver_inner)
            liver_generated["lesion"] |= np.logical_and(liver_band, label_arr == 2)
            liver_generated["liver"] |= np.logical_and(liver_band, label_arr == 1)
            liver_generated["background"] |= np.logical_and(liver_band, label_arr == 0)

        return (
            lesion_generated["lesion"]
            | lesion_generated["liver"]
            | lesion_generated["background"]
            | liver_generated["lesion"]
            | liver_generated["liver"]
            | liver_generated["background"]
        )

    if boundary_generation_mode == "3d" or boundary_generation_mode == "2d_image":
        for b in range(safe_target_np.shape[0]):
            boundary_np[b] = _compute_single(safe_target_np[b])
    else:
        for b in range(safe_target_np.shape[0]):
            for s in range(safe_target_np.shape[1]):
                boundary_np[b, s] = _compute_single(safe_target_np[b, s])

    boundary_np &= valid_np
    boundary_mask = torch.from_numpy(boundary_np).to(device=target.device, dtype=torch.bool)
    return boundary_mask.unsqueeze(1)


class DC_and_CE_relax_boundary_prior_loss(DC_and_CE_relax_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        ce_kwargs = dict(ce_kwargs)
        self.boundary_generation_mode = str(ce_kwargs.pop("boundary_generation_mode", "2d"))
        self.lesion_core_ratio_min = float(ce_kwargs.pop("lesion_core_ratio_min", 0.6))
        self.liver_outer_width = int(ce_kwargs.pop("liver_outer_width", 3))
        self.lesion_boundary_target_ratio = ce_kwargs.pop("lesion_boundary_target_ratio", None)
        if self.lesion_boundary_target_ratio is not None:
            self.lesion_boundary_target_ratio = float(self.lesion_boundary_target_ratio)
        self.lesion_boundary_fill_fraction = float(ce_kwargs.pop("lesion_boundary_fill_fraction", 0.5))
        super(DC_and_CE_relax_boundary_prior_loss, self).__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor, dismiss_label: list = [], permit_label: list = []):
        target_dice, base_mask, active_mask, inactive_mask, num_fg = self._build_relax_supervision_masks(
            net_output, target, thresh, dismiss_label, permit_label
        )

        num_active_before = active_mask.sum().float()
        boundary_mask = _compute_relax_boundary_prior_mask(
            target_dice,
            lesion_core_ratio_min=self.lesion_core_ratio_min,
            liver_outer_width=self.liver_outer_width,
            boundary_generation_mode=self.boundary_generation_mode,
            ignore_label=self.ignore_label,
            lesion_boundary_target_ratio=self.lesion_boundary_target_ratio,
            lesion_boundary_fill_fraction=self.lesion_boundary_fill_fraction,
        )
        boundary_mask = boundary_mask & base_mask
        active_mask = active_mask | boundary_mask
        inactive_mask = base_mask & (~active_mask)

        dc_loss = self.dc(net_output, target_dice, loss_mask=base_mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            "boundary_pixels": boundary_mask.sum().detach(),
            "num_active_before_boundary": num_active_before.detach(),
            "num_active_after_boundary": active_mask.sum().detach(),
            "boundary_generation_mode": self.boundary_generation_mode,
            "lesion_core_ratio_min": self.lesion_core_ratio_min,
            "liver_outer_width": self.liver_outer_width,
            "lesion_boundary_target_ratio": self.lesion_boundary_target_ratio,
            "lesion_boundary_fill_fraction": self.lesion_boundary_fill_fraction,
            **ce_debug,
        }
        return result


class DC_and_CE_relax_boundary_prior_dual_thresh_loss(DC_and_CE_relax_boundary_prior_loss):
    def _build_relax_supervision_masks_dual_thresh(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        thresh_core: torch.Tensor,
        thresh_boundary: torch.Tensor,
        dismiss_label: list,
        permit_label: list,
    ):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            base_mask = target != self.ignore_label
            target_dice = torch.where(base_mask, target, 0)
            num_fg = base_mask.sum()
        else:
            target_dice = target
            base_mask = torch.ones_like(target, dtype=torch.bool)
            num_fg = None

        dismiss_mask = torch.ones_like(target_dice, dtype=torch.bool)
        for lbl in dismiss_label:
            dismiss_mask &= (target_dice != lbl)

        permit_mask = torch.zeros_like(target_dice, dtype=torch.bool)
        for lbl in permit_label:
            permit_mask |= (target_dice == lbl)

        target_dice = torch.where(base_mask & dismiss_mask, target_dice, 0)
        boundary_mask = _compute_relax_boundary_prior_mask(
            target_dice,
            lesion_core_ratio_min=self.lesion_core_ratio_min,
            liver_outer_width=self.liver_outer_width,
            boundary_generation_mode=self.boundary_generation_mode,
            ignore_label=self.ignore_label,
            lesion_boundary_target_ratio=self.lesion_boundary_target_ratio,
            lesion_boundary_fill_fraction=self.lesion_boundary_fill_fraction,
        )
        boundary_mask = boundary_mask & base_mask

        target_squeeze = target_dice[:, 0].long()
        true_logit = torch.gather(net_output, dim=1, index=target_squeeze.unsqueeze(1)).squeeze(1)
        true_thresh_core = thresh_core[target_squeeze]
        true_thresh_boundary = thresh_boundary[target_squeeze]
        true_thresh = torch.where(boundary_mask[:, 0], true_thresh_boundary, true_thresh_core)
        relax_mask = (true_logit <= true_thresh).unsqueeze(1)
        relax_mask[permit_mask] = 1

        active_mask = base_mask & dismiss_mask & relax_mask
        inactive_mask = base_mask & dismiss_mask & (~relax_mask)
        return target_dice, base_mask, boundary_mask, active_mask, inactive_mask, num_fg

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        thresh_core: torch.Tensor,
        thresh_boundary: torch.Tensor,
        dismiss_label: list = [],
        permit_label: list = [],
    ):
        target_dice, base_mask, boundary_mask, active_mask, inactive_mask, num_fg = \
            self._build_relax_supervision_masks_dual_thresh(
                net_output,
                target,
                thresh_core,
                thresh_boundary,
                dismiss_label,
                permit_label,
            )

        num_active_before = active_mask.sum().float()
        dc_loss = self.dc(net_output, target_dice, loss_mask=base_mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            "boundary_pixels": boundary_mask.sum().detach(),
            "num_active_before_boundary": num_active_before.detach(),
            "num_active_after_boundary": active_mask.sum().detach(),
            "boundary_generation_mode": self.boundary_generation_mode,
            "lesion_core_ratio_min": self.lesion_core_ratio_min,
            "liver_outer_width": self.liver_outer_width,
            "lesion_boundary_target_ratio": self.lesion_boundary_target_ratio,
            "lesion_boundary_fill_fraction": self.lesion_boundary_fill_fraction,
            **ce_debug,
        }
        return result


class DC_and_CE_relax_boundary_prior_correct_bg_suppression_dual_thresh_loss(
    DC_and_CE_relax_boundary_prior_dual_thresh_loss
):
    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        thresh_core: torch.Tensor,
        thresh_boundary: torch.Tensor,
        dismiss_label: list = [],
        permit_label: list = [],
    ):
        target_dice, base_mask, boundary_mask, active_mask, inactive_mask, num_fg = \
            self._build_relax_supervision_masks_dual_thresh(
                net_output,
                target,
                thresh_core,
                thresh_boundary,
                dismiss_label,
                permit_label,
            )

        pred = net_output.argmax(dim=1, keepdim=True)
        correct_bg_mask = base_mask & (target_dice == 0) & (pred == 0)
        num_active_before_correct_bg = active_mask.sum().float()
        active_mask = active_mask & (~correct_bg_mask)
        inactive_mask = base_mask & (~active_mask)

        dc_loss = self.dc(net_output, target_dice, loss_mask=base_mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            "boundary_pixels": boundary_mask.sum().detach(),
            "num_active_before_correct_bg_suppression": num_active_before_correct_bg.detach(),
            "num_correct_bg": correct_bg_mask.sum().detach(),
            "num_active_after_correct_bg_suppression": active_mask.sum().detach(),
            "boundary_generation_mode": self.boundary_generation_mode,
            "lesion_core_ratio_min": self.lesion_core_ratio_min,
            "liver_outer_width": self.liver_outer_width,
            "lesion_boundary_target_ratio": self.lesion_boundary_target_ratio,
            "lesion_boundary_fill_fraction": self.lesion_boundary_fill_fraction,
            **ce_debug,
        }
        return result


class DC_and_CE_relax_bg_outside_fg_suppression_loss(DC_and_CE_relax_prior_dice_loss):
    def _apply_bg_outside_foreground_suppression(
        self,
        net_output: torch.Tensor,
        target_dice: torch.Tensor,
        thresh: torch.Tensor,
        base_mask: torch.Tensor,
        active_mask: torch.Tensor,
        inactive_mask: torch.Tensor,
        background_label: int = 0,
        foreground_labels: tuple = (1, 2),
    ):
        target_labels = target_dice[:, 0].long()
        bg_mask = base_mask & (target_dice == background_label)

        foreground_inside = torch.zeros_like(bg_mask, dtype=torch.bool)
        for lbl in foreground_labels:
            if lbl >= net_output.shape[1]:
                continue
            foreground_inside |= (net_output[:, lbl:lbl + 1] > thresh[lbl])

        unsafe_bg = bg_mask & foreground_inside
        safe_bg = bg_mask & (~foreground_inside)

        base_mask = base_mask & (~safe_bg)
        active_mask = (active_mask & (~safe_bg)) | unsafe_bg
        inactive_mask = base_mask & (~active_mask)
        return base_mask, active_mask, inactive_mask, safe_bg, unsafe_bg

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor, dismiss_label: list=[], permit_label: list=[]):
        target_dice, base_mask, active_mask, inactive_mask, num_fg = self._build_relax_supervision_masks(
            net_output, target, thresh, dismiss_label, permit_label
        )

        base_mask, active_mask, inactive_mask, safe_bg, unsafe_bg = \
            self._apply_bg_outside_foreground_suppression(
                net_output,
                target_dice,
                thresh,
                base_mask,
                active_mask,
                inactive_mask,
            )

        dc_loss = self.dc(
            net_output,
            target_dice,
            loss_mask=active_mask,
            inactive_mask=inactive_mask,
        ) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, base_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        ce_debug = getattr(self.ce, "last_debug", {})
        dice_debug = getattr(self.dc, "last_debug", {})
        loss_ce_debug = ce_loss.detach() if torch.is_tensor(ce_loss) else net_output.new_tensor(float(ce_loss))
        loss_dice_debug = dc_loss.detach() if torch.is_tensor(dc_loss) else net_output.new_tensor(float(dc_loss))
        self.last_debug = {
            "loss_ce": loss_ce_debug,
            "loss_dice": loss_dice_debug,
            "num_safe_bg_outside_fg": safe_bg.sum().detach(),
            "num_unsafe_bg_inside_fg": unsafe_bg.sum().detach(),
            **ce_debug,
            **{f"dice_{k}": v for k, v in dice_debug.items()},
            **dice_debug,
        }
        return result


class DC_and_CE_relax_activemix_loss(DC_and_CE_relax_loss):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aux_kwargs,
        aux_loss_class=AllClassActiveMixTverskyLoss,
        weight_ce=1,
        weight_dice=0,
        lambda_active=0.1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )
        self.lambda_active = float(lambda_active)
        self.aux_loss = aux_loss_class(**aux_kwargs)
        self.aux_loss_name = (
            "active_mix_dice_loss"
            if isinstance(self.aux_loss, AllClassActiveMixDiceLoss)
            else "active_mix_tversky_loss"
        )

    @staticmethod
    def _compute_classwise_active_safe_ratios(target_labels: torch.Tensor, active_mask: torch.Tensor, valid_mask: torch.Tensor, num_classes: int):
        active_ratios = []
        safe_ratios = []
        for cls_idx in range(num_classes):
            cls_valid = valid_mask & (target_labels == cls_idx)
            cls_total = cls_valid.sum().float()
            cls_active = (active_mask & cls_valid).sum().float()
            cls_safe = cls_total - cls_active
            safe_total = torch.clamp(cls_total, min=1.0)
            active_ratios.append((cls_active / safe_total).detach())
            safe_ratios.append((cls_safe / safe_total).detach())
        return torch.stack(active_ratios), torch.stack(safe_ratios)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, thresh: torch.Tensor, dismiss_label: list = [], permit_label: list = []):
        target_dice, base_mask, active_mask, _, num_fg = self._build_relax_supervision_masks(
            net_output, target, thresh, dismiss_label, permit_label
        )

        valid_mask = base_mask
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, valid_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else net_output.sum() * 0.0
        aux_loss = self.aux_loss(net_output, target_dice, active_mask, valid_mask) \
            if self.lambda_active != 0 else net_output.sum() * 0.0

        result = self.weight_ce * ce_loss + self.lambda_active * aux_loss

        valid_labels = valid_mask[:, 0].bool()
        active_labels = active_mask[:, 0].bool()
        target_labels = target_dice[:, 0].long()
        active_ratio_per_class, safe_ratio_per_class = self._compute_classwise_active_safe_ratios(
            target_labels,
            active_labels,
            valid_labels,
            net_output.shape[1],
        )

        ce_debug = getattr(self.ce, "last_debug", {})
        aux_debug = getattr(self.aux_loss, "last_debug", {})
        self.last_debug = {
            "relax_ce_loss": ce_loss.detach(),
            self.aux_loss_name: aux_loss.detach(),
            "lambda_active": self.lambda_active,
            "active_ratio_per_class": active_ratio_per_class,
            "safe_ratio_per_class": safe_ratio_per_class,
            **ce_debug,
            **aux_debug,
        }
        return result


class DC_and_CE_relax_boundary_activemix_dual_thresh_loss(DC_and_CE_relax_activemix_loss):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aux_kwargs,
        aux_loss_class=AllClassActiveMixTverskyLoss,
        weight_ce=1,
        weight_dice=0,
        lambda_active=0.1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        ce_kwargs = dict(ce_kwargs)
        self.boundary_generation_mode = str(ce_kwargs.pop("boundary_generation_mode", "2d"))
        self.lesion_core_ratio_min = float(ce_kwargs.pop("lesion_core_ratio_min", 0.6))
        self.liver_outer_width = int(ce_kwargs.pop("liver_outer_width", 3))
        self.lesion_boundary_target_ratio = ce_kwargs.pop("lesion_boundary_target_ratio", None)
        if self.lesion_boundary_target_ratio is not None:
            self.lesion_boundary_target_ratio = float(self.lesion_boundary_target_ratio)
        self.lesion_boundary_fill_fraction = float(ce_kwargs.pop("lesion_boundary_fill_fraction", 0.5))
        super().__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aux_kwargs=aux_kwargs,
            aux_loss_class=aux_loss_class,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            lambda_active=lambda_active,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )

    def _build_boundary_activemix_masks(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        thresh_core: torch.Tensor,
        thresh_boundary: torch.Tensor,
        dismiss_label: list,
        permit_label: list,
    ):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            base_mask = target != self.ignore_label
            target_dice = torch.where(base_mask, target, 0)
            num_fg = base_mask.sum()
        else:
            target_dice = target
            base_mask = torch.ones_like(target, dtype=torch.bool)
            num_fg = None

        dismiss_mask = torch.ones_like(target_dice, dtype=torch.bool)
        for lbl in dismiss_label:
            dismiss_mask &= (target_dice != lbl)

        permit_mask = torch.zeros_like(target_dice, dtype=torch.bool)
        for lbl in permit_label:
            permit_mask |= (target_dice == lbl)

        target_dice = torch.where(base_mask & dismiss_mask, target_dice, 0)
        boundary_mask = _compute_relax_boundary_prior_mask(
            target_dice,
            lesion_core_ratio_min=self.lesion_core_ratio_min,
            liver_outer_width=self.liver_outer_width,
            boundary_generation_mode=self.boundary_generation_mode,
            ignore_label=self.ignore_label,
            lesion_boundary_target_ratio=self.lesion_boundary_target_ratio,
            lesion_boundary_fill_fraction=self.lesion_boundary_fill_fraction,
        )
        boundary_mask = boundary_mask & base_mask

        target_squeeze = target_dice[:, 0].long()
        true_logit = torch.gather(net_output, dim=1, index=target_squeeze.unsqueeze(1)).squeeze(1)
        true_thresh_core = thresh_core[target_squeeze]
        true_thresh_boundary = thresh_boundary[target_squeeze]
        true_thresh = torch.where(boundary_mask[:, 0], true_thresh_boundary, true_thresh_core)
        relax_mask = (true_logit <= true_thresh).unsqueeze(1)
        relax_mask[permit_mask] = 1

        core_relax_mask = (true_logit <= true_thresh_core).unsqueeze(1)
        core_relax_mask[permit_mask] = 1

        active_mask = base_mask & dismiss_mask & relax_mask
        core_active_mask = base_mask & dismiss_mask & core_relax_mask
        return target_dice, base_mask, boundary_mask, core_active_mask, active_mask, num_fg

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        thresh_core: torch.Tensor,
        thresh_boundary: torch.Tensor,
        dismiss_label: list = [],
        permit_label: list = [],
    ):
        target_dice, base_mask, boundary_mask, core_active_mask, active_mask, num_fg = \
            self._build_boundary_activemix_masks(
                net_output,
                target,
                thresh_core,
                thresh_boundary,
                dismiss_label,
                permit_label,
            )

        valid_mask = base_mask
        ce_loss = self.ce(net_output, target_dice[:, 0], active_mask, valid_mask) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else net_output.sum() * 0.0
        aux_loss = self.aux_loss(net_output, target_dice, active_mask, valid_mask) \
            if self.lambda_active != 0 else net_output.sum() * 0.0

        result = self.weight_ce * ce_loss + self.lambda_active * aux_loss

        valid_labels = valid_mask[:, 0].bool()
        active_labels = active_mask[:, 0].bool()
        target_labels = target_dice[:, 0].long()
        active_ratio_per_class, safe_ratio_per_class = self._compute_classwise_active_safe_ratios(
            target_labels,
            active_labels,
            valid_labels,
            net_output.shape[1],
        )

        ce_debug = getattr(self.ce, "last_debug", {})
        aux_debug = getattr(self.aux_loss, "last_debug", {})
        self.last_debug = {
            "relax_ce_loss": ce_loss.detach(),
            self.aux_loss_name: aux_loss.detach(),
            "lambda_active": self.lambda_active,
            "active_ratio_per_class": active_ratio_per_class,
            "safe_ratio_per_class": safe_ratio_per_class,
            "boundary_pixels": boundary_mask.sum().detach(),
            "num_active_core_threshold": core_active_mask.sum().detach(),
            "num_active_after_boundary_threshold": active_mask.sum().detach(),
            "boundary_generation_mode": self.boundary_generation_mode,
            "lesion_core_ratio_min": self.lesion_core_ratio_min,
            "liver_outer_width": self.liver_outer_width,
            "lesion_boundary_target_ratio": self.lesion_boundary_target_ratio,
            "lesion_boundary_fill_fraction": self.lesion_boundary_fill_fraction,
            **ce_debug,
            **aux_debug,
        }
        return result


class DC_and_CE_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustFocalLoss(torch.tensor(ce_kwargs["alpha"]), gamma=ce_kwargs["gamma"])
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class CurriculumFocalLoss(nn.Module):
    def __init__(self, total_epochs, start_alpha=[1.0, 1.0, 1.0], end_alpha=[1.0, 1.0, 1.0], start_gamma=0.0, end_gamma=0.0, ignore_index=-100):
        """
        total_epochs: 总训练轮数，用于计算课程进度
        max_gamma: 终极难度下的 gamma 值
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.ignore_index = ignore_index
        self.start_alpha = torch.tensor(start_alpha)
        self.end_alpha = torch.tensor(end_alpha)

    def get_dynamic_gamma(self, current_epoch):
        """
        👑 你的核心思想：课程学习退火函数
        前期 gamma=0 (退化为CE，稳扎稳打学简单特征)
        后期 gamma=max_gamma (极致 Focal，死磕难分边界)
        """
        alpha = current_epoch / self.total_epochs  # 线性系数
        gamma = (1 - alpha) * self.start_gamma + alpha * self.end_gamma
        return gamma

        # 专家提示：也可以用 Cosine 退火，曲线更平滑
        # import math
        # return self.max_gamma * 0.5 * (1 - math.cos(math.pi * progress))

    def get_dynamic_alpha(self, current_epoch):
        """
        动态计算 Alpha：
        初始权重：[0, 1, 1]
        随 epoch 线性变化 → 最终权重：[1, 1, 1]
        """
        # 2. 计算线性插值系数 (从 0 → 1 随 epoch 变化)
        alpha = current_epoch / self.total_epochs  # 线性系数

        # 3. 线性插值：start → target
        dynamic_weights = (1 - alpha) * self.start_alpha + alpha * self.end_alpha

        return dynamic_weights

    def forward(self, logits, targets, current_epoch=0):
        # 1. 计算当前的动态超参数
        if targets.ndim == logits.ndim:
            assert targets.shape[1] == 1, "target第二维必须为1"
            targets = targets[:, 0]
        targets = targets.long()
        gamma = self.get_dynamic_gamma(current_epoch)
        alphas = self.get_dynamic_alpha(current_epoch) # [3]
        alphas = alphas.to(targets.device)
        # 2. 计算 CE
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # 3. 提取概率 P_t
        probs = F.softmax(logits, dim=1)
        # 获取真实标签对应的概率 P_t
        p_t = torch.gather(probs, 1, targets.unsqueeze(1).clamp(min=0)).squeeze(1)
        
        # 4. 获取对应的 alpha_t
        alpha_t = alphas[targets.clamp(min=0)]
        
        # 5. 👑 动态 Focal Loss 计算
        # 当 epoch=0 时, gamma=0, (1-p_t)^0 = 1, 退化为普通的 加权 CE
        # 随着 epoch 增大, (1-p_t)^gamma 的威力开始显现，易样本梯度被抹杀！
        if gamma < 0:
            focal_term = (1.0 + p_t) ** (-gamma)  # 负 gamma 反向强化易样本
        else:
            focal_term = (1.0 - p_t) ** gamma
        
        loss = alpha_t * focal_term * ce_loss
        
        # 计算有效像素的平均
        valid_mask = (targets != self.ignore_index)
        return loss[valid_mask].mean()

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result




import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

def compute_sdf1_gt(gt):
    """
    计算 真实标签的 带符号距离场 SDF (φ_G)
    对应论文公式：水平集函数 φ_G(q)
    前景内部 = 负距离
    背景区域 = 正距离
    """
    gt_np = gt.cpu().numpy().astype(np.bool_)
    batch_size, num_classes, *dims = gt_np.shape

    sdf = np.zeros_like(gt_np, dtype=np.float32)

    for b in range(batch_size):
        for c in range(num_classes):
            seg = gt_np[b, c]
            # 内部距离：到边界的距离（负数）
            dist_inside = distance_transform_edt(seg)
            # 外部距离：到边界的距离（正数）
            dist_outside = distance_transform_edt(~seg)
            # 合成 SDF
            sdf[b, c] = dist_outside * (seg == 0) - dist_inside * (seg == 1)

    return torch.from_numpy(sdf).to(gt.device)

class BoundaryLoss(nn.Module):
    """
    Boundary Loss 论文原版实现
    公式：L_B = ∫ φ_G(q) * s_θ(q) dq
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, gt):
        """
        logits: 网络输出 (B, C, H, W) 或 (B, C, D, H, W)
        gt: 真实标签 (B, 1, H, W) 或 one-hot (B, C, H, W)
        """
        num_classes = logits.shape[1]

        # 1. 把标签转成 one-hot
        if gt.shape[1] == 1:
            gt = torch.squeeze(gt, dim=1)
            gt = F.one_hot(gt.long(), num_classes=num_classes)
            gt = gt.permute(0, -1, *list(range(1, len(gt.shape)-1)))

        # 2. 网络输出 → 概率 (0~1)
        pred = torch.sigmoid(logits) if num_classes == 1 else torch.softmax(logits, dim=1)

        # 3. 计算真实标签的水平集函数 φ_G (SDF)
        phi_gt = compute_sdf1_gt(gt)

        # 4. 论文 Boundary Loss 公式：逐像素相乘 + 求和
        loss = (phi_gt * pred).mean()

        return loss


class DC_and_CE_Boundary_loss(nn.Module):
    """DiceCE plus the standard signed-distance boundary loss."""

    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_boundary=0.01,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        super().__init__()
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.weight_boundary = float(weight_boundary)
        self.ignore_label = ignore_label
        self.seg_loss = DC_and_CE_loss(
            soft_dice_kwargs,
            ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=dice_class,
        )
        self.boundary_loss = BoundaryLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        loss_seg = self.seg_loss(net_output, target)
        boundary_target = target
        if self.ignore_label is not None:
            if target.ndim != net_output.ndim or target.shape[1] != 1:
                raise ValueError("Boundary loss expects a single-channel class target")
            boundary_target = target.clone()
            boundary_target[boundary_target == self.ignore_label] = 0
        loss_boundary = self.boundary_loss(net_output, boundary_target)
        return loss_seg + self.weight_boundary * loss_boundary


# ------------------------------
# 🔥 总损失（必须这样用！）
# ------------------------------
class TotalLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.seg_loss = nn.BCEWithLogitsLoss()  # 区域损失（也可换成 DiceLoss）
        self.boundary_loss = BoundaryLoss()
        self.alpha = alpha  # 论文推荐 0.01 → 慢慢增大

    def forward(self, logits, gt):
        l_region = self.seg_loss(logits, gt.float())
        l_boundary = self.boundary_loss(logits, gt)
        total = l_region + self.alpha * l_boundary
        return total, l_region, l_boundary
