"""LGSR: logit-gated Gaussian mixture learning for segmentation."""

from lgsr.losses.lgsr_loss import LGSRSelectiveLoss
from lgsr.models.gmm_head import LGSRHead

__all__ = ["LGSRHead", "LGSRSelectiveLoss"]
