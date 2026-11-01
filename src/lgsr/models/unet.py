from __future__ import annotations

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, UNetDecoder

from lgsr.models.gmm_head import LGSRHead


class LGSRDecoder(UNetDecoder):
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, components_per_class=(1, 3, 3)):
        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first=False)
        if deep_supervision:
            raise ValueError("LGSR uses the final decoder stage only")
        self.classify = LGSRHead(
            feat_dim=int(encoder.output_channels[0]),
            num_classes=int(num_classes),
            components_per_class=components_per_class,
        )

    def forward(self, skips, mode="test"):
        low_resolution = skips[-1]
        for stage_idx in range(len(self.stages)):
            upsampled = self.transpconvs[stage_idx](low_resolution)
            merged = torch.cat((upsampled, skips[-(stage_idx + 2)]), dim=1)
            features = self.stages[stage_idx](merged)
            if stage_idx == len(self.stages) - 1:
                payload = self.classify(features.float(), return_components=(mode == "train"))
                if mode == "train":
                    payload["features"] = features.float()
                return payload
            low_resolution = features
        raise RuntimeError("LGSR decoder has no stages")


class LGSRUNet(PlainConvUNet):
    """nnU-Net PlainConvUNet backbone with the paper's 32-D GMM decoder head."""

    def __init__(self, input_channels, num_classes, gmm_components_per_class=(1, 3, 3), **architecture_kwargs):
        deep_supervision = bool(architecture_kwargs.get("deep_supervision", False))
        super().__init__(input_channels=input_channels, num_classes=num_classes, **architecture_kwargs)
        self.decoder = LGSRDecoder(
            self.encoder,
            num_classes,
            architecture_kwargs["n_conv_per_stage_decoder"],
            deep_supervision,
            components_per_class=tuple(gmm_components_per_class),
        )

    def forward(self, x, mode="test"):
        skips = self.encoder(x)
        return self.decoder(skips, mode=mode)

    def get_back_feats(self):
        raise RuntimeError("Decoder features are returned in the training payload")
