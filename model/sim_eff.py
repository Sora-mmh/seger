import torch
import torch.nn as nn
import torchvision.models as models

import sys

sys.path.insert(
    0,
    "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/data/railsseg",
)
from model.decoder import UNetDecoder
from model import MODEL


class Effbone(nn.Module):
    def __init__(self, version, out_levels=(8,), pretrained=True):
        super(Effbone, self).__init__()
        model_versions = {
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            "b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            "b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        # last block is discarded because it would be redundant with the pooling layer
        model = model_fn(weights=weights if pretrained else None).features[:-1]
        self.stages = nn.ModuleList([model[i] for i in range(len(model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features


class SimEffSegHead(nn.Module):
    def __init__(
        self,
        last_decoder_channel=16,
        output_channels=1,
        target_size=(640, 640),
    ):
        super(SimEffSegHead, self).__init__()
        self.target_size = target_size
        self.seg_head = nn.Conv2d(
            in_channels=last_decoder_channel,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        seg_map = self.seg_head(x)
        seg_map = nn.functional.interpolate(
            seg_map, size=self.target_size, mode="bilinear", align_corners=False
        )
        return seg_map


class SimEffSeg(nn.Module):
    def __init__(self):
        super(SimEffSeg, self).__init__()
        self.encoder = Effbone(
            version="b5",
            out_levels=(1, 3, 4, 6, 8),
            pretrained=True,
        )
        self.decoder = UNetDecoder(self.encoder.out_channels)
        self.seg_head = SimEffSegHead(self.decoder.decoder_channels[-1])

    def forward(self, imgs):
        encoded = self.encoder(imgs)
        encoded = [f.detach() for f in encoded]
        decoded = self.decoder(encoded)
        seg_map = self.seg_head(decoded)
        return seg_map


@MODEL.register_module
def simeffseg(pretrained=False, **kwargs):
    model = SimEffSeg(**kwargs)
    return model


if __name__ == "__main__":
    model = simeffseg().cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    y = model(x)
    print("done")
