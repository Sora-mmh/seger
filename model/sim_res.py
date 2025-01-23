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


class Resbone(nn.Module):
    def __init__(self, version, out_levels=(5,), pretrained=False):
        super(Resbone, self).__init__()
        model_versions = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "101": (models.resnet101, models.ResNet101_Weights),
            "152": (models.resnet152, models.ResNet152_Weights),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        model = model_fn(weights=weights if pretrained else None)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(model.conv1, model.bn1, model.relu),
                nn.Sequential(model.maxpool, model.layer1),
                model.layer2,
                model.layer3,
                model.layer4,
            ]
        )
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


class SimReSSegHead(nn.Module):
    def __init__(
        self,
        last_decoder_channel=16,
        output_channels=1,
        target_size=(640, 640),
    ):
        super(SimReSSegHead, self).__init__()
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


class SimReSSeg(nn.Module):
    def __init__(self):
        super(SimReSSeg, self).__init__()
        self.encoder = Resbone(
            version="152",
            out_levels=(1, 2, 3, 4, 5),
            pretrained=True,
        )
        self.decoder = UNetDecoder(self.encoder.out_channels)
        self.seg_head = SimReSSegHead(self.decoder.decoder_channels[-1])

    def forward(self, imgs):
        encoded = self.encoder(imgs)
        encoded = [f.detach() for f in encoded]
        decoded = self.decoder(encoded)
        seg_map = self.seg_head(decoded)
        return seg_map


@MODEL.register_module
def simresseg(pretrained=False, **kwargs):
    model = SimReSSeg(**kwargs)
    return model


if __name__ == "__main__":
    model = simresseg().cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    y = model(x)
    print("done")
