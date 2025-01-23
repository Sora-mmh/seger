import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# import sys

# sys.path.insert(
#     0,
#     "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/data/railsseg",
# )


from model import MODEL


class SmpUnetSeg(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super(SmpUnetSeg, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, imgs):
        return self.model(imgs)


@MODEL.register_module
def smpunetseg(pretrained=False, **kwargs):
    model = SmpUnetSeg(**kwargs)
    return model
