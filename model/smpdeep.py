import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# import sys

# sys.path.insert(
#     0,
#     "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/data/railsseg",
# )


from model import MODEL


class SmpDeepSeg(nn.Module):
    def __init__(self, encoder_name, encoder_weights, activation, classes):
        super(SmpDeepSeg, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            activation=activation,
            classes=classes,
        )

    def forward(self, imgs):
        return self.model(imgs)


@MODEL.register_module
def smpdeepseg(pretrained=False, **kwargs):
    model = SmpDeepSeg(**kwargs)
    return model


# if __name__ == "__main__":
#     model = smp.DeepLabV3Plus(
#         encoder_name="resnet101",
#         encoder_weights="imagenet",
#         activation="sigmoid",
#         in_channels=3,
#         classes=1,
#     )
#     print(model)
#     x = torch.rand(16, 3, 640, 640)
#     y = model(x)
#     print("done")
