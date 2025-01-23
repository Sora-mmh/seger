from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class cfg_model_smpunetseg(Namespace):

    def __init__(self):
        Namespace.__init__(self)
        self.model = Namespace()
        self.model.kwargs = dict(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            pretrained=False,
            checkpoint_path="",
            strict=True,
        )
