from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class cfg_model_vitseg(Namespace):

    def __init__(self):
        Namespace.__init__(self)
        self.encoder = Namespace()
        self.encoder.name = "vit_small_patch16_224_dino"
        self.encoder.kwargs = dict(
            pretrained=True,
            checkpoint_path="",
            strict=True,
            img_size=256,
            teachers=[3, 6, 9],
            neck=[12],
        )
        self.model = Namespace()
        self.model.name = "vitseg"
        self.model.kwargs = dict(
            pretrained=False, checkpoint_path="", strict=True, encoder=self.encoder
        )
