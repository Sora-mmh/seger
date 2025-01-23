from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class cfg_model_resseg(Namespace):

    def __init__(self):
        Namespace.__init__(self)

        size = 256
        in_chas = [256, 512, 1024]
        out_indices = [i + 1 for i in range(len(in_chas))]
        out_cha = 256
        style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
        in_strides = [
            2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))
        ]  # [4, 2, 1]
        latent_channel_size = 64
        self.model_encoder = Namespace()
        self.model_encoder.name = "timm_wide_resnet50_2"
        self.model_encoder.kwargs = dict(
            pretrained=False,
            checkpoint_path="model/pretrain/wide_resnet50_racm-8234f177.pth",
            strict=False,
            features_only=True,
            out_indices=out_indices,
        )
        self.model = Namespace()
        self.model.name = "resseg"
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path="",
            strict=True,
            model_encoder=self.model_encoder,
        )
