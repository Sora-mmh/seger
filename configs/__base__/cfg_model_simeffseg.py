from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class cfg_model_simeffseg(Namespace):

    def __init__(self):
        Namespace.__init__(self)
        self.model = Namespace()
        self.model.name = "simeffseg"
        self.model.kwargs = dict(pretrained=False, checkpoint_path="", strict=True)
