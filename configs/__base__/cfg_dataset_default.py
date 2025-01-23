from argparse import Namespace
from pathlib import Path
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class cfg_dataset_default(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        self.data = Namespace()
        self.data.sampler = "naive"
        self.data.loader_type = "pil"
        self.data.loader_type_target = "pil_L"
        # ---------- RS19 ----------
        RS19 = {
            "road": 0,
            "sidewalk": 1,
            "construction": 2,
            "tram track": 3,
            "fence": 4,
            "pole": 5,
            "traffic light": 6,
            "traffic sign": 7,
            "vegetation": 8,
            "terrain": 9,
            "sky": 10,
            "human": 11,
            "railroad": 12,
            "car": 13,
            "truck": 14,
            "trackbed": 15,
            "on rails": 16,
            "rail raised": 17,
            "rail embedded": 18,
        }
        self.data.type = "RS19Dataset"
        self.data.folder_name = "rs19_val"
        self.data.root = "datasets"
        self.data.config = "rs19-config.json"
        self.data.splits = Path(
            "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/data/wilddash2/rs19_splits4000"
        )
        self.data.cls_names = ["railroad"]

        self.data.train_transforms = [
            dict(
                type="Resize",
                size=(640, 640),
                interpolation=F.InterpolationMode.BILINEAR,
            ),
            dict(type="ToTensor"),
            dict(
                type="Normalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                inplace=True,
            ),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(
                type="Resize",
                size=(640, 640),
                interpolation=F.InterpolationMode.BILINEAR,
            ),
            # dict(type="CenterCrop", size=(256, 256)),
            dict(type="ToTensor"),
        ]
        # self.data.train_transforms = [
        # 	dict(type='RandomResizedCrop', size=(self.size, self.size), scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR),
        # 	dict(type='CenterCrop', size=(self.size, self.size)),
        # 	dict(type='RandomHorizontalFlip', p=0.5),
        # 	dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0),
        # 	dict(type='RandomRotation', degrees=(-17, 17), interpolation=F.InterpolationMode.BILINEAR, expand=False),
        # 	dict(type='CenterCrop', size=(self.size, self.size)),
        # 	dict(type='ToTensor'),
        # 	dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        # ]
        # self.data.test_transforms = self.data.train_transforms
        # self.data.target_transforms = [
        # 	dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
        # 	dict(type='CenterCrop', size=(self.size, self.size)),
        # 	dict(type='ToTensor'),
        # ]
