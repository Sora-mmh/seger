from typing import Callable
import json
import logging
from pathlib import Path
from tqdm import tqdm

import cv2
from PIL import Image
import imgaug.augmenters as iaa
from skimage import morphology
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

from torch.utils.data import dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
from utils.data import get_img_loader, crop_roi
from data.utils import get_transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torch.utils.data as data


import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# from . import DATA
from data import DATA


def extract_cls_from_json(config_json: dict):
    return {cls["name"]: idx for idx, cls in enumerate(config_json["labels"])}


# ---------- RS19 ----------
RS19_str_to_int = {
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

RS19_int_to_str = {
    0: "road",
    1: "sidewalk",
    2: "construction",
    3: "tram track",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "human",
    12: "railroad",
    13: "car",
    14: "truck",
    15: "trackbed",
    16: "on rails",
    17: "rail raised",
    18: "rail embedded",
}

CLS_IDX = 12


@DATA.register_module
class RS19Dataset(data.Dataset):
    def __init__(self, cfg, train=True, transform=None, target_transform=None):
        self.root = cfg.data.root
        self.folder_name = cfg.data.folder_name
        self.cls_name = cfg.data.cls_names[0]
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # self.cls_names = ["railroad"]
        self.data_all = []
        self.loader = get_img_loader(cfg.data.loader_type)
        self.loader_target = get_img_loader(cfg.data.loader_type_target)
        self.dataset_path = Path(self.root) / self.cls_name
        # self.imgs_paths = sorted(
        #     list((self.dataset_path / "jpgs" / self.folder_name).iterdir())
        # )[:100]
        self.imgs_paths = sorted(list((self.dataset_path / "images").iterdir()))
        # self.dataset_config = json.load(open(self.dataset_path / cfg.data.config, "r"))
        # self.cls = extract_cls_from_json(self.dataset_config)
        #### extract only target cls masks ####
        # target_mask = mask == CLS_IDX
        # [(mask == idx) for idx in self.cls.values()]
        # target_mask = (
        #     np.stack(target_mask, axis=-1)
        #     if len(target_mask.shape) == 3
        #     else target_mask
        # )
        if self.train:
            self.imgs_paths = self.imgs_paths[: cfg.data.train_samples]
            # train_split = []
            # with open((cfg.data.splits / "train.txt").as_posix(), "r") as f:
            #     for line in f:
            #         train_split.append(line.strip())
            for img_path in tqdm(
                self.imgs_paths,
                desc="Calling Data",
                total=len(self.imgs_paths),
            ):
                # if img_path.stem in train_split:
                dt = {}
                dt["img_path"] = img_path.as_posix()
                img_num = img_path.stem
                needed_parts = img_path.as_posix().split("/")
                mask_path = Path(
                    "/".join(needed_parts[:-2] + ["ground_truth"] + [img_num + ".png"])
                )
                dt["mask_path"] = mask_path.as_posix()
                dt["cls_name"] = RS19_int_to_str[CLS_IDX]
                self.data_all.append(dt)
            # self.data_all = self.data_all[: cfg.data.train_samples]
        else:
            # test_split = []
            # with open((cfg.data.splits / "test.txt").as_posix(), "r") as f:
            #     for line in f:
            #         test_split.append(line.strip())
            self.imgs_paths = self.imgs_paths[-cfg.data.test_samples :]
            for img_path in tqdm(
                self.imgs_paths,
                desc="Calling Data",
                total=len(self.imgs_paths),
            ):
                # if img_path.stem in test_split:
                dt = {}
                dt["img_path"] = img_path.as_posix()
                img_num = img_path.stem
                needed_parts = img_path.as_posix().split("/")
                mask_path = Path(
                    "/".join(needed_parts[:-2] + ["ground_truth"] + [img_num + ".png"])
                )
                dt["mask_path"] = mask_path.as_posix()
                dt["cls_name"] = RS19_int_to_str[CLS_IDX]
                self.data_all.append(dt)
            # self.data_all = self.data_all[: cfg.data.test_samples]
        self.length = len(self.data_all)

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name = (
            data["img_path"],
            data["mask_path"],
            data["cls_name"],
        )
        img = self.loader(img_path)
        img = self.transform(img) if self.transform is not None else img
        img_mask = np.array(self.loader_target(mask_path)) > 0
        img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode="L")
        img_mask = (
            self.target_transform(img_mask)
            if self.target_transform is not None and img_mask is not None
            else img_mask
        )
        return {
            "img": img,
            "img_mask": img_mask,
            "img_path": img_path,
            "cls_name": cls_name,
        }

    def __len__(self):
        return self.length


# class RS19Dataset(Dataset):
#     def __init__(self):
#         self.dir_pth = Path(
#             "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/data/wilddash2"
#         )
#         self.name = "rs19_val"
#         self.dataset_pth = self.dir_pth / self.name
#         self.imgs_pths = sorted(list((self.dataset_pth / "jpgs" / self.name).iterdir()))
#         self.masks_pths = sorted(
#             list((self.dataset_pth / "uint8" / self.name).iterdir())
#         )
#         self.dataset_config = json.load(
#             open(self.dir_pth / self.name / "rs19-config.json", "r")
#         )
#         self.cls = extract_cls_from_json(self.dataset_config)
#         self.load()
#         self.img_transform, self.mask_transform = get_data_transforms(
#             target_shape=(640, 640)
#         )

#     def load(self):
#         self.data_gt = []
#         for img_pth, mask_pth in zip(self.imgs_pths, self.masks_pths):
#             mask = cv2.imread(mask_pth.as_posix(), 0)
#             target_mask = mask == 18
#             # [(mask == idx) for idx in self.cls.values()]
#             target_mask = (
#                 np.stack(target_mask, axis=-1)
#                 if len(target_mask.shape) == 3
#                 else target_mask
#             )
#             if any(np.unique(target_mask)):
#                 self.data_gt.append((img_pth, target_mask))

#     def __getitem__(self, idx: int):
#         img_pth, mask = self.data_gt[idx]
#         img = cv2.imread(img_pth.as_posix())
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return self.img_transform(img), self.mask_transform(mask)

#     def __len__(self):
#         return len(self.data_gt)


#  class ToTensor(object):
#     def __call__(self, image):
#         try:
#             image = torch.from_numpy(image.transpose(2, 0, 1))
#         except:
#             logging.info(
#                 "Invalid_transpose, please make sure images have shape (H, W, C) before transposing"
#             )
#         if not isinstance(image, torch.FloatTensor):
#             image = image.float()
#         return image


# class Normalize(object):
#     def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.600, 0.225]):
#         self.mean = np.array(mean)
#         self.std = np.array(std)

#     def __call__(self, image):
#         image = (image - self.mean) / self.std
#         return image


# def get_data_transforms(target_shape=(640, 640)):
#     mean, std = [0.485, 0.456, 0.406], [0.229, 0.600, 0.225]
#     img_transform = T.Compose(
#         [
#             T.ToTensor(),
#             T.Resize(target_shape),
#             T.Normalize(mean=mean, std=std),
#         ]
#     )
#     mask_transform = T.Compose([T.ToTensor(), T.Resize(target_shape)])
#     return img_transform, mask_transform

if __name__ == "__main__":
    data = RS19Dataset()
    train_size = int(0.7 * len(data))
    val_size = int(0.2 * len(data))
    train_data = Subset(data, range(train_size))
    val_data = Subset(data, range(train_size, train_size + val_size))
    test_data = Subset(data, range(train_size + val_size, len(data)))
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    for imgs, masks in dataloader:
        print("debugging")
