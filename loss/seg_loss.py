import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as F_tv
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import segmentation_models_pytorch as smp

# import sys

# sys.path.insert(
#     0,
#     "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/data/railsseg",
# )
from loss import LOSS
from model import get_model

__all__ = ["CE", "DiceLoss"]


@LOSS.register_module
class CE(nn.CrossEntropyLoss):
    def __init__(self, lam=1):
        super(CE, self).__init__()
        self.lam = lam

    def forward(self, pr, gt):
        return super(CE, self).forward(pr, gt) * self.lam


@LOSS.register_module
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        # self.loss = smp.losses.DiceLoss(mode="binary")
        # prediction = torch.sigmoid(prediction)
        # prediction = prediction.flatten(start_dim=1)  # (B, H * W)
        # target = target.flatten(start_dim=1)
        # zero_target_mask = target.sum(dim=1) == 0  # (B,)
        # if zero_target_mask.any():
        #     prediction[zero_target_mask] = 1 - prediction[zero_target_mask]
        #     target[zero_target_mask] = 1 - target[zero_target_mask]
        # intersection = (prediction * target).sum(dim=1)  # (B,)
        # cardinality = (prediction + target).sum(dim=1)
        # scores = 2 * intersection / cardinality
        # return 1 - scores.mean()  # ()

    # def forward(self, pr, gt, smooth=1):
    #     pr = F.sigmoid(inputs)
    #     pr = pr.view(-1)
    #     gt = targets.view(-1)
    #     inter = (pr * gt).sum()
    #     dice = (2. * inter + smooth) / (pr.sum() + gt.sum() + smooth)
    #     return 1 - dice

    def forward(self, pr, gt):
        pr = torch.sigmoid(pr)
        pr = pr.flatten(start_dim=1)
        gt = gt.flatten(start_dim=1)
        # bg = gt.sum(dim=1) == 0
        # if bg.any():
        #     pr[bg] = 1 - pr[bg]
        #     gt[bg] = 1 - gt[bg]
        inter = (pr * gt).sum(dim=1)
        scores = (2 * inter) / (pr + gt).sum(dim=1)
        return 1 - scores.mean()
        # return self.loss(pr, gt)


if __name__ == "__main__":
    pr = torch.randn(2, 2, 256, 256)
    gt = torch.randn(2, 2, 256, 256)
    ce = CE()
    dl = DiceLoss()
    res1 = ce(pr, gt)
    res2 = dl(pr, gt)
    print("ce loss", res1)
    print("dice loss smp", res2[0], "dice loss custom", res2[1])
    print("done")
