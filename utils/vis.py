import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color
import torch.nn.functional as F


def vis_seg_map(
    img_paths, imgs, img_masks, pr_masks, method, root_out, color=(189, 80, 80)
):
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(
            imgs, size=img_masks.shape[-1], mode="bilinear", align_corners=False
        )
    for _, (img_path, img, img_mask, pr_mask) in enumerate(
        zip(img_paths, imgs, img_masks, pr_masks)
    ):
        img_mask = img_mask[0] if len(img_mask.shape) == 3 else img_mask
        pr_mask = pr_mask[0] if len(pr_mask.shape) == 3 else pr_mask
        parts = img_path.split("/")
        needed_part = parts[-1]
        # specific_root = "/".join(needed_parts)
        img_num = needed_part.split(".")[0]
        out_dir = f"{root_out}/{method}/{img_num}"
        os.makedirs(out_dir, exist_ok=True)
        img_path = f"{out_dir}/{img_num}_img.png"
        pr_mask_path = f"{out_dir}/{img_num}_pr.png"
        gt_mask_path = f"{out_dir}/{img_num}_gt.png"
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        img_rec = img * std[:, None, None] + mean[:, None, None]
        # RGB image
        img_rec = Image.fromarray(
            (img_rec * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        )
        img_rec.save(img_path)
        img_rec_pr_mask = img_rec.copy()
        # RGB image with segmentation map
        pr_mask = (pr_mask - pr_mask.min()) / (pr_mask.max() - pr_mask.min())
        # pr_mask[pr_mask > 0] = 1
        # pr_mask = cm.jet(pr_mask)
        # pr_mask = cm.rainbow(pr_mask)
        # pr_mask = (pr_mask[:, :, :3] * 255).astype("uint8")
        pr_mask = Image.fromarray((pr_mask * 255).astype(np.uint8), mode="L")
        mask = Image.new("RGBA", pr_mask.size, color + (0,))
        mask.putalpha(pr_mask)
        # img_rec_pr_mask = Image.blend(img_rec, mask.convert("RGB"), alpha=0.4)
        img_rec_pr_mask.paste(mask, (0, 0), mask)
        img_rec_pr_mask.save(pr_mask_path)
        # mask
        img_mask = Image.fromarray((img_mask * 255).astype(np.uint8))
        img_mask.save(gt_mask_path)
