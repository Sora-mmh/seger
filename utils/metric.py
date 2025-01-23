import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    auc,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from monai.metrics import compute_iou, compute_dice
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from skimage import measure
import multiprocessing
import copy

import numpy as np
from numba import jit
import torch
from torch.nn import functional as F

from utils.util import get_timepc, log_msg
from utils.registry import Registry

from adeval import EvalAccumulatorCuda

EVALUATOR = Registry("Evaluator")


# def func(th, amaps, binary_amaps, masks):
#     print("start", th)
#     binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
#     pro = []
#     for binary_amap, mask in zip(binary_amaps, masks):
#         for region in measure.regionprops(measure.label(mask)):
#             tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
#             pro.append(tp_pixels / region.area)
#     inverse_masks = 1 - masks
#     fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
#     fpr = fp_pixels / inverse_masks.sum()
#     print("end", th)
#     return [np.array(pro).mean(), fpr, th]


def compute_iou(gt, pr):
    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
    iou = total_area_intersect / total_area_union
    return iou


def compute_dice(gt, pr):
    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
    total_area_pred_label = pr.sum(axis=(0, 1, 2))
    total_area_label = gt.sum(axis=(0, 1, 2))
    dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
    return dice


def compute_metrics(gt, pr):
    iou = compute_iou(gt, pr) * 100
    dice = compute_dice(gt, pr) * 100
    all_ones_pred = np.ones_like(pr)
    all_ones_dice = compute_dice(gt, all_ones_pred) * 100
    ones_dice_diff = dice - all_ones_dice
    all_zeros_dice = np.zeros_like(pr)
    all_zeros_dice = compute_dice(gt, all_zeros_dice) * 100
    zeros_dice_diff = dice - all_zeros_dice
    return {
        "iou": iou,
        "dice": dice,
        "ones_dice_diff": ones_dice_diff,
        "zeros_dice_diff": zeros_dice_diff,
    }


class Evaluatorv0(object):
    def __init__(self, metrics=[]):
        if len(metrics) == 0:
            self.metrics = ["IoU", "Dice", "IoU_max", "Dice_max"]
        else:
            self.metrics = metrics
        # self.pooling_ks = pooling_ks
        # self.max_step_aupro = max_step_aupro
        # self.mp = mp
        self.eps = 1e-8
        self.beta = 1.0
        self.boundary = 1e-7

    def run(self, results, cls_name, logger=None):
        idxes = results["cls_names"] == cls_name
        gt_masks = results["imgs_masks"][idxes].squeeze(1)
        pr_masks = results["pr_masks"][idxes]
        if len(gt_masks.shape) == 4:
            gt_masks = gt_masks.squeeze(1)
        if len(pr_masks.shape) == 4:
            pr_masks = pr_masks.squeeze(1)
        pr_masks = (pr_masks - pr_masks.min()) / (pr_masks.max() - pr_masks.min())
        metric_str = f"==> Metric Time for {cls_name:<15}: "
        metric_results = {}
        for metric in self.metrics:
            t0 = get_timepc()
            if metric.startswith("IoU_rng") or metric.startswith("Dice_rng"):
                coms = metric.split("_")
                assert (
                    len(coms) == 5
                ), f"{metric} should contain parameters 'score_l', 'score_h', and 'score_step' "
                score_l, score_h, score_step = (
                    float(metric.split("_")[-3]),
                    float(metric.split("_")[-2]),
                    float(metric.split("_")[-1]),
                )
                gt = gt_masks.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_masks > score
                    if metric.startswith("IoU"):
                        iou_score = compute_iou(gt, pr)
                        metric_scores.append(iou_score)
                    elif metric.startswith("Dice"):
                        dice_score = compute_dice(gt, pr)
                        metric_scores.append(dice_score)
                    else:
                        raise f"invalid metric: {metric}"
                metric_results[metric] = np.array(metric_scores).mean()

            elif metric.startswith("IoU_max") or metric.startswith("Dice_max"):
                score_l, score_h, score_step = 0.0, 1.0, 0.05
                scores = []
                gt = gt_masks.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_masks > score
                    scores.append(score)
                    if metric.startswith("IoU_max"):
                        iou_score = compute_iou(gt, pr)
                        metric_scores.append(iou_score)
                    elif metric.startswith("Dice_max"):
                        dice_score = compute_dice(gt, pr)
                        metric_scores.append(dice_score)
                    else:
                        raise f"invalid metric: {metric}"
                metric_results[metric] = np.array(metric_scores).max()
                best_score_idx = np.array(metric_scores).argmax()
                if metric == "IoU_max":
                    metric_results["Opt_Thres_IoU_max"] = scores[best_score_idx]
                elif metric == "Dice_max":
                    metric_results["Opt_Thres_Dice_max"] = scores[best_score_idx]

            t1 = get_timepc()
            metric_str += f"{t1 - t0:7.3f} ({metric})\t"
        log_msg(logger, metric_str)
        return metric_results


class Evaluator(object):
    def __init__(self, metrics=[]):
        if len(metrics) == 0:
            self.metrics = [
                "IoU",
                "Dice",
                "F1",
                "Acc",
                "IoU_max",
                "Dice_max",
                "F1_max",
                "Acc_max",
            ]
            # self.metrics = ["mIoU", "Dice"]
        else:
            self.metrics = metrics

        # self.pooling_ks = pooling_ks
        # self.max_step_aupro = max_step_aupro
        # self.mp = mp

        self.eps = 1e-8
        self.beta = 1.0

        self.boundary = 1e-7

    def run(self, results, cls_name, logger=None):
        idxes = results["cls_names"] == cls_name
        gt_masks = results["imgs_masks"][idxes].squeeze(1)
        pr_masks = results["pr_masks"][idxes]
        if len(gt_masks.shape) == 4:
            gt_masks = gt_masks.squeeze(1)
        if len(pr_masks.shape) == 4:
            pr_masks = pr_masks.squeeze(1)
        # cls_names = results["cls_names"][idxes]
        metric_str = f"==> Metric Time for {cls_name:<15}: "
        metric_results = {}
        for metric in self.metrics:
            t0 = get_timepc()
            ################
            ################
            ################
            ################
            if (
                metric.startswith("IoU_rng")
                or metric.startswith("Dice_rng")
                or metric.startswith("F1_rng")
                or metric.startswith("Acc_rng")
            ):  # example: F1_sp_0.3_0.8
                # F1_px equals Dice_px
                coms = metric.split("_")
                assert (
                    len(coms) == 5
                ), f"{metric} should contain parameters 'score_l', 'score_h', and 'score_step' "
                score_l, score_h, score_step = (
                    float(metric.split("_")[-3]),
                    float(metric.split("_")[-2]),
                    float(metric.split("_")[-1]),
                )
                gt = gt_masks.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_masks > score
                    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
                    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
                    total_area_pred_label = pr.sum(axis=(0, 1, 2))
                    total_area_label = gt.sum(axis=(0, 1, 2))
                    if metric.startswith("IoU"):
                        iou_score = total_area_intersect / (total_area_union + self.eps)
                        metric_scores.append(iou_score)
                    elif metric.startswith("Dice"):
                        dice_score = (
                            2
                            * total_area_intersect
                            / (total_area_pred_label + total_area_label + self.eps)
                        )
                        metric_scores.append(dice_score)
                    elif metric.startswith("F1"):
                        precision = total_area_intersect / (
                            total_area_pred_label + self.eps
                        )
                        recall = total_area_intersect / (total_area_label + self.eps)
                        f1_score = (
                            (1 + self.beta**2)
                            * precision
                            * recall
                            / (self.beta**2 * precision + recall + self.eps)
                        )
                        metric_scores.append(f1_score)
                    elif metric.startswith("Acc"):
                        acc_score = total_area_intersect / (total_area_label + self.eps)
                        metric_scores.append(acc_score)
                    else:
                        raise f"invalid metric: {metric}"
                metric_results[metric] = np.array(metric_scores).mean()

            elif (
                metric.startswith("IoU_max")
                or metric.startswith("Dice_max")
                or metric.startswith("F1_max")
                or metric.startswith("Acc_max")
            ):

                score_l, score_h, score_step = 0.0, 1.0, 0.05
                gt = gt_masks.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_masks > score
                    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
                    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
                    total_area_pred_label = pr.sum(axis=(0, 1, 2))
                    total_area_label = gt.sum(axis=(0, 1, 2))
                    if metric.startswith("IoU_max"):
                        iou_score = total_area_intersect / (total_area_union + self.eps)
                        metric_scores.append(iou_score)
                    elif metric.startswith("Dice_max"):
                        dice_score = (
                            2
                            * total_area_intersect
                            / (total_area_pred_label + total_area_label + self.eps)
                        )
                        metric_scores.append(dice_score)
                    elif metric.startswith("F1_max"):
                        precision = total_area_intersect / (
                            total_area_pred_label + self.eps
                        )
                        recall = total_area_intersect / (total_area_label + self.eps)
                        f1_score = (
                            (1 + self.beta**2)
                            * precision
                            * recall
                            / (self.beta**2 * precision + recall + self.eps)
                        )
                        metric_scores.append(f1_score)
                    elif metric.startswith("Acc_max"):
                        acc_score = total_area_intersect / (total_area_label + self.eps)
                        metric_scores.append(acc_score)
                    else:
                        raise f"invalid metric: {metric}"
                metric_results[metric] = np.array(metric_scores).max()
            ################
            ################
            ################
            ################
            # metric_scores = []
            # miou = MeanIoU(num_classes=1)
            # gds = GeneralizedDiceScore(num_classes=1)
            # if metric.startswith("mIoU"):
            #     metric_scores.append(miou(pr_masks, gt_masks))
            # elif metric.startswith("Dice"):
            #     metric_scores.append(gds(pr_masks, gt_masks))
            # metric_results[metric] = np.array(metric_scores).mean()
            t1 = get_timepc()
            metric_str += f"{t1 - t0:7.3f} ({metric})\t"
        log_msg(logger, metric_str)
        return metric_results


def get_evaluator(cfg_evaluator):
    evaluator, kwargs = Evaluator, cfg_evaluator.kwargs
    return evaluator(**kwargs)


if __name__ == "__main__":
    pr = np.random.rand(512, 512)
    gt = np.random.rand(512, 512)
    pr[pr > 0.5] = 1
    pr[pr < 0.5] = 0
    gt[gt > 0.5] = 1
    gt[gt < 0.5] = 0
    metrics = compute_metrics(gt, pr)
    print("done.")
