import monai.metrics
import torch


def compute_accuracy(pr, gt, thres=0.5):
    pr = pr > thres
    gt = gt == torch.max(gt)
    corr = torch.sum(pr == gt)
    tensor_size = pr.size(0) * pr.size(1) * pr.size(2) * pr.size(3)
    acc = float(corr) / float(tensor_size)
    return acc


def compute_sensitivity(pr, gt, thres=0.5):
    pr = pr > thres
    gt = gt == torch.max(gt)
    tp = ((pr == 1) + (gt == 1)) == 2
    fn = ((pr == 0) + (gt == 1)) == 2
    se = float(torch.sum(tp)) / (float(torch.sum(tp + fn)) + 1e-6)
    return se


def compute_specificity(pr, gt, thres=0.5):
    pr = pr > thres
    gt = gt == torch.max(gt)
    tn = ((pr == 0) + (gt == 0)) == 2
    fp = ((pr == 1) + (gt == 0)) == 2
    sp = float(torch.sum(tn)) / (float(torch.sum(tn + fp)) + 1e-6)
    return sp


def compute_precision(pr, gt, thres=0.5):
    pr = pr > thres
    gt = gt == torch.max(gt)
    tp = ((pr == 1) + (gt == 1)) == 2
    fp = ((pr == 1) + (gt == 0)) == 2
    pc = float(torch.sum(tp)) / (float(torch.sum(tp + fp)) + 1e-6)
    return pc


def compute_f1(pr, gt, thres=0.5):
    dc = compute_sensitivity(pr, gt, thres=thres)
    pc = compute_precision(pr, gt, thres=thres)
    f1 = 2 * dc * pc / (dc + pc + 1e-6)
    return f1


######### TODO: Check !!! ############
def compute_jaccard(pr, gt, thres=0.5):
    pr = pr > thres
    gt = gt == torch.max(gt)
    inter = torch.sum((pr + gt) == 2)
    union = torch.sum((pr + gt) >= 1)
    js = float(inter) / (float(union) + 1e-6)
    return js


def compute_iou(pr, gt, threshold=0.5):
    pr = (pr > threshold).float()
    intersection = torch.sum((pr + gt) == 2)
    union = torch.sum((pr + gt) >= 1)
    iou = float(intersection) / (float(union) + 1e-6)
    return iou


######### TODO: Check !!! ############
def compute_dice(pr, gt, thres=0.5):
    pr = pr > thres
    gt = gt == torch.max(gt)
    inter = torch.sum((pr + gt) == 2)
    dc = float(2 * inter) / (float(torch.sum(pr) + torch.sum(gt)) + 1e-6)
    return dc


if __name__ == "__main__":
    pr = torch.ones(2, 2, 256, 256)  # torch.randn(2, 2, 256, 256)
    gt = torch.ones(2, 2, 256, 256)  # torch.randn(2, 2, 256, 256)
    import monai

    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean")
    print("iou ", compute_iou(pr, gt), monai.metrics.compute_iou(pr, gt))
    print("dice ", compute_dice(pr, gt), dice_metric(pr, gt))
    print("jaccard", compute_jaccard(pr, gt))
    print("done")
