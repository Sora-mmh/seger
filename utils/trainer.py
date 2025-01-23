import os
import albumentations as album
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class BatchVisualizer:

    def __init__(
        self,
        cfg,
    ):
        # if not os.path.exists(cfg.logdir):
        #     os.makedirs(cfg.logdir)
        self.writer = SummaryWriter(log_dir=cfg.logdir)

    def visualize_image_batch(self, image_batch, n_iter, image_name="Image_batch"):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name, grid, n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name="loss"):
        self.writer.add_scalar(loss_name, loss_val, n_iter)


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=720, width=720, always_apply=True),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(
            min_height=1536, min_width=1536, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(test_transform)


def get_test_augmentation():
    test_transform = [
        album.PadIfNeeded(
            min_height=1920, min_width=1920, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "predicted_mask_railtrack" or name == "predicted_mask_railraised":
            x_start = 0
            x_end = 1920
            y_start = 420
            y_end = 1500
            image = image[y_start:y_end, x_start:x_end]
        plt.imshow(image, interpolation="nearest")
    plt.imsave(
        "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/data/railsseg/predictions.png",
        image,
    )


def train_one_epoch(net, loader, loss_fn, optimizer, device):
    net.train()
    epoch_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.long()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validate_one_epoch(net, loader, loss_fn, device):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.long()
            outputs = net(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(loader)


def compute_iou(preds, masks):
    preds = (preds > 0.5).float().cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()
    return jaccard_score(masks, preds)
