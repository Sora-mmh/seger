from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import cfg_common, cfg_dataset_default, cfg_model_simeffseg


class cfg(cfg_common, cfg_dataset_default, cfg_model_simeffseg):

    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_simeffseg.__init__(self)
        self.seed = 42
        self.size = 640
        self.epoch_full = 500
        self.warmup_epochs = 0
        self.test_start_epoch = self.epoch_full
        self.test_per_epoch = self.epoch_full // self.epoch_full
        self.batch_train = 8
        self.batch_test_per = 8
        self.lr = 1e-3  # * self.batch_train / 16
        self.weight_decay = 0.0001
        self.metrics = [
            "IoU_rng_0.3_0.7_0.1",
            "Dice_rng_0.3_0.7_0.1",
            "F1_rng_0.3_0.7_0.1",
            "Acc_rng_0.3_0.7_0.1",
            "IoU_max",
            "Dice_max",
            "F1_max",
            "Acc_max",
        ]

        # ==> data
        self.data.type = "RS19Dataset"
        self.data.folder_name = "rs19_val"
        self.data.root = "datasets"
        self.data.meta = "meta.json"
        self.data.cls_names = ["railroad"]
        #### tot of rail images is 7772 #######
        self.data.train_samples = 7008  # 2711
        self.data.test_samples = 764  # 462

        ########## AUG CONFIG 1 #######
        self.data.train_transforms = [
            dict(
                type="Resize",
                size=(self.size, self.size),
                interpolation=F.InterpolationMode.BILINEAR,
            ),
            dict(
                type="ColorJitter",
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.2,
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
                size=(self.size, self.size),
                interpolation=F.InterpolationMode.BILINEAR,
            ),
            dict(type="ToTensor"),
        ]
        self.model = Namespace()
        self.model.name = "simeffseg"
        self.model.kwargs = dict(pretrained=False, checkpoint_path="", strict=True)

        # ==> evaluator
        self.evaluator.kwargs = dict(metrics=self.metrics)
        self.vis = False
        self.vis_dir = "viz"

        # ==> optimizer
        self.optim.lr = self.lr
        # self.optim.kwargs = dict(
        #     name="sgd",
        #     momentum=0.9,
        #     weight_decay=self.weight_decay,
        #     nesterov=True,
        # )
        self.optim.kwargs = dict(name="adam")

        # ==> trainer
        self.trainer.name = "SegTrainer"
        self.trainer.logdir_sub = ""
        self.trainer.resume_dir = ""
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = None
        # self.trainer.scheduler_kwargs = dict(
        #     name="cyclic",
        #     lr_noise=None,
        #     noise_pct=0.67,
        #     noise_std=1.0,
        #     noise_seed=42,
        #     lr_min=self.lr / 1e2,
        #     warmup_lr=self.lr / 1e3,
        #     warmup_iters=-1,
        #     cooldown_iters=0,
        #     warmup_epochs=self.warmup_epochs,
        #     cooldown_epochs=0,
        #     use_iters=True,
        #     patience_iters=0,
        #     patience_epochs=0,
        #     decay_iters=0,
        #     decay_epochs=int(self.epoch_full * 0.8),
        #     cycle_decay=0.1,
        #     decay_rate=0.1,
        # )
        self.trainer.mixup_kwargs = None
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch

        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        # ==> loss
        self.loss.loss_terms = [
            dict(type="DiceLoss", name="dice"),
        ]

        # ==> logging
        self.logging.log_terms_train = [
            dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
            dict(name="data_t", fmt=":>5.3f"),
            dict(name="optim_t", fmt=":>5.3f"),
            dict(name="lr", fmt=":>7.6f"),
            dict(name="dice", suffixes=[""], fmt=":>5.3f", add_name="avg"),
        ]
        self.logging.log_terms_test = [
            dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
            dict(name="dice", suffixes=[""], fmt=":>5.3f", add_name="avg"),
        ]
