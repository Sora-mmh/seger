import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from utils.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from utils.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from utils.net import get_loss_scaler, get_autocast, distribute_bn


import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base import BaseTrainer
from . import TRAINER
from utils.trainer import BatchVisualizer
from utils.vis import vis_seg_map


@TRAINER.register_module
class SegTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(SegTrainer, self).__init__(cfg)
        self.tsrboard = BatchVisualizer(cfg)

    def set_input(self, inputs):
        self.imgs = inputs["img"].cuda()
        self.imgs_masks = inputs["img_mask"].cuda()
        self.imgs_paths = inputs["img_path"]
        self.labels = inputs["cls_name"]
        self.bs = self.imgs.shape[0]

    def forward(self):
        self.pr_masks = self.net(self.imgs)

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(
                self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device)
            )
        with self.amp_autocast():
            self.forward()
            seg_loss = self.loss_terms["dice"](self.imgs_masks, self.pr_masks)

        self.tsrboard.plot_loss(seg_loss, self.iter, loss_name="dice loss")
        self.tsrboard.visualize_image_batch(
            self.imgs_masks, self.iter, image_name="GT masks"
        )
        self.tsrboard.visualize_image_batch(
            self.pr_masks, self.iter, image_name="PR masks"
        )
        self.backward_term(seg_loss, self.optim)
        update_log_term(
            self.log_terms.get("dice"),
            reduce_tensor(seg_loss, self.world_size).clone().detach().item(),
            1,
            self.master,
        )

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        imgs_masks, pr_masks, labels = [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            batch_idx += 1
            t1 = get_timepc()
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            seg_loss = self.loss_terms["dice"](self.imgs_masks, self.pr_masks)
            update_log_term(
                self.log_terms.get("dice"),
                reduce_tensor(seg_loss, self.world_size).clone().detach().item(),
                1,
                self.master,
            )

            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_seg_map(
                    self.imgs_paths,
                    self.imgs,
                    self.imgs_masks.cpu().numpy().astype(int),
                    self.pr_masks.cpu().numpy(),
                    self.cfg.model.name,
                    root_out,
                )
            imgs_masks.append(self.imgs_masks.cpu().numpy().astype(int))
            pr_masks.append(self.pr_masks.cpu().numpy())
            labels.append(np.array(self.labels))
            t2 = get_timepc()
            update_log_term(self.log_terms.get("batch_t"), t2 - t1, 1, self.master)
            print(f"\r{batch_idx}/{test_length}", end="") if self.master else None
            # ---------- log ----------
            if self.master:
                if (
                    batch_idx % self.cfg.logging.test_log_per == 0
                    or batch_idx == test_length
                ):
                    msg = able(
                        self.progress.get_msg(
                            batch_idx, test_length, 0, 0, prefix=f"Test"
                        ),
                        self.master,
                        None,
                    )
                    log_msg(self.logger, msg)

        ######################
        ######################
        ######################
        ######################
        # merge results
        if self.cfg.dist:
            results = dict(cls_names=labels, imgs_masks=imgs_masks, pr_masks=pr_masks)
            # torch.save(
            #     results,
            #     f"{self.tmp_dir}/{self.rank}.pth",
            #     _use_new_zipfile_serialization=False,
            # )
            if self.master:
                results = dict(imgs_masks=[], pr_masks=[], cls_names=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f"{self.tmp_dir}/*.pth")
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True
        else:
            results = dict(cls_names=labels, imgs_masks=imgs_masks, pr_masks=pr_masks)
        ######################
        ######################
        ######################
        ######################

        if self.master:
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg["Name"] = msg.get("Name", [])
                msg["Name"].append(cls_name)
                avg_act = (
                    True
                    if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1
                    else False
                )
                msg["Name"].append("Avg") if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f"{metric}_{cls_name}"].append(metric_result)
                    max_metric = max(self.metric_recorder[f"{metric}_{cls_name}"])
                    max_metric_idx = (
                        self.metric_recorder[f"{metric}_{cls_name}"].index(max_metric)
                        + 1
                    )
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f"{metric} (Max)"] = msg.get(f"{metric} (Max)", [])
                    msg[f"{metric} (Max)"].append(
                        f"{max_metric:.3f} (epoch {max_metric_idx:<3d})"
                    )
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f"{metric}_Avg"].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f"{metric}_Avg"])
                        max_metric_idx = (
                            self.metric_recorder[f"{metric}_Avg"].index(max_metric) + 1
                        )
                        msg[metric].append(metric_result_avg)
                        msg[f"{metric} (Max)"].append(
                            f"{max_metric:.3f} (epoch {max_metric_idx:<3d})"
                        )
                # msg["Opt_Thres_IoU_max"] = [metric_results["Opt_Thres_IoU_max"]]
                # msg["Opt_Thres_Dice_max"] = [metric_results["Opt_Thres_Dice_max"]]
            msg = tabulate.tabulate(
                msg,
                headers="keys",
                tablefmt="pipe",
                floatfmt=".3f",
                numalign="center",
                stralign="center",
            )
            log_msg(self.logger, f"\n{msg}")

            # cls_key = ["Name"]
            # group1_keys = [
            #     "IoU_rng_0.3_0.7_0.1",
            #     "IoU_rng_0.3_0.7_0.1 (Max)",
            #     "Dice_rng_0.3_0.7_0.1",
            #     "Dice_rng_0.3_0.7_0.1 (Max)",
            # ]
            # group2_keys = ["IoU_max", "IoU_max (Max)", "Dice_max", "Dice_max (Max)"]
            # group3_keys = ["Opt_Thres_IoU_max", "Opt_Thres_Dice_max"]
            # cls = {key: msg[key] for key in cls_key}
            # group1 = {key: msg[key] for key in group1_keys}
            # group2 = {key: msg[key] for key in group2_keys}
            # group3 = {key: msg[key] for key in group3_keys}
            # group4 = {key: msg[key] for key in group4_keys}

            # def dict_to_tabulate(data):
            #     headers = list(data.keys())
            #     rows = list(zip(*data.values()))
            #     return headers, rows

            # headers0, rows0 = dict_to_tabulate(cls)
            # headers1, rows1 = dict_to_tabulate(group1)
            # headers2, rows2 = dict_to_tabulate(group2)
            # headers3, rows3 = dict_to_tabulate(group3)
            # headers4, rows4 = dict_to_tabulate(group4)
            # table1 = tabulate.tabulate(
            #     rows1,
            #     headers=headers1,
            #     tablefmt="fancy_grid",
            #     numalign="right",
            #     stralign="center",
            # )
            # table2 = tabulate.tabulate(
            #     rows2,
            #     headers=headers2,
            #     tablefmt="fancy_grid",
            #     numalign="right",
            #     stralign="center",
            # )
            # table3 = tabulate.tabulate(
            #     rows3,
            #     headers=headers3,
            #     tablefmt="fancy_grid",
            #     numalign="right",
            #     stralign="center",
            # )
            # log_msg(self.logger, f"\n{table1}")
            # log_msg(self.logger, f"\n{table2}")
            # log_msg(self.logger, f"\n{table3}")
