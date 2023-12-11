from utils import Util
from utils.logger import Logger
from utils.metric_logger import MetricLogger
from utils.tensor_utils import Tsr

import sys
from torch import Tensor
import torch
import numpy as np
import time
from typing import Optional, Any

from .metric_fns import MetricFun


# ======= helper to accumulate metrics and print summary ===========
class Statistics(object):

    SUPPORTED_METRIC = ["loss", "grad_norm", "top1", "top5", "iou", "psnr"]

    # metric_dict = {"loss": None} metric_counter={"loss": 0}
    def __init__(self, metric_names=("loss",)):
        assert len(metric_names) > 0, "Metric names list cannot be empty"

        assert all(
            [name == "coco_map" or name in Statistics.SUPPORTED_METRIC
             for name in metric_names]), "Not all metric are supported"

        self.metric_dict = {name: None for name in metric_names}
        self.metric_names = list(self.metric_dict.keys())
        self.metric_counters = {name: 0 for name in self.metric_names}

        self.round_places = 4  # 4 decimals

        self.batch_load_time = 0
        self.batch_counter = 0

    # metric_dict: {"loss": last_value + v * n }  metric_counter: {"loss": last_value + n }
    def update(
            self, metrics_values: dict, batch_load_time: float, n: Optional[int] = 1
    ) -> None:
        for k, v in metrics_values.items():
            if k in self.metric_names:  # only care for specified metric names
                # first visit
                if self.metric_dict[k] is None:
                    if k == "iou":
                        self.metric_dict[k] = {
                            "inter": v["inter"] * n,
                            "union": v["union"] * n,
                        }
                    else:
                        self.metric_dict[k] = v * n  # {"loss": v * n }
                # again visit, accumulating value
                else:
                    if k == "iou":
                        self.metric_dict[k]["inter"] += v["inter"] * n
                        self.metric_dict[k]["union"] += v["union"] * n
                    else:
                        self.metric_dict[k] += v * n  # {"loss": last_value + v * n }

                self.metric_counters[k] += n  # accumulating count {"loss": last_value + n }
        self.batch_load_time += batch_load_time  # accumulate batch load time
        self.batch_counter += 1  # accumulate batch count

    # metric_stats["loss"] = metric_dict["loss"] / metric_counter["loss"]
    # all metrics
    def avg_statistics_all(self, sep=": ") -> list:
        metric_stats = []
        for k, v in self.metric_dict.items():
            v_avg = self.avg_statistics(k)
            metric_stats.append("{:<}{}{:.4f}".format(k, sep, v_avg))
        return metric_stats

    # only one specific metric
    def avg_statistics(self, metric_name: str) -> float:
        avg_val = None
        if metric_name in self.metric_names:
            counter = self.metric_counters[metric_name]
            v = self.metric_dict[metric_name]

            if metric_name == "iou":
                inter = (v["inter"] * 1.0) / counter
                union = (v["union"] * 1.0) / counter
                iou = inter / union
                if isinstance(iou, Tensor):
                    iou = iou.cpu().numpy()
                # Converting iou from [0, 1] to [0, 100]
                # other metrics are by default in [0, 100 range]
                avg_val = np.mean(iou) * 100.0
            else:
                avg_val = (v * 1.0) / counter

            avg_val = round(avg_val, self.round_places)
        return avg_val

    # helper to print iteration summary of per batch
    # Elapsed Time: time elapsed from epoch start
    # Learning Rate: display parameter be given
    # n_processed_samples: number of handled samples, display parameter be given
    # total_samples: total number of samples, display parameter be given
    # Avg. batch load time: average loading time for batches
    def iter_summary(
            self,
            epoch: int,
            processed_samples: int,  # number of processed samples
            total_samples: int,  # number of total samples
            epoch_start_time: float,  # time when the epoch starts
            learning_rate: float or list,  # learning_rate when the epoch end
    ) -> None:
        if not Util.is_master_node():
            return

        metric_stats = self.avg_statistics_all()  # calculate all metrics' average values
        el_time_str = "Elapsed time: {:5.2f}".format(time.time() - epoch_start_time)  # time elapsed from epoch start
        # print learning rate
        if isinstance(learning_rate, float):
            lr_str = "LR: {:1.6f}".format(learning_rate)
        else:
            learning_rate = [round(lr, 6) for lr in learning_rate]
            lr_str = "LR: {}".format(learning_rate)
        # print current epoch, number of samples processed, total number of samples
        epoch_str = "Epoch: {:3d} [{:8d}/{:8d}]".format(
            epoch, processed_samples, total_samples
        )
        # print average batch load time
        batch_str = "Avg. batch load time: {:1.3f}".format(
            self.batch_load_time / self.batch_counter
        )

        stats_summary = [epoch_str]
        stats_summary.extend(metric_stats)
        stats_summary.append(lr_str)
        stats_summary.append(batch_str)
        stats_summary.append(el_time_str)  # time elapse form epoch start

        summary_str = ", ".join(stats_summary)
        Logger.log(summary_str)
        sys.stdout.flush()

    # helper to print epoch summary ( metrics' average values, for example, loss, top1, top5)
    # average loss/top1/top5 for all batches/iterations
    def epoch_summary(self, epoch: int, stage: Optional[str] = "Training") -> None:
        if not Util.is_master_node():
            return

        metric_stats = self.avg_statistics_all(sep="=")
        metric_stats_str = " || ".join(metric_stats)
        Logger.log("*** {} summary for epoch {}".format(stage.title(), epoch))
        print("\t {}".format(metric_stats_str))
        sys.stdout.flush()

        # log to file
        if "training" in stage.lower():
            loss = self.avg_statistics('loss') if 'loss' in self.metric_names else 0.0
            MetricLogger.log_train(epoch, loss)
        elif "validation" in stage.lower():
            loss = self.avg_statistics('loss') if 'loss' in self.metric_names else 0.0
            top1 = self.avg_statistics('top1') if 'top1' in self.metric_names else 0.0
            top5 = self.avg_statistics('top5') if 'top5' in self.metric_names else 0.0
            if 'ema' in stage.lower():
                MetricLogger.log_ema_val(epoch, loss, top1, top5)
            else:
                MetricLogger.log_val(epoch, loss, top1, top5)

    @staticmethod
    def metric_calculator(
            metric_names: list,
            predict_label: Any,
            target_label: Any,
            loss: Tensor or float,
            grad_norm: Optional = None,
    ):
        metric_vals = dict()
        # loss --> numpy array or float scalar
        if "loss" in metric_names:
            loss = Tsr.tensor_to_numpy_or_float(loss)
            metric_vals["loss"] = loss

        # grad_norm (all gradients norm(module) sum) --> reset to average value across processes
        if "grad_norm" in metric_names:
            if grad_norm is None:
                metric_vals["grad_norm"] = 1e-7  # init value
            else:
                grad_norm = Tsr.tensor_to_numpy_or_float(grad_norm)
                metric_vals["grad_norm"] = grad_norm

        # calculate top1 top5 across processes
        if "top1" in metric_names:
            top_1_acc, top_5_acc = MetricFun.top_k_accuracy(predict_label, target_label, top_k=(1, 5))
            top_1_acc = Tsr.tensor_to_numpy_or_float(top_1_acc)
            metric_vals["top1"] = top_1_acc
            if "top5" in metric_names:
                top_5_acc = Tsr.tensor_to_numpy_or_float(top_5_acc, )
                metric_vals["top5"] = top_5_acc

        # for segmentation task
        if "iou" in metric_names:
            inter, union = MetricFun.compute_miou_batch(prediction=predict_label, target=target_label)

            inter = Tsr.tensor_to_numpy_or_float(inter)
            union = Tsr.tensor_to_numpy_or_float(union)
            metric_vals["iou"] = {"inter": inter, "union": union}

        # similar to mse
        if "psnr" in metric_names:
            psnr = MetricFun.compute_psnr(prediction=predict_label, target=target_label)
            metric_vals["psnr"] = Tsr.tensor_to_numpy_or_float(psnr)

        return metric_vals


