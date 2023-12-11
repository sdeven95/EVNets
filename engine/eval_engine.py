import torch
from torch.cuda.amp import autocast
import time

from common import Dev, DDP, Common
from metrics import Metric
from .utils import EngineUtil
from utils.tensor_utils import Tsr
from utils.logger import Logger

from data.constants import DataConstants
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils import Util


class Evaluator(object):
	def __init__(self, model, eval_loader):
		super().__init__()

		self.model = model
		self.eval_loader = eval_loader

		self.dev = Dev()
		self.ddp = DDP()
		self.common = Common()
		self.metric = Metric()

		# exclude loss
		if "loss" in self.metric.val_metric_names:
			self.metric.val_metric_names.pop(self.metric.val_metric_names.index("loss"))

		if Util.is_master_node():
			EngineUtil.print_summary(
				eval_dataloader=eval_loader,
				model=model
			)

		self.eval_fn = self.eval_fn_image
		if self.common.inference_modality == "video":
			self.eval_fn = self.eval_fn_video

	def eval_fn_image(self):
		eval_metrics = Metric(self.metric.val_metric_names)
		self.model.eval()
		with torch.no_grad():
			epoch_start_time = time.time()
			total_samples = len(self.eval_loader)
			processed_samples = 0

			for batch_id, batch in enumerate(self.eval_loader):
				batch = Tsr.move_to_device(batch, self.dev.device)
				input_img, target_label = batch["image"], batch["label"]
				batch_size = input_img.shape[0]
				with autocast(enabled=self.common.mixed_precision):
					predict_label = self.model(input_img)
				processed_samples += batch_size
				metric_value_dict = Metric.metric_calculator(
					predict_label=predict_label,
					target_label=target_label,
					loss=0.0,
					metric_names=self.metric.val_metric_names
				)
				eval_metrics.update(
					metrics_values=metric_value_dict,
					batch_load_time=0.0,
					n=batch_size
				)

				if batch_id % self.common.log_freq == 0 and self.ddp.rank == 0:
					eval_metrics.iter_summary(
						epoch=-1,
						total_samples=total_samples,
						processed_samples=processed_samples,
						epoch_start_time=epoch_start_time,
						learning_rate=0.0,
					)

		eval_metrics.epoch_summary(epoch=-1, stage="evaluation")

	def eval_fn_video(self):
		evaluation_metrics = Metric(self.metric.val_metric_names)

		self.model.eval()

		clips_per_video = Opt.get_by_cfg_path(Entry.Sampler, "clips_per_video")
		voting_fn = Opt.get_by_cfg_path(Entry.Sampler, "clip_out_voting_fn")
		if voting_fn is None:
			voting_fn = "sum"
		voting_fn = voting_fn.lower()

		with torch.no_grad():
			epoch_start_time = time.time()
			total_samples = len(self.eval_loader)
			processed_samples = 0

			for batch_id, batch in enumerate(self.eval_loader):
				batch = Tsr.move_to_device(x=batch, device=self.dev.device)

				input_img, target_label = batch["image"], batch["label"]
				# target_label is Batch*Num_clips
				batch_size_ = target_label.shape[0]
				batch_size = batch_size_ // clips_per_video
				if batch_size_ != (batch_size * clips_per_video):
					Logger.log(
						"Skipping batch. Expected batch size= {}. Got: (bxc:{}x{})".format(
							batch_size_, batch_size, clips_per_video
						)
					)
					continue

				# prediction
				with autocast(enabled=self.common.mixed_precision):
					pred_label = self.model(input_img)

				target_label = target_label.reshape(batch_size, clips_per_video)
				# label is the same for all clips in the video
				target_label = target_label[:, 0]
				pred_label = pred_label.reshape(batch_size, clips_per_video, -1)

				if voting_fn == "sum":
					pred_label = torch.sum(pred_label, dim=1)
				elif voting_fn == "max":
					pred_label = torch.max(pred_label, dim=1)
				else:
					Logger.error(
						"--model.video-classification.clip-out-fusion-fn can be {}. Got: {}".format(
							DataConstants.SUPPORTED_VIDEO_CLIP_VOTING_FN, voting_fn
						)
					)

				processed_samples += batch_size
				metrics_values = Metric.metric_calculator(
					predict_label=pred_label,
					target_label=target_label,
					loss=0.0,
					metric_names=self.metric.val_metric_names,
				)

				evaluation_metrics.update(
					metrics_values=metrics_values, batch_load_time=0.0, n=batch_size
				)

				if batch_id % self.common.log_freq == 0 and self.ddp.rank == 0:
					evaluation_metrics.iter_summary(
						epoch=-1,
						processed_samples=processed_samples,
						total_samples=total_samples,
						epoch_start_time=epoch_start_time,
						learning_rate=0.0,
					)

		evaluation_metrics.epoch_summary(epoch=-1, stage="evaluation")

	def run(self):
		eval_start_time = time.time()
		self.eval_fn()
		eval_duration = time.time() - eval_start_time
		Logger.log(f"Evaluation took {eval_duration} seconds")
