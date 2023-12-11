import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

from utils.logger import Logger
from utils.file_utils import File
from utils.tensor_utils import Tsr
import time


class TrainUtil:

	@staticmethod
	def get_batch_size(x):
		if isinstance(x, torch.Tensor):
			return x.shape[0]
		elif isinstance(x, dict):
			return x["image"].shape[0]

	@staticmethod
	def compute_grad_norm(trainer):
		parameters = [p for p in trainer.model.parameters() if p.grad is not None]
		if len(parameters) == 0:
			return None

		norm_type = 2.0  # L2 norm

		# note: need to adjust according gradient scalar
		# inv_scale = 1.0 / self.gradient_scalar.get_scale()
		total_norm = torch.norm(
			torch.stack(
				[
					# torch.norm(p.grad.detach() * inv_scale, norm_type).to(self.dev.device)
					torch.norm(p.grad.detach(), norm_type).to(trainer.dev.device)
					for p in parameters
				]
			),
			norm_type,
		)
		if total_norm.isnan() or total_norm.isinf():
			return None
		return total_norm

	@staticmethod
	def apply_mixup_transforms(trainer, data):
		from data.transform.torch import RandomMixup, RandomCutmix
		import random
		mixup_transforms = []
		mixup = RandomMixup(num_classes=trainer.inner_model.n_classes)
		if mixup.enable:
			mixup_transforms.append(mixup)
		cutmix = RandomCutmix(num_classes=trainer.inner_model.n_classes)
		if cutmix.enable:
			mixup_transforms.append(cutmix)

		if len(mixup_transforms) > 0:  # random choice one
			_mixup_transform = random.choice(mixup_transforms)
			data = _mixup_transform(data)
		return data

	# identify easy samples in the training set and removes them from training (epoch unit based)
	@staticmethod
	def find_easy_samples(self, epoch, model: torch.nn.Module):
		time.sleep(2)  # to prevent possible deadlock during epoch transition
		model.eval()
		Logger.log(f"Trying to find easy samples in epoch {epoch}")
		with torch.no_grad():
			# count the easy samples
			easy_sample_ids_tensor = torch.zeros_like(self.running_sum_tensor)
			for batch_id, batch in enumerate(self.train_loader_copy):
				batch = Tsr.move_to_device(batch, self.dev.device)
				input_images, target_labels = batch["image"], batch["label"]
				if "sample_id" in batch:
					sample_ids = batch["sample_id"]
				else:
					self.sample_efficient_training_cfg.enable = False
					Logger.log(
							"Sample Ids are required in a batch for sample efficient training."
							"sample_id not found in batch. Disabling sample efficient training."
						)
					break
				if not sample_ids:
					Logger.log("Sample Ids can't be none")
					break

				with autocast(self.common.mixed_precision):
					predict_labels = model(input_images)
					predict_labels = F.softmax(predict_labels, dim=-1)

				predict_confidences, predict_indices = torch.max(predict_labels, dim=-1)
				easy_samples = torch.logical_and(
					predict_indices.eq(predict_labels),  # condition 1: predicted label == target label
					predict_confidences >= torch.as_tensor(self.sample_efficient_training_cfg.sample_confidence),  # condition 2: prediction confidence >= specified confidence
				)

				if easy_samples.numel() > 0:
					easy_sample_ids = sample_ids[easy_samples]
					# find easy samples as per condition 1 and 2 and set their values to 1
					easy_sample_ids_tensor[easy_sample_ids] = 1

			# synchronize tensors
			if self.ddp.use_distributed:
				# sync across all GPUs
				easy_sample_ids_tensor = Tsr.reduce_tensor_sum(easy_sample_ids_tensor)

			# convert 0 to -1, so easy = 1, hard = -1
			easy_sample_ids_tensor[easy_sample_ids_tensor == 0] = -1

			Logger.debug(
				"Number of easy samples found during epoch {} are {}".format(
					epoch,
					easy_sample_ids_tensor[easy_sample_ids_tensor > 0].sum().item()
				)
			)

			# update running_sum_tensor which track samples' state across epochs
			# that is, accumulating easy flag for every sample
			# in each epoch, easy flag of every sample is 1 or -1
			# so element of running_sum_tensor could be < 0, == 0 or > 0
			# more bigger, more easy
			# but the element will be clipped between 0 and 5
			# so following will occur: (1 or -1) + (0~5)
			self.running_sum_tensor = torch.clip(
				self.running_sum_tensor + easy_sample_ids_tensor,
				min=0,
				max=self.sample_efficient_training_cfg.min_sample_frequency,  # default 5
			)

			# only the accumulated value which surpasses min_sample_frequency will be skipped
			if self.running_sum_tensor.sum() > 0:
				skip_sample_ids = (
					self.running_sum_tensor >= self.sample_efficient_training_cfg.min_sample_frequency
				).nonzero(as_tuple=True)[0]

				if skip_sample_ids.numel() > 0:
					skip_samples = skip_sample_ids.cpu().numpy().tolist()

					new_sample_ids = [
						s_id
						for s_id in self.sample_ids_orig
						if s_id not in skip_sample_ids
					]

					# update the train loader indices
					self.train_loader.update_indices(new_sample_ids)

					Logger.debug(f"Number of samples to skip after epoch {epoch} are {len(skip_samples)}")

	@staticmethod
	def setup_log_writer(trainer):
		if trainer.common.tensorboard_logging:
			try:
				from torch.utils.tensorboard import SummaryWriter
			except ImportError:
				Logger.log(
					"Unable to import SummaryWriter from torch.utils.tensorboard. Disabling tensorboard logging"
				)
			if SummaryWriter:
				exp_dir = f"{trainer.common.exp_loc}/tb_logs"
				File.create_directories(exp_dir)
				trainer.tb_log_writer = SummaryWriter(
					log_dir=exp_dir, comment="Training and Validation logs"
				)

		if trainer.common.bolt_logging:
			try:
				from utils.bolt_logger import BoltLogger
			except ModuleNotFoundError:
				BoltLogger = None

			if not BoltLogger:
				Logger.log("Unable to import bolt. Disabling bolt logging")
			else:
				trainer.bolt_log_writer = BoltLogger()

	@staticmethod
	def log_metrics(
			log_writer,
			lrs,
			train_loss,
			val_loss,
			epoch,
			best_metric,
			val_ema_loss,
			ckpt_metric_name,
			train_ckpt_metric,
			val_ckpt_metric,
			val_ema_ckpt_metric
	):

		# learning rates
		lrs = lrs if isinstance(lrs, list) else [lrs]
		for lr_id, lr_val in enumerate(lrs):
			log_writer.add_scalar(f"LR/Group-{lr_id}", round(lr_val, 6), epoch)

		# losses
		log_writer.add_scalar("Train/Loss", round(train_loss, 2), epoch)
		log_writer.add_scalar("Val/Loss", round(val_loss, 2), epoch)
		log_writer.add_scalar("Common/Best Metric", round(best_metric, 2), epoch)
		if val_ema_loss:
			log_writer.add_scalar("Val_EMA/Loss", round(val_ema_loss, 2), epoch)

		# another metric except loss
		if ckpt_metric_name and ckpt_metric_name != "loss":
			if train_ckpt_metric:
				log_writer.add_scalar(f"Train{ckpt_metric_name.title()}", train_ckpt_metric, epoch)
			if val_ckpt_metric:
				log_writer.add_scalar(f"Val{ckpt_metric_name.title()}", val_ckpt_metric, epoch)
			if val_ema_ckpt_metric:
				log_writer.add_scalar(f"Val_EMA{ckpt_metric_name.title()}", val_ema_ckpt_metric, epoch)
