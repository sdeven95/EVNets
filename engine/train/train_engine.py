from common import Common, DDP, Dev
from layer.normalization import AdjustBatchNormMomentum
from data.dataloader import EVNetsDataloader
from data.sampler import SampleEfficientTraining, BaseSampler
from scheduler import ExtensionBaseLRScheduler
from model.utils.ema_util import EMAWraper
from metrics.stats import Statistics
from metrics import Metric

from utils.logger import Logger
from utils.metric_logger import MetricLogger
from utils.checkpoint_utils import CheckPoint
from engine.utils import EngineUtil
from utils.loss_landscape import LossLandscape
from utils.tensor_utils import Tsr
from utils import Util

import os
import shutil
import copy
import time
import gc
import traceback
import numpy as np
from itertools import product

import torch
from torch.cuda.amp import autocast
import torch.distributed as dist

from .util import TrainUtil


class Trainer:
	def __init__(
		self, model: torch.nn.Module, train_loader: EVNetsDataloader, val_loader: EVNetsDataloader, train_sampler: BaseSampler,
		criterion, optimizer: torch.optim.Optimizer, scheduler: ExtensionBaseLRScheduler, gradient_scalar: torch.cuda.amp.GradScaler,
		start_epoch=0, start_iteration=0, best_metric=0.0, ema_cfg: EMAWraper = None,
	):

		self.model, self.train_loader, self.val_loader, self.train_sampler, \
			self.criterion, self.optimizer, self.scheduler, self.gradient_scalar, \
			self.start_epoch, self.current_iteration, self.best_metric, self.ema_cfg = \
			model, train_loader, val_loader, train_sampler, \
			criterion, optimizer, scheduler, gradient_scalar, \
			start_epoch, start_iteration, best_metric, ema_cfg

		self.inner_model = self.model.module if hasattr(self.model, "module") else self.model

		# ============= common settings ==================
		self.dev = Dev()
		self.common = Common()
		self.ddp = DDP()

		# ============ metric settings ====================
		# make sure "loss" in metric names
		self.metric = Metric()
		if "loss" not in self.metric.train_metric_names:
			self.metric.train_metric_names.append("loss")
		if "loss" not in self.metric.val_metric_names:
			self.metric.val_metric_names.append("loss")
		# make sure checkpoint performance metric in validation metric names
		assert self.metric.checkpoint_metric in self.metric.val_metric_names

		# ---------- log settings --------------------
		self.tb_log_writer, self.bolt_log_writer = None, None
		if Util.is_master_node():
			TrainUtil.setup_log_writer(self)
			EngineUtil.print_summary(
				train_dataloader=train_loader,
				val_dataloader=val_loader,
				model=self.model,
				criteria=self.criterion,
				optimizer=self.optimizer,
				scheduler=self.scheduler,
			)

		# ------- adjust normalization momentum ------------
		self.adjust_norm_mom = AdjustBatchNormMomentum()

		# ------- sample-efficient_training ----------
		self.sample_efficient_training_cfg = SampleEfficientTraining()
		if self.sample_efficient_training_cfg.enable:
			self.train_loader_copy = copy.deepcopy(self.train_loader)
			self.sample_ids_orig = self.train_loader_copy.get_sample_indices()
			n_samples = len(self.sample_ids_orig)
			self.running_sum_tensor = torch.zeros(
				(n_samples,), device=self.dev.device, dtype=torch.int
			)
			Logger.log("Configuring for sample efficient training")

		self.max_iteration_reached = False

	def train_epoch(self, epoch):
		time.sleep(2)  # to prevent possible deadlock during epoch transition

		# ------ logging --------------
		Logger.double_dash_line()
		Logger.debug(f"Training epoch {epoch} with {self.train_loader.samples_in_dataset()}")

		# =========== preparing ============
		train_stats = Statistics(metric_names=self.metric.train_metric_names)  # reset Statistics helper
		self.model.train()  # --- to train model -----
		accum_freq = self.common.accum_freq if epoch >= self.common.accum_after_epoch else 1
		max_norm = self.common.grad_clip

		self.optimizer.zero_grad(set_to_none=self.common.set_grad_to_none)  # --- zero grad -----
		epoch_start_time = time.time()
		batch_load_start = time.time()
		grad_norm = 0.0

		# ========= loop by batch ============
		for batch_id, batch in enumerate(self.train_loader):
			# check max iteration limitation
			if self.current_iteration > self.scheduler.max_iterations:
				self.max_iteration_reached = True
				return -1, -1

			# move to gpu
			batch = Tsr.move_to_device(x=batch, device=self.dev.device)

			# ------------- apply mix-up transform if any --------------------
			batch = TrainUtil.apply_mixup_transforms(self, data=batch)

			# -------------- record batch load time -------------------------
			batch_load_time = time.time() - batch_load_start

			input_images, target_labels = batch["image"], batch["label"]
			batch_size = TrainUtil.get_batch_size(input_images)

			# =========== update the learning rate ========
			self.optimizer = self.scheduler.update_lr(
				optimizer=self.optimizer, epoch=epoch, curr_iter=self.current_iteration
			)

			# ----------- adjust bn moment -------------
			if self.adjust_norm_mom.enable:
				self.adjust_norm_mom.adjust_momentum(
					model=self.model, epoch=epoch, iteration=self.current_iteration
				)

			# ============ forward ========
			with autocast(self.common.mixed_precision):
				predict_labels = self.model(input_images)
				loss = self.criterion(
					prediction=predict_labels, target=target_labels
				)

				# check if loss = nan
				if isinstance(loss, torch.Tensor) and torch.isnan(loss):
					Logger.error("Nan encountered in the loss")

			# ========= backward =============
			self.gradient_scalar.scale(loss).backward()

			# ========== step =============
			if (batch_id + 1) % accum_freq == 0:
				self.gradient_scalar.unscale_(self.optimizer)
				# clip gradients
				if max_norm:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
				if "grad_norm" in self.metric.train_metric_names:
					grad_norm = TrainUtil.compute_grad_norm(self)

				# ============ optimizer step ===========
				self.gradient_scalar.step(self.optimizer)
				# ============ update the scale for next batch
				self.gradient_scalar.update()
				# ============ set the gradients to zero or None
				self.optimizer.zero_grad(set_to_none=self.common.set_grad_to_none)

				# ============= extra operations ===========
				self.current_iteration += 1
				if self.ema_cfg.enable:
					self.ema_cfg.update_parameters(self.model)

			# region metrics operations

			# calculate metrics
			metrics_values = Statistics.metric_calculator(
				metric_names=self.metric.train_metric_names,
				predict_label=predict_labels,
				target_label=target_labels,
				loss=loss,
				grad_norm=grad_norm,
			)

			# accumulate metric values
			train_stats.update(
				metrics_values=metrics_values, batch_load_time=batch_load_time, n=batch_size
			)

			# endregion

			# iteration summary
			if batch_id % self.common.log_freq == 0 and Util.is_master_node():
				train_stats.iter_summary(
					epoch=epoch,
					processed_samples=self.current_iteration,
					total_samples=self.scheduler.max_iterations,
					learning_rate=self.scheduler.retrieve_lr(self.optimizer),
					epoch_start_time=epoch_start_time
				)

			batch_load_start = time.time()  # reset batch load time

		# epoch summary
		train_stats.epoch_summary(epoch=epoch, stage="training")

		# calculate return metric values
		avg_loss = train_stats.avg_statistics(metric_name="loss")
		avg_ckpt_metric = train_stats.avg_statistics(metric_name=self.metric.checkpoint_metric)

		gc.collect()

		return avg_loss, avg_ckpt_metric

	def val_epoch(self, epoch, model: torch.nn.Module, extra_str=""):
		if not self.val_loader:
			return 0.0, 0.0

		time.sleep(2)

		# ----------- prepare Statistics ----------------------
		validation_stats = Statistics(metric_names=self.metric.val_metric_names)
		if "coco_map" in self.metric.val_metric_names:
			from metrics.coco_evaluator import COCOEvaluator
			coco_evaluator = COCOEvaluator(iou_types=["bbox"])
		else:
			coco_evaluator = None

		# evaluating
		model.eval()
		with torch.no_grad():
			epoch_start_time = time.time()
			total_samples = len(self.val_loader)
			processed_samples = 0
			lr = 0.0
			batch_start_time = time.time()
			for batch_id, batch in enumerate(self.val_loader):
				batch = Tsr.move_to_device(x=batch, device=self.dev.device)
				input_images, target_labels = batch["image"], batch["label"]
				batch_size = TrainUtil.get_batch_size(input_images)

				batch_load_time = time.time() - batch_start_time

				with autocast(self.common.mixed_precision):
					predict_labels = model(input_images)
					loss = self.criterion(
						prediction=predict_labels, target=target_labels
					)

				processed_samples += batch_size

				metrics_values = Statistics.metric_calculator(
					metric_names=self.metric.val_metric_names,
					predict_label=predict_labels,
					target_label=target_labels,
					loss=loss,
				)
				validation_stats.update(metrics_values=metrics_values, batch_load_time=batch_load_time, n=batch_size)

				if coco_evaluator is not None:
					coco_evaluator.prepare_predictions(
						predictions=predict_labels, targets=target_labels
					)

				if batch_id % self.common.log_freq == 0 and Util.is_master_node():
					validation_stats.iter_summary(
						epoch=epoch,
						total_samples=total_samples,
						processed_samples=processed_samples,
						epoch_start_time=epoch_start_time,
						learning_rate=lr,
					)

		validation_stats.epoch_summary(epoch=epoch, stage="validation" + extra_str)

		avg_loss = validation_stats.avg_statistics(metric_name="loss")
		avg_ckpt_metric = validation_stats.avg_statistics(metric_name=self.metric.checkpoint_metric)

		if coco_evaluator is not None:
			# synchronize across different processes and aggregate the results
			coco_evaluator.gather_coco_results()
			coco_map = coco_evaluator.summarize_coco_results()

			if self.metric.checkpoint_metric == "coco_map" and "bbox" in coco_map:
				avg_ckpt_metric = round(coco_map["bbox"], 5)

		avg_ckpt_metric = avg_ckpt_metric if avg_ckpt_metric else avg_loss

		gc.collect()

		return avg_loss, avg_ckpt_metric

	def run(self):
		save_dir = self.common.exp_loc
		if Util.is_master_node():
			MetricLogger.open_file(os.path.join(save_dir, "metric_log.txt"))
			MetricLogger.print_header()

		cfg_file = self.common.config_file
		if cfg_file and Util.is_master_node():
			dst_cfg_file = f"{save_dir}/config.yaml"
			shutil.copy(src=cfg_file, dst=dst_cfg_file)
			Logger.info(f"Configuration file is stored here: {Logger.color_text(dst_cfg_file)}")

		ema_best_metric = self.best_metric
		train_start_time = time.time()

		try:
			for epoch in range(self.start_epoch, self.scheduler.max_epochs):
				self.train_sampler.set_epoch(epoch)
				self.train_sampler.update_scales()

				train_loss, train_ckpt_metric = self.train_epoch(epoch)
				val_loss, val_ckpt_metric = self.val_epoch(epoch, self.model)

				# copy weights from ema model at specified epoch & validate model
				if self.ema_cfg.enable and epoch == self.ema_cfg.copy_at_epoch:
					Logger.log("Copying EMA weights")
					CheckPoint.copy_weights(model_src=self.ema_cfg, model_tar=self.model)
					Logger.log("EMA weights copied")
					Logger.log("Running validation after Copying EMA model weights")
					self.val_epoch(epoch, model=self.model)

				# update best metric value
				is_best = (val_ckpt_metric >= self.best_metric) if self.metric.checkpoint_metric_max else (val_ckpt_metric <= self.best_metric)
				self.best_metric = val_ckpt_metric if is_best else self.best_metric
				if self.best_metric > self.metric.terminate_ratio:
					break

				# validate ema model
				val_ema_loss, val_ema_ckpt_metric = None, None
				is_ema_best = False
				if self.ema_cfg.enable:
					val_ema_loss, val_ema_ckpt_metric = self.val_epoch(epoch, model=self.ema_cfg.ema_model, extra_str=" (EMA)")
					# update ema best metric
					is_ema_best = (val_ema_ckpt_metric >= ema_best_metric) if self.metric.checkpoint_metric_max else (val_ema_ckpt_metric <= ema_best_metric)
					ema_best_metric = val_ema_ckpt_metric if is_ema_best else ema_best_metric
					if ema_best_metric > self.metric.terminate_ratio:
						break

				# sample efficient training, remove easy samples
				if self.sample_efficient_training_cfg.enable \
					and (epoch + 1) % self.sample_efficient_training_cfg.find_easy_samples_every_k_epochs == 0:
					TrainUtil.find_easy_samples(
						self,
						epoch,
						model=self.ema_cfg.ema_model if self.ema_cfg.enable else self.model  # prefer EMA model
					)

				gc.collect()

				if Util.is_master_node():
					# save checkpoint
					CheckPoint.save_checkpoint(
						iteration=self.current_iteration,
						epoch=epoch,
						model=self.model,
						optimizer=self.optimizer,
						best_metric=self.best_metric,
						is_best=is_best,
						save_dir=save_dir,
						ema_cfg=self.ema_cfg,
						is_ema_best=is_ema_best,
						ema_best_metric=ema_best_metric,
						gradient_scalar=self.gradient_scalar,
						max_metric=self.metric.checkpoint_metric_max,
						k_best_state_dicts=self.metric.k_best_checkpoints,
						save_all_checkpoints=self.metric.save_all_checkpoints,
					)
					Logger.info(
						"Checkpoints saved at: {}".format(save_dir), print_line=True
					)

					# write log
					lr_list = self.scheduler.retrieve_lr(self.optimizer)
					if self.tb_log_writer:
						self.log_metrics(
							lrs=lr_list,
							log_writer=self.tb_log_writer,
							train_loss=train_loss,
							val_loss=val_loss,
							epoch=epoch,
							best_metric=self.best_metric,
							val_ema_loss=val_ema_loss,
							ckpt_metric_name=self.metric.checkpoint_metric,
							train_ckpt_metric=train_ckpt_metric,
							val_ckpt_metric=val_ckpt_metric,
							val_ema_ckpt_metric=val_ema_ckpt_metric,
						)
					if self.bolt_log_writer:
						self.log_metrics(
							lrs=lr_list,
							log_writer=self.bolt_log_writer,
							train_loss=train_loss,
							val_loss=val_loss,
							epoch=epoch,
							best_metric=self.best_metric,
							val_ema_loss=val_ema_loss,
							ckpt_metric_name=self.metric.checkpoint_metric,
							train_ckpt_metric=train_ckpt_metric,
							val_ckpt_metric=val_ckpt_metric,
							val_ema_ckpt_metric=val_ema_ckpt_metric,
						)

				# check iterations
				if self.max_iteration_reached:
					if self.ddp.use_distributed:
						dist.barrier()
					Logger.info("Max. iterations for training reached")
					break

		except KeyboardInterrupt as e:
			Logger.log("Keyboard interruption. Exiting from early training")
			raise e
		except Exception as e:
			if "out of memory" in str(e):
				Logger.log("OOM exception occurred")
				for dev_id in range(self.dev.num_gpus):
					mem_summary = torch.cuda.memory_summary(
						device=torch.device("cuda:{}".format(dev_id)), abbreviated=True
					)
					Logger.log("Memory summary for device id: {}".format(dev_id))
					print(mem_summary)
			else:
				Logger.log(f"Exception occurred that interrupted the training. {str(e)}")
				print(e)
				traceback.print_exc()
				raise e
		finally:
			if self.ddp.use_distributed:
				dist.destroy_process_group()
			torch.cuda.empty_cache()

			if self.ddp.rank == 0:
				MetricLogger.close_file()
				if self.tb_log_writer:
					self.tb_log_writer.close()

				hours, rem = divmod(time.time() - train_start_time, 3600)
				minutes, seconds = divmod(rem, 60)
				train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
					int(hours), int(minutes), seconds
				)
				Logger.log("Training took {}".format(train_time_str))

			try:
				exit(0)
			except Exception:
				pass
			finally:
				pass

	def run_loss_landscape(self):
		# Loss landscape code is adapted from https://github.com/xxxnell/how-do-vits-work
		ll_start_time = time.time()
		try:
			ll_cfg = LossLandscape()

			Logger.log(
				"Loss landscape coord space params: \n\tmin_x={}\n\tmax_x={}\n\tmin_y={}\n\tmax_y={}\n\tn_points={}".format(
					ll_cfg.min_x, ll_cfg.max_x, ll_cfg.min_y, ll_cfg.max_y, ll_cfg.n_points)
			)

			ll_metrics = ["loss"]
			ll_stats = Metric(metric_names=ll_metrics)
			has_module = hasattr(self.model, "module")
			model_name = (
				self.model.module.__class__.__name__
				if has_module
				else self.model.__class__.__name__
			)
			save_dir = Util.get_export_location()

			# copy the model and create bases
			model = copy.deepcopy(self.model)
			weight_state_0 = (
				copy.deepcopy(model.module.state_dict())
				if has_module
				else copy.deepcopy(model.state_dict())
			)

			# create offset matrix for every weight matrix(biased offset are zeros)
			bases = LossLandscape.create_bases(
				model=model, device=self.dev.device, has_module=has_module
			)

			xs = np.linspace(ll_cfg.min_x, ll_cfg.max_x, ll_cfg.n_points)
			ys = np.linspace(ll_cfg.min_y, ll_cfg.max_y, ll_cfg.n_points)

			# create mesh grid
			grid_a, grid_b = np.meshgrid(xs, ys, indexing="xy")
			loss_surface = np.empty_like(grid_a)

			epoch = -1
			for coord_a, coord_b in product(range(ll_cfg.n_points), range(ll_cfg.n_points)):
				epoch += 1
				coords_list = [grid_a[coord_a, coord_b], grid_b[coord_a, coord_b]]  # get point coordinate
				weight_state_1 = copy.deepcopy(weight_state_0)
				# adjust weight offset by coordinate
				gs = [{k: r * bs[k] for k in bs} for r, bs in zip(coords_list, bases)]
				# computer new offset parameters
				gs = {
					k: torch.sum(torch.stack([g[k] for g in gs]), dim=0)
					+ weight_state_1[k]
					for k in gs[0]
				}

				# set offset parameters to model and evaluating
				# load the weights
				model.module.load_state_dict(
					gs
				) if has_module else model.load_state_dict(gs)

				model = model.to(device=self.dev.device)
				model.eval()

				total_samples = len(self.val_loader)
				with torch.no_grad():
					epoch_start_time = time.time()
					processed_samples = 0
					for batch_id, batch in enumerate(self.val_loader):
						batch = Tsr.move_to_device(
							x=batch, device=self.dev.device
						)
						input_img, target_label = batch["image"], batch["label"]

						batch_size = self.__get_batch_size(x=input_img)
						processed_samples += batch_size

						# make the prediction and compute loss
						pred_label = model(input_img)
						loss = self.criterion(
							input_sample=input_img,
							prediction=pred_label,
							target=target_label,
						)

						if isinstance(loss, torch.Tensor) and torch.isnan(loss):
							Logger.error("Nan encountered in the loss.")

						metrics = Statistics.metric_calculator(
							predict_label=pred_label,
							target_label=target_label,
							loss=loss,
							metric_names=ll_metrics,
						)

						ll_stats.update(
							metrics_values=metrics, batch_load_time=0.0, n=batch_size
						)

						if batch_id % self.common.log_freq == 0 and self.ddp.rank == 0:
							ll_stats.iter_summary(
								epoch=epoch,
								processed_samples=processed_samples,
								total_samples=total_samples,
								epoch_start_time=epoch_start_time,
								learning_rate=0.0,
							)

					avg_loss = ll_stats.avg_statistics(metric_name="loss")
					loss_surface[coord_a, coord_b] = avg_loss
					if self.ddp.rank == 0:
						print(
							"x: {:.2f}, y: {:.2f}, loss: {:.2f}".format(
								coords_list[0], coords_list[1], avg_loss
							)
						)

					if self.ddp.rank == 0:
						lr_list = [0.0]

						if self.tb_log_writer is not None:
							self.log_metrics(
								lrs=lr_list,
								log_writer=self.tb_log_writer,
								train_loss=0.0,
								val_loss=avg_loss,
								epoch=epoch,
								best_metric=self.best_metric,
								val_ema_loss=None,
								ckpt_metric_name=None,
								train_ckpt_metric=None,
								val_ckpt_metric=None,
								val_ema_ckpt_metric=None,
							)
						if self.bolt_log_writer is not None:
							self.log_metrics(
								lrs=lr_list,
								log_writer=self.bolt_log_writer,
								train_loss=0.0,
								val_loss=avg_loss,
								epoch=epoch,
								best_metric=self.best_metric,
								val_ema_loss=None,
								ckpt_metric_name=None,
								train_ckpt_metric=None,
								val_ckpt_metric=None,
								val_ema_ckpt_metric=None,
							)

					gc.collect()
					# take a small nap
					time.sleep(1)

			if self.ddp.rank == 0:
				LossLandscape.plot_save_graphs(
					save_dir=save_dir,
					model_name=model_name,
					grid_a=grid_a,
					grid_b=grid_b,
					loss_surface=loss_surface,
					resolution=ll_cfg.n_points,
				)
		except KeyboardInterrupt as e:
			if self.ddp.rank == 0:
				Logger.log("Keyboard interruption. Exiting from early training")
				raise e
		except Exception as e:
			if "out of memory" in str(e):
				Logger.log("OOM exception occured")
				n_gpus = self.dev.num_gpus
				for dev_id in range(n_gpus):
					mem_summary = torch.cuda.memory_summary(
						device=torch.device("cuda:{}".format(dev_id)), abbreviated=True
					)
					Logger.log("Memory summary for device id: {}".format(dev_id))
					print(mem_summary)
			else:
				Logger.log(
					"Exception occurred that interrupted the training. {}".format(
						str(e)
					)
				)
				print(e)
				raise e
		finally:
			if self.ddp.use_distributed:
				torch.distributed.destroy_process_group()

			torch.cuda.empty_cache()

			if self.ddp.rank == 0:
				ll_end_time = time.time()
				hours, rem = divmod(ll_end_time - ll_start_time, 3600)
				minutes, seconds = divmod(rem, 60)
				train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
					int(hours), int(minutes), seconds
				)
				Logger.log("Loss landspace evaluation took {}".format(train_time_str))


			try:
				exit(0)
			except Exception as e:
				pass
			finally:
				pass
