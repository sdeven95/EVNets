import math
import torch

from common import Dev, DDP, Common
from metrics import Metric
from gbl import Constants

from utils.entry_utils import Entry
from utils.logger import Logger
from utils.checkpoint_utils import CheckPoint

from .utils import RunUtil

from engine.train.train_engine import Trainer


class TrainModel:
	@classmethod
	def runner(cls):
		dev = Dev()
		ddp = DDP()
		cmm = Common()
		stat = Metric()

		# -------- dataloader --------------
		train_loader, val_loader, train_sampler = Entry.get_entity(Entry.DataLoader, entry_name="train_val")

		# correct max_iterations(iteration based) or max_epochs(epoch based)
		scheduler = Entry.get_entity(Entry.Scheduler)
		if scheduler.is_iteration_based:
			# make sure of right number of iterations(i.e. max_iterations)
			if scheduler.max_iterations is None or scheduler.max_iterations <= 0:
				scheduler.max_iterations = Constants.DEFAULT_ITERATIONS
				Logger.log(f"Setting max. iterations to {Constants.DEFAULT_ITERATIONS}")
				scheduler.max_epochs = Constants.DEFAULT_MAX_EPOCHS  # big as possible for iteration_based training
			Logger.log(f"Max. iterations for training: {scheduler.max_iterations}")
		else:
			# make sure of right number of epochs(i.e. max_epochs)
			if scheduler.max_epochs is None or scheduler.max_epochs <= 0:
				scheduler.max_epochs = Constants.DEFAULT_EPOCHS
				Logger.log(f"Setting max. epochs to {Constants.DEFAULT_EPOCHS}")
				scheduler.max_iterations = Constants.DEFAULT_MAX_ITERATIONS  # big as possible for epoch_based training
			Logger.log(f"Max. epochs for training: {scheduler.max_epochs}")

		# --------- model ------------------
		model = Entry.get_entity(Entry.Model)
		memory_format = torch.channels_last if cmm.channel_last else torch.contiguous_format
		if dev.num_gpus == 0:
			Logger.warning("No GPUs are available, so training on CPU. Consider training on GPU for faster training")
			model = model.to(device=dev.device, memory_format=memory_format)
		elif dev.num_gpus == 1:
			model = model.to(device=dev.device, memory_format=memory_format)
		elif ddp.use_distributed:
			model = model.to(device=dev.device, memory_format=memory_format)
			model = torch.nn.parallel.DistributedDataParallel(
				module=model,
				device_ids=[dev.device_id],
				output_device=dev.device_id,
				find_unused_parameters=ddp.find_unused_params,
			)
			Logger.log("Using DistributedDataParallel for training")
		else:
			model = model.to(memory_format=memory_format)
			model = torch.nn.DataParallel(model)
			model = model.to(dev.device)
			Logger.log("Using DataParallel for training")

		# --------- criteria (loss function) ----------------
		criteria: torch.nn.Module = Entry.get_entity(Entry.Loss)
		criteria = criteria.to(device=dev.device)

		# --------- optim ------------------
		optimizer = Entry.get_entity(Entry.Optimizer, model=model)

		# ---------- Gradient Scalar ----------------
		gradient_scalar = torch.cuda.amp.GradScaler(enabled=cmm.mixed_precision)

		# ---------- recreate scheduler ---------------------
		scheduler = Entry.get_entity(Entry.Scheduler)

		# ----------- ema --------------------------
		ema_cfg = Entry.get_entity(Entry.EMAConfigure, model=model)
		if ema_cfg.enable:
			Logger.log("Using EMA")

		# ----------- init state -------------------
		best_metric = 0.0 if stat.checkpoint_metric_max else math.inf
		start_epoch, start_iteration = 0, 0

		# resume if it was interrupted
		if cmm.auto_resume or cmm.resume_loc:
			(
				model,
				optimizer,
				gradient_scalar,
				start_epoch,
				start_iteration,
				best_metric,
				ema_cfg,
			) = CheckPoint.load_checkpoint(
				model=model,
				optimizer=optimizer,
				gradient_scalar=gradient_scalar,
				ema_cfg=ema_cfg,
			)
		# load parameters, for fine tune ???
		elif cmm.model_state_loc is not None:
			model, ema_cfg = CheckPoint.load_model_state(model=model, ema_cfg=None)
			Logger.log("Load model and model_ema states from path: {}".format(cmm.model_state_loc))

		# run training
		training_engine = Trainer(
			model=model,
			val_loader=val_loader,
			train_loader=train_loader,
			optimizer=optimizer,
			criterion=criteria,
			scheduler=scheduler,
			start_epoch=start_epoch,
			start_iteration=start_iteration,
			best_metric=best_metric,
			ema_cfg=ema_cfg,
			gradient_scalar=gradient_scalar,
			train_sampler=train_sampler,
		)
		training_engine.run()

	@classmethod
	def run(cls, model_cls):
		RunUtil.start_worker(model_cls, cls.runner)


