import math
from torch import nn
from utils.type_utils import Cfg, Par, Dsp, Prf
from utils.logger import Logger
from utils.entry_utils import Entry
from scheduler import ExtensionBaseLRScheduler


# the base class of normalization classes except GroupNorm and LayerNorm
class BaseNorm(Cfg, Dsp, Prf):
	__slots__ = ["num_features", "eps", "momentum", "affine", "track_running_stats"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [None, 1e-5, 0.1, True, True]
	_types_ = [int, float, float, bool, bool]

	_cfg_path_ = Entry.Norm

	def __init__(
			self,
			num_features: int = None,
			eps: float = None,
			momentum: float = None,
			affine: bool = None,
			track_running_stats=None,
	):
		para_dict = Par.purify(locals())
		Cfg.__init__(self, **para_dict)
		Par.init_helper(self, para_dict, super(Dsp, self))


# This class enables to adjust the momentum
@Entry.register_entry(Entry.Norm)
class AdjustBatchNormMomentum(Cfg, Dsp):
	__slots__ = [
		"enable", "momentum", "final_momentum_value", "anneal_type",
		"is_iteration_based", "warmup_iterations", "max_steps", "max_momentum", "min_momentum", "anneal_fn", "anneal_method"]
	_keys_ = __slots__[:4]
	_disp_ = __slots__
	_defaults_ = [False, 0.1, 1e-6, "cosine"]
	_types_ = [bool, float, float, str]

	_cfg_path_ = Entry.Norm

	round_places = 6

	def __init__(
			self,
			momentum: float = None,
			final_momentum_value: float = None,
			anneal_type: str = None,
	):
		super().__init__(**Par.purify(locals()))

		scheduler_cfg: ExtensionBaseLRScheduler = Entry.get_entity(Entry.Scheduler)

		self.is_iteration_based = scheduler_cfg.is_iteration_based
		self.warmup_iterations = scheduler_cfg.warmup_iterations

		# -------- get max steps ( max_iterations - warmup_iterations or max epochs)------------
		# if iteration based, adjusted by warmup iterations
		if self.is_iteration_based:
			self.max_steps = scheduler_cfg.max_iterations
			self.max_steps -= self.warmup_iterations
			assert self.max_steps > 0
		# if epochs based, what will happen???
		else:
			Logger.warning(
				"Running {} for epoch-based methods. Not yet validation.".format(
					self.__class__.__name__
				)
			)
			self.max_steps = scheduler_cfg.max_epochs

		# --------- get momentum and min_momentum ( moment range)-------------
		self.max_momentum = self.momentum
		self.min_momentum = self.final_momentum_value
		if self.min_momentum >= self.max_momentum:
			Logger.error(
				"Min. momentum value in {} should be <= momentum. Got {} and {}".format(
					self.__class__.__name__, self.min_momentum, self.max_momentum
				)
			)

		# ------------ get anneal method ------------------
		anneal_method = self.anneal_type
		anneal_method = anneal_method.lower()
		if anneal_method == "cosine":
			self.anneal_fn = self._cosine
		elif anneal_method == "linear":
			self.anneal_fn = self._linear
		else:
			raise RuntimeError(
				"Anneal method ({}) not yet implemented".format(anneal_method)
			)
		self.anneal_method = anneal_method

	# the momentum decreases along cosine curve
	def _cosine(self, step: int) -> float:
		curr_momentum = self.min_momentum + 0.5 * (
				self.max_momentum - self.min_momentum
		) * (1 + math.cos(math.pi * step / self.max_steps))
		return round(curr_momentum, self.round_places)

	# the momentum decreases linearly
	def _linear(self, step: int) -> float:
		momentum_step = (self.max_momentum - self.min_momentum) / self.max_steps  # calculate decreased value every step
		curr_momentum = self.max_momentum - (step * momentum_step)  # compute current momentum
		return round(curr_momentum, self.round_places)

	# just set momentum to 1/step
	def _reciprocal(self, step: int) -> float:
		return round(1.0/step, self.round_places)

	# adjust the momentum of normalization layer of the model according iteration or epoch
	def adjust_momentum(self, model: nn.Module, iteration: int, epoch: int):
		# only do after completing warm up
		if iteration >= self.warmup_iterations:
			# -------- get current step ----------
			# note: iteration based or epoch based
			step = (
				iteration - self.warmup_iterations if self.is_iteration_based else epoch
			)
			curr_momentum = max(0.0, self.anneal_fn(step))

			# adjust all normalization layer momentum of the model
			for m in model.modules():
				if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and m.training:
					m.momentum = curr_momentum


# from .builtins import BatchNorm2d, BatchNorm1d, BatchNorm3d, SyncBatchNorm, \
# 	InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, GroupNorm, LayerNorm
