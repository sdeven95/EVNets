from utils.type_utils import Cfg, Dsp, Par
from utils.entry_utils import Entry
import torch


class BaseLRScheduler(Cfg, Dsp):

	_cfg_path_ = Entry.Scheduler

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		Par.init_helper(self, kwargs, super(Dsp, self))


class ExtensionBaseLRScheduler(Cfg, Dsp):
	__slots__ = [
		"lr_multipliers", "is_iteration_based", "max_epochs", "max_iterations",
		"warmup_iterations", "warmup_init_lr", "adjust_period_for_epochs",
		"noise_type", "noise_range_percent", "max_noise_percent", "noise_seed",
		"round_places", "warmup_epochs", "noise_range"
	]
	_cfg_path_ = Entry.Scheduler
	_keys_ = __slots__[:-3]
	_disp_ = __slots__
	_defaults_ = [
		None, False, 300, 150000,
		0, 1e-7, False,
		None, [0.42, 0.9], 0.67, 42
	]
	_types_ = [
		(float,), bool, int, int,
		int, float, bool,
		str, (float,), float, int
	]

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.round_places = 8
		self.warmup_epochs = 0

		if self.noise_type is not None:
			if isinstance(self.noise_range_percent, (list, tuple)):
				self.noise_range = [p * self.max_epochs for p in self.noise_range_percent]
				if len(self.noise_range) == 1:
					self.noise_range = self.noise_range[0]
			else:
				self.noise_range = self.noise_range_percent * self.max_epochs
		else:
			self.noise_range = None

	def __repr__(self):
		return super()._repr_by_line()

	@staticmethod
	def retrieve_lr(optimizer):
		lr_list = []
		for param_group in optimizer.param_groups:
			lr_list.append(param_group["lr"])
		return lr_list

	def get_lr(self, epoch, curr_iter):
		raise NotImplementedError

	# post handling for learning rate ( noise and multiplier)
	def update_lr(self, optimizer, epoch, curr_iter):
		lr = self.get_lr(epoch, curr_iter)
		lr = max(0.0, lr)
		# add noise to lr according epoch
		#  this is a new function, be carefully!!!
		lr = self._add_noise([lr], epoch)[0]
		if self.lr_multipliers is not None:
			assert len(self.lr_multipliers) == len(optimizer.param_groups)
			for g_id, param_group in enumerate(optimizer.param_groups):
				param_group["lr"] = round(
					lr * self.lr_multipliers[g_id], self.round_places
				)
		else:
			for param_group in optimizer.param_groups:
				param_group["lr"] = round(lr, self.round_places)
		return optimizer

	# epoch_based, add noise if current epoch in some range
	def _add_noise(self, lrs, epoch):
		if self._is_apply_noise(epoch):
			noise = self._calculate_noise_percent(epoch)
			lrs = [v + v * noise for v in lrs]
		return lrs

	def _is_apply_noise(self, epoch):
		"""Return True if epoch in specific range."""
		if self.noise_range is not None:
			if isinstance(self.noise_range, (list, tuple)):
				return self.noise_range[0] <= epoch < self.noise_range[1]
			else:
				return epoch >= self.noise_range
		return False

	def _calculate_noise_percent(self, epoch):
		g = torch.Generator()
		g.manual_seed(self.noise_seed + epoch)
		if self.noise_type == 'normal':
			while True:
				# resample if noise out of percent limit, brute force but shouldn't spin much
				noise = torch.randn(1, generator=g).item()
				if abs(noise) < self.max_noise_percent:
					return noise
		else:
			# noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.max_noise_percent
			raise ValueError("Only support normal noise type currently")

# obsoleted
# from .builtins import Cosine, Fixed, Step, MultiStep, Polynomial, Cyclic
