from . import ExtensionBaseLRScheduler
from utils.type_utils import Par
from utils.entry_utils import Entry

import math
import numpy as np

from typing import Tuple


@Entry.register_entry(Entry.Scheduler)
class Cosine(ExtensionBaseLRScheduler):
	__slots__ = ["min_lr", "max_lr", "warmup_step", "period"]
	_keys_ = __slots__[:2]
	_disp_ = __slots__
	_defaults_ = [1e-5, 0.4]
	_types_ = [float, float]

	def __init__(self, min_lr: float = None, max_lr: float = None):
		super().__init__(min_lr=min_lr, max_lr=max_lr)

		# calculate lr increment every warmup step
		if self.warmup_iterations > 0:
			self.warmup_step = (self.max_lr - self.warmup_init_lr) / self.warmup_iterations

		# get total period
		self.period = self.max_iterations - self.warmup_iterations + 1 if self.is_iteration_based else self.max_epochs

	def get_lr(self, epoch, curr_iter):
		if curr_iter < self.warmup_iterations:
			curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
			self.warmup_epochs = epoch
		else:
			if self.is_iteration_based:
				curr_iter = curr_iter - self.warmup_iterations
				curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
					1 + math.cos(math.pi * curr_iter / self.period)
				)
			else:
				adjust_num = self.warmup_epochs + 1 if self.adjust_period_for_epochs else 0
				adjust_den = self.warmup_epochs if self.adjust_period_for_epochs else 0
				curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
					1
					+ math.cos(
						math.pi * (epoch - adjust_num) / (self.period - adjust_den)
					)
				)
		return max(0.0, curr_lr)


@Entry.register_entry(Entry.Scheduler)
class Fixed(ExtensionBaseLRScheduler):
	__slots__ = ["lr", "warmup_step"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 3000]
	_types_ = [float, int]

	def __init__(self, lr: float = None):
		super().__init__(lr=lr)
		if self.warmup_iterations > 0:
			self.warmup_step = (self.lr - self.warmup_init_lr) / self.warmup_iterations

	def get_lr(self, epoch, curr_iter):
		if curr_iter < self.warmup_iterations:
			return max(0.0, self.warmup_init_lr + self.warmup_step * curr_iter)
		else:
			return self.lr


@Entry.register_entry(Entry.Scheduler)
class Step(ExtensionBaseLRScheduler):
	__slots__ = ["lr", "gamma", "decay_epoch", "warmup_step"]
	_keys_ = __slots__[:-1]
	_disp_ = __slots__
	_defaults_ = [0.1, 1.0, 2.4]
	_types_ = [float, float, float]

	def __init__(self, lr: float = None, gamma: float = None, decay_eoch: float = None):
		super().__init__(lr=lr, gamma=gamma, decay_eoch=decay_eoch)

		if self.warmup_iterations > 0:
			self.warmup_step = (self.lr - self.warmup_init_lr) / self.warmup_iterations

	def get_lr(self, epoch, curr_iter):
		if curr_iter < self.warmup_iterations:
			return max(0.0, self.warmup_init_lr + curr_iter * self.warmup_step)
		else:
			return max(0.0, self.lr * self.gamma ** (epoch // self.decay_epoch))


@Entry.register_entry(Entry.Scheduler)
class MultiStep(ExtensionBaseLRScheduler):
	__slots__ = ["lr", "milestones", "warmup_step"]
	_keys_ = __slots__[:-1]
	_disp_ = __slots__
	_defaults_ = [0.1, (99,)]
	_types_ = [float, (int,)]

	def __init__(self, lr: float = None, milestones: Tuple[int, ...] = None):
		super().__init__(lr=lr, milestones=milestones)

		if self.warmup_iterations > 0:
			self.warmup_step = (self.lr - self.warmup_init_lr) / self.warmup_iterations

		if isinstance(self.milestones, int):
			self.milestones = [self.milestones]

		self.milestones = sorted(list(set(self.milestones)))

	def get_lr(self, epoch, curr_iter):
		if curr_iter < self.warmup_iterations:
			return max(0.0, self.warmup_init_lr + curr_iter * self.warmup_step)
		else:
			if epoch in self.milestones:
				self.lr *= self.gamma
				self.milestones.remove(epoch)
			return max(0.0, self.lr)


@Entry.register_entry(Entry.Scheduler)
class Polynomial(ExtensionBaseLRScheduler):
	__slots__ = ["power", "start_lr", "end_lr", "warmup_step", "period"]
	_keys_ = __slots__[:-2]
	_disp_ = __slots__
	_defaults_ = [0.9, 0.1, 0.0]
	_types_ = [float, float, float]

	def __init__(self, power: float = None, start_lr: float = None, end_lr: float = None):
		super().__init__(power=power, start_lr=start_lr, end_lr=end_lr)

		# calculate lr increment every warmup step
		if self.warmup_iterations > 0:
			self.warmup_step = (self.start_lr - self.warmup_init_lr) / self.warmup_iterations

		# get total period
		self.period = self.max_iterations - self.warmup_iterations + 1 if self.is_iteration_based else self.max_epochs

	def get_lr(self, epoch, curr_iter):
		if curr_iter < self.warmup_iterations:
			curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
			self.warmup_epochs = epoch
		else:
			if self.is_iteration_based:
				factor = (curr_iter - self.warmup_iterations) / self.period
			else:
				adjust_num = self.warmup_epochs + 1 if self.adjust_period_for_epochs else 0
				adjust_den = self.warmup_epochs if self.adjust_period_for_epochs else 0
				factor = (epoch - adjust_num) / (self.period - adjust_den)

			curr_lr = (self.start_lr - self.end_lr) * (
				(1.0 - factor) ** self.power
			) + self.end_lr

		return max(0.0, curr_lr)


# divide all epochs to some cycles, on every cycle(include several epochs),  decrease lr from
# max_lr to min_lr, where max_lr = min_lr * epochs_per_cycle
# across several epochs
# on the remaining epochs, decrease lr from min_lr to end_lr
@Entry.register_entry(Entry.Scheduler)
class Cyclic(ExtensionBaseLRScheduler):
	__slots__ = [
		"steps", "gamma", "min_lr", "epochs_per_cycle", "total_cycles", "last_cycle_anneal_type", "last_cycle_end_lr",
		"warmup_step", "max_lr", "cyclic_epochs", "last_cycle_epochs", "epochs_lr_stepped", "cycle_lrs"]
	_keys_ = __slots__[:-6]
	_disp_ = __slots__
	_defaults_ = [25, 0.5, 0.1, 5, 9, "linear", 1e-3]
	_types_ = [(int,), float, float, int, int, str, float]

	SUPPORTED_LAST_CYCLES = ["cosine", "linear"]

	def __init__(
			self,
			steps: Tuple[int, ...] = None,
			gamma: float = None,
			min_lr: float = None,
			epochs_per_cycle: int = None,
			total_cycles: int = None,
			last_cycle_anneal_type: str = None,
			last_cycle_end_lr: float = None
	):
		super().__init__(**Par.purify(locals()))
		assert self.last_cycle_type in Cyclic.SUPPORTED_LAST_CYCLES
		assert self.min_lr >= self.last_cycle_end_lr

		if self.warmup_iterations > 0:
			self.warmup_step = (self.min_lr - self.warmup_init_lr) / self.warmup_iterations

		if self.steps is None:
			self.steps = [self.max_epochs]
		elif isinstance(self.steps, int):
			self.steps = [self.steps]

		self.max_lr = self.min_lr * self.epochs_per_cycle
		self.cyclic_epochs = self.epochs_per_cycle * self.total_cycles
		self.last_cycle_epochs = self.max_epochs - self.cyclic_epochs

		self._lr_per_cycle()
		self.epochs_lr_stepped = []

	# create lr for per epoch in one cycle
	# in fact, it is [min_lr, max_lr, ..., 2 * min_lr], where max_lr = min_lr * epochs_per_cycle
	# for example, if epochs_per_cycle = 5, min_lr = 0.1, then max_lr = 0.5
	# [0.1, 0.5, 0.4, 0.3, 0.2]
	def _lr_per_cycle(self) -> None:
		lrs = list(
			np.linspace(self.max_lr, self.min_lr, self.epochs_per_cycle, dtype=np.float)
		)
		# note here!!! the effect is: [0.1, 0.5, 0.4, 0.3, 0.2] [0.1, 0.5, 0.4, 0.3, 0.2]
		lrs = [lrs[-1]] + lrs[:-1]
		self.cycle_lrs = lrs

	def get_lr(self, epoch, curr_iter):
		if curr_iter < self.warmup_iterations:
			curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
		else:
			# if current epoch need to get lr from cycle lr list
			if epoch <= self.cyclic_epochs:
				# update cycle lr list in the specified steps to make lr decrease as epoch increase
				# steps: specific epochs where lr decrease
				# gamma: decrease rate
				if epoch in self.steps and epoch not in self.epochs_lr_stepped:
					self.min_lr *= self.gamma ** (self.steps.index(epoch) + 1)
					self.max_lr *= self.gamma ** (self.steps.index(epoch) + 1)
					self._lr_per_cycle()
					self.epochs_lr_stepped.append(epoch)
				# get lr for epoch from cycle lr list
				idx = epoch % self.epochs_per_cycle
				curr_lr = self.cycle_lrs[idx]
			# if all cycles have completed, annealing for remain epochs, that is lr is from min_lr to end_lr
			# note : min_lr > end_lr
			else:
				base_lr = self.min_lr
				end_lr = self.last_cycle_end_lr
				if self.last_cycle_anneal_type == "linear":  # linear
					lr_step = (base_lr - end_lr) / self.last_cycle_epochs
					curr_lr = base_lr - (epoch - self.cyclic_epochs + 1) * lr_step
				elif self.last_cycle_anneal_type == "cosine":  # cosine
					curr_epoch = epoch - self.cyclic_epochs
					period = self.max_epochs - self.cyclic_epochs + 1
					curr_lr = end_lr + 0.5 * (base_lr - end_lr) * (
						1 + math.cos(math.pi * curr_epoch / period)
					)
				else:
					raise NotImplementedError
		return max(0.0, curr_lr)
