import torch.optim.lr_scheduler
from . import BaseLRScheduler
from utils.entry_utils import Entry
from utils.type_utils import Par


@Entry.register_entry(Entry.Scheduler)
class LambdaLR(BaseLRScheduler, torch.optim.lr_scheduler.LambdaLR):
	__slots__ = ["last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [-1, False]
	_types_ = [int, bool]

	def __init__(
			self,
			lr_lambda,
			last_epoch=None,
			verbose=None,
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class MultiplicativeLR(BaseLRScheduler, torch.optim.lr_scheduler.MultiplicativeLR):
	__slots__ = ["last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [-1, False]
	_types_ = [int, bool]

	def __init__(self, lr_lambda, last_epoch=-2, verbose=False):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class StepLR(BaseLRScheduler, torch.optim.lr_scheduler.StepLR):
	__slots__ = ["step_size", "gamma", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [50, 0.1, -1, False]
	_types_ = [int, float, int, bool]

	def __init__(self, step_size=None, gamma=None, last_epoch=-1, verbose=False):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class MultiStepLR(BaseLRScheduler, torch.optim.lr_scheduler.MultiStepLR):
	__slots__ = ["milestones", "gamma", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [[30, 80], 0.1, -1, False]
	_types_ = [(int,), float, int, bool]

	def __init__(self, milestones=(30, 80), gamma=0.1, last_epoch=-1, verbose=False):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class ConstantLR(BaseLRScheduler, torch.optim.lr_scheduler.ConstantLR):
	__slots__ = ["factor", "total_iters", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0/3, 5, -1, False]
	_types_ = [float, int, int, bool]

	def __init__(self, factor=None, total_iters=None, last_epoch=None, verbose=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class LinearLR(BaseLRScheduler, torch.optim.lr_scheduler.LinearLR):
	__slots__ = ["start_factor", "end_factor", "total_iters", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0/3, 1.0, 5, -1, False]
	_types_ = [float, float, int, int, bool]

	def __init__(self, start_factor=None, end_factor=None, total_iters=None, last_epoch=None, verbose=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class ExponentialLR(BaseLRScheduler, torch.optim.lr_scheduler.ExponentialLR):
	__slots__ = ["gamma", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [0.99, -1, False]
	_types_ = [float, int, bool]

	def __init__(self, gamma=None, last_epoch=None, verbose=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class PolynomialLR(BaseLRScheduler, torch.optim.lr_scheduler.PolynomialLR):
	__slots__ = ["total_iters", "power", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [5, 1.0, False]
	_types_ = [int, float, bool]

	def __init__(self, total_iters=None, power=None, verbose=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class CosineAnnealingLR(BaseLRScheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
	__slots__ = ["T_max", "eta_min", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [50, 0.0, -1, False]
	_types_ = [int, float, int, bool]

	def __init__(
			self,
			T_max=None,
			eta_min=None,
			last_epoch=None,
			verbose=None
	):
		super().__init__(**Par.purify(locals()))


class ChainedScheduler(torch.optim.lr_scheduler.ChainedScheduler):
	pass


@Entry.register_entry(Entry.Scheduler)
class SequentialLR(BaseLRScheduler, torch.optim.lr_scheduler.SequentialLR):
	__slots__ = ["milestones", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [[99, ], -1, False]
	_types_ = [(int, ), int, bool]

	def __init__(self, schedulers, milestones=None, last_epoch=None, verbose=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class ReduceLROnPlateau(BaseLRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
	__slots__ = ["mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lr", "eps", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = ["min", 0.1, 10, 1.0e-4, "rel", 0, 0.0, 1.e-8, False]
	_types_ = [str, float, int, float, str, int, float, float, bool]

	def __init__(
			self, mode=None, factor=None, patience=None, threshold=None, threshold_mode=None,
			cooldown=None, min_lr=None, eps=None, verbose=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class CyclicLR(BaseLRScheduler, torch.optim.lr_scheduler.CyclicLR):
	__slots__ = [
		"base_lr", "max_lr", "step_size_up", "step_size_down", "mode",
		"gamma", "scale_mode", "cycle_momentum", "base_momentum", "max_momentum",
		"last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [
		0.0001, 0.01, 2000, 2000, "triangular",
		1.0, "cycle", True, 0.8, 0.9,
		-1, False
	]
	_types_ = [
		float, float, int, int, str,
		float, str, bool, float, float,
		int, bool
	]

	def __init__(
			self,
			scale_fn,
			base_lr=None, max_lr=None, step_size_up=None, step_size_down=None, mode=None,
			gamma=None, scale_mode=None, cycle_momentum=None, base_momentum=None, max_momentum=None,
			last_epoch=None, verbose=None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class OneCycleLR(BaseLRScheduler, torch.optim.lr_scheduler.OneCycleLR):
	__slots__ = [
		"max_lr", "total_steps", "epochs", "steps_per_epoch", "pct_start",
		"anneal_strategy", "cycle_momentum", "base_momentum", "max_momentum",
		"div_factor", "final_div_factor", "three_phase",
		"last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [
		None, None, None, None, 0.3,
		"cos", True, 0.85, 0.95,
		25.0, 1.0e4, False,
		-1, False
	]
	_types_ = [
		float, int, int, int, float,
		str, bool, float, float,
		float, float, bool,
		int, bool
	]

	def __init__(
			self, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=None,
			anneal_strategy=None, cycle_momentum=None, base_momentum=None, max_momentum=None,
			div_factor=None, final_div_factor=None, three_phase=None,
			last_epoch=None, verbose=None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Scheduler)
class CosineAnnealingWarmRestarts(BaseLRScheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
	__slots__ = ["T_0", "T_mult", "eta_min", "last_epoch", "verbose"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1000, 1, 0.0, -1, False]
	_types_ = [int, int, float, int, bool]

	def __init__(self, T_0=None, T_mult=None, eta_min=None, last_epoch=None, verbose=None):
		super().__init__(**Par.purify(locals()))
