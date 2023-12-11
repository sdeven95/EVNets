import torch.optim
from . import BaseOptim
from utils.entry_utils import Entry
from utils.type_utils import Par

from typing import Optional, Sequence


@Entry.register_entry(Entry.Optimizer)
class SGD(BaseOptim, torch.optim.SGD):
	__slots__ = ["lr", "momentum", "dampening", "nesterov"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 0.9, 0.0, False]
	_types_ = [float, float, float, bool]

	def __init__(
			self,
			params,
			weight_decay: float = None,
			lr: float = None,
			momentum: float = None,
			dampening: float = None,
			nesterov: bool = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class Adadelta(BaseOptim, torch.optim.Adadelta):
	__slots__ = ["lr", "eps", "rho"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 1e-8, 0.9]
	_types_ = [float, float, float]

	def __init__(
			self,
			params,
			lr: float = None,
			rho: float = None,
			eps: float = None,
			weight_decay: float = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class Adagrad(BaseOptim, torch.optim.Adagrad):
	__slots__ = ["lr", "eps", "lr_decay", "initial_accumulator_value"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 1e-8, 0.0, 0.0]
	_types_ = [float, float, float, float]

	def __init__(
			self,
			params,
			lr: float = None,
			lr_decay: float = None,
			weight_decay: float = None,
			initial_accumulator_value: float = None,
			eps: float = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class Adam(BaseOptim, torch.optim.Adam):
	__slots__ = ["lr", "betas", "eps", "amsgrad"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, [0.9, 0.999], 1e-8, False]
	_types_ = [float, (float, ), float, bool]

	def __init__(
			self,
			params,
			lr: float = None,
			betas: Sequence[float] = None,
			eps: float = None,
			weight_decay: float = None,
			amsgrad: bool = None,
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class Adamax(BaseOptim, torch.optim.Adamax):
	__slots__ = ["lr", "betas", "eps", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, [0.9, 0.999], 1e-8]
	_types_ = [float, (float, ), float, ]

	def __init__(
			self,
			params,
			lr: float = None,
			betas: Sequence[float] = None,
			eps: float = None,
			weight_decay: float = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class AdamW(BaseOptim, torch.optim.AdamW):
	__slots__ = ["lr", "betas", "eps", "amsgrad"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, [0.9, 0.999], 1e-8, False]
	_types_ = [float, (float, ), float, bool]

	def __init__(
			self,
			params,
			lr: float = None,
			betas: Sequence[float] = None,
			eps: float = None,
			weight_decay: float = None,
			amsgrad: bool = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class ASGD(BaseOptim, torch.optim.ASGD):
	__slots__ = ["lr", "lambd", "alpha", "t0", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 1e-4, 0.75, 1e6, ]
	_types_ = [float, float, float, float, ]


@Entry.register_entry(Entry.Optimizer)
class LBFGS(BaseOptim, torch.optim.LBFGS):
	__slots__ = ["lr", "max_iter", "max_eval", "tolerance_grad", "tolerance_change", "history_size", "line_search_fn"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 20, 25, 1e-5, 1e-9, 100, None]
	_types_ = [float, int, int, float, float, int, str]

	def __init__(
			self,
			params,
			lr: float = None,
			max_iter: int = None,
			max_eval: Optional[int] = None,
			tolerance_grad: float = None,
			tolerance_change: float = None,
			history_size: int = None,
			line_search_fn: Optional[str] = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class NAdam(BaseOptim, torch.optim.NAdam):
	__slots__ = ["lr", "betas", "eps", "momentum_decay"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, [0.9, 0.999], 1e-8, 4e-3]
	_types_ = [float, (float, ), float, float]

	def __init__(
			self,
			params,
			lr: float = None,
			betas: Sequence[float] = None,
			eps: float = None,
			weight_decay: float = None,
			momentum_decay: float = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class RAdam(BaseOptim, torch.optim.RAdam):
	__slots__ = ["lr", "betas", "eps", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1e-3, [0.9, 0.999], 1e-8, ]
	_types_ = [float, (float, ), float, ]

	def __init__(
			self,
			params,
			lr: float = None,
			betas: Sequence[float] = None,
			eps: float = None,
			weight_decay: float = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class RMSprop(BaseOptim, torch.optim.RMSprop):
	__slots__ = ["lr", "alpha", "eps", "momentum", "centered"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, 0.99, 1e-8, 0.0, False]
	_types_ = [float, float, float, float, bool]

	def __init__(
			self,
			params,
			lr: float = None,
			alpha: float = None,
			eps: float = None,
			weight_decay: float = None,
			momentum: float = None,
			centered: bool = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class Rprop(BaseOptim, torch.optim.Rprop):
	__slots__ = ["lr", "etas", "step_sizes"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, [0.5, 1.2], [1e-6, 50]]
	_types_ = [float, (float, ), (float, )]

	def __init__(
			self,
			params,
			lr: float = None,
			etas: Sequence[float] = None,
			step_sizes: Sequence[float] = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Optimizer)
class SparseAdam(BaseOptim, torch.optim.SparseAdam):
	__slots__ = ["lr", "betas", "eps"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.1, [0.9, 0.999], 1e-8]
	_types_ = [float, (float, ), float]

	def __init__(
			self,
			params,
			lr: float = None,
			betas: Sequence[float] = None,
			eps: float = None
	):
		super().__init__(**Par.purify(locals()))
