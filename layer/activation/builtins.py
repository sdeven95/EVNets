from torch import nn
from . import BaseActivation
from utils.entry_utils import Entry
from utils.type_utils import Par


@Entry.register_entry(Entry.Activation)
class ReLU(BaseActivation, nn.ReLU):
	__slots__ = ["inplace", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False]
	_types_ = [bool]

	def __init__(self, inplace: bool = None, **kwargs):
		super().__init__(inplace=inplace)


@Entry.register_entry(Entry.Activation)
class ReLU6(BaseActivation, nn.ReLU6):
	__slots__ = ["inplace", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False]
	_types_ = [bool]

	def __init__(self, inplace: bool = None, **kwargs):
		super().__init__(inplace=inplace)


@Entry.register_entry(Entry.Activation)
class LeakyReLU(BaseActivation, nn.LeakyReLU):
	__slots__ = ["inplace", "negative_slope"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 1e-2]
	_types_ = [bool, float]

	def __init__(self, inplace: bool = None, negative_slope: float = None, **kwargs):
		super().__init__(inplace=inplace, negative_slope=negative_slope)


@Entry.register_entry(Entry.Activation)
class Sigmoid(BaseActivation, nn.Sigmoid):
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]


@Entry.register_entry(Entry.Activation)
class Swish(BaseActivation, nn.SiLU):
	__slots__ = ["inplace", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False]
	_types_ = [bool]

	def __init__(self, inplace: bool = None, **kwargs):
		super().__init__(inplace=inplace)


@Entry.register_entry(Entry.Activation)
class GELU(BaseActivation, nn.GELU):
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]


@Entry.register_entry(Entry.Activation)
class Hardsigmoid(BaseActivation, nn.Hardsigmoid):
	__slots__ = ["inplace", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False]
	_types_ = [bool]

	def __init__(self, inplace: bool = None, **kwargs):
		super().__init__(inplace=inplace)


@Entry.register_entry(Entry.Activation)
class Hardswish(BaseActivation, nn.Hardswish):
	__slots__ = ["inplace", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False]
	_types_ = [bool]

	def __init__(self, inplace: bool = None, **kwargs):
		super().__init__(inplace=inplace)


@Entry.register_entry(Entry.Activation)
class PReLU(BaseActivation, nn.PReLU):
	__slots__ = ["num_parameters", "init"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [None, 0.25]
	_types_ = [int, float]

	def __init__(self, num_parameters: int = None, init: float = None, **kwargs):
		super().__init__(num_parameters=num_parameters, init=init)


@Entry.register_entry(Entry.Activation)
class Tanh(BaseActivation, nn.Tanh):
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]
