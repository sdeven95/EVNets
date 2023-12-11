import torch.nn

from utils.type_utils import Cfg, Dsp, Par, Prf
from utils.entry_utils import Entry


class BaseActivation(Cfg, Dsp, Prf):
	_cfg_path_ = Entry.Activation

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		if self.__class__.mro()[-2] == torch.nn.Module:
			torch.nn.Module.__init__(self)
		else:
			Par.init_helper(self, kwargs, super(Prf, self))


# from .builtins import ReLU, ReLU6, LeakyReLU, Sigmoid, Swish, GELU, Hardswish, Hardsigmoid, PReLU, Tanh
# from .extensions import SoftReLU
