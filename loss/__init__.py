from utils.type_utils import Cfg, Dsp, Prf, Par
from utils.entry_utils import Entry
import torch
from torch import Tensor, nn


class BaseLossBuiltin(Cfg, Dsp, Prf):
	__slots__ = ["reduction"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = ["mean"]
	_types_ = [str]

	_cfg_path_ = Entry.Loss

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		Par.init_helper(self, kwargs, super(Prf, self))

	def __repr__(self):
		return super()._repr_by_line()


class BaseLossExtension(Cfg, Dsp, Prf, nn.Module):

	_cfg_path_ = Entry.Loss

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		nn.Module.__init__(self)

	def __repr__(self):
		return super()._repr_by_line()

	def forward(self, prediction, target, input_sample=None):
		raise NotImplementedError

	@staticmethod
	# samples with more frequency( such as, batch_size = 32, class = 3 emerges 16 times) will have less weight
	# classes which do not have samples will have weight 0
	def _class_weights(target: Tensor, n_classes: int, norm_val: float = 1.1) -> Tensor:
		class_hist: Tensor = torch.histc(
			target.float(), bins=n_classes, min=0, max=n_classes - 1
		)
		mask_indices = class_hist == 0

		# normalize between 0 and 1 by dividing by the sum
		norm_hist = torch.div(class_hist, class_hist.sum())
		norm_hist = torch.add(norm_hist, norm_val)  # for following log operation

		# compute class weights.
		# samples with more frequency will have less weight and vice-versa
		class_wts = torch.div(torch.ones_like(class_hist), torch.log(norm_hist))

		# mask the classes which do not have samples in the current batch
		class_wts[mask_indices] = 0.0

		return class_wts.to(device=target.device)


# from .builtins import CrossEntropy, BinaryCrossEntropy
# from .extensions import SSDLoss
