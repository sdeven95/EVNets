import torch
from torch import nn
from torch.nn import functional as F
from . import BaseActivation
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Activation)
class SoftReLU(BaseActivation, nn.Module):
	__slots__ = ["inplace", "offset", "offset_to_parameter"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, -2.0, False]
	_types_ = [bool, float, bool]

	def __init__(
		self,
		inplace: bool = None,
		offset: float = None,
		offset_to_parameter: bool = None
	):
		super().__init__(inplace=inplace, offset=offset, offset_to_parameter=offset_to_parameter)
		if self.offset_to_parameter:
			self.offset = nn.Parameter(torch.tensor([self.offset], dtype=torch.float32))

	def forward(self, x):
		return x * F.relu6(x - self.offset, inplace=self.inplace) / 6
