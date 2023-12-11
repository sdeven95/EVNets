from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry


# Base class for anchor generators for the task of object detection
class BaseAnchor(Cfg, Dsp, nn.Module):

	_cfg_path_ = Entry.Anchor

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		nn.Module.__init__(self)
		self.anchor_dict = dict()

	def num_anchors_per_out_stride(self):
		raise NotImplementedError

	@torch.no_grad()
	def _generate_anchors(
		self, height: int, width: int, output_stride: int,
		device: Optional[str] = "cpu", *args, **kwargs
	) -> Union[Tensor, Tuple[Tensor, ...]]:
		raise NotImplementedError

	@torch.no_grad()
	def forward(
		self, fm_height: int, fm_width: int, fm_output_stride: int,
		device: Optional[str] = "cpu", *args, **kwargs
	) -> Union[Tensor, Tuple[Tensor, ...]]:
		key = "h_{}_w_{}_os_{}".format(fm_height, fm_width, fm_output_stride)
		if key not in self.anchors_dict:
			default_anchors_ctr = self._generate_anchors(
				height=fm_height, width=fm_width, output_stride=fm_output_stride,
				device=device, *args, **kwargs
			)
			self.anchors_dict[key] = default_anchors_ctr
			return default_anchors_ctr
		else:
			return self.anchors_dict[key]


from .ssd import SSDAnchorGenerator
