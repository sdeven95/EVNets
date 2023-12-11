from . import BaseAnchor
from utils.logger import Logger
from utils.entry_utils import Entry

from itertools import product
import numpy as np
import torch
from torch import Tensor

from typing import Optional


# ssd anchor generator, generally provides six boxes for every anchor point (cx, cy)
# (cx, cy, min_scale, min_scale), (cx, cy, max_scale, max_scale)
# for each aspect ratio, generate two boxes (cx, cy, min_scale * ratio, min_scale / ratio), (cx, cy, min_scale / ratio, min_scale * ratio)
@Entry.register_entry(Entry.Anchor)
class SSDAnchorGenerator(BaseAnchor):
	__slots__ = [
		"output_strides",
		"aspect_ratios",
		"min_scale_ratio", "max_scale_ratio", "steps",
		"clip",
		"output_strides_aspect_ratio", "sizes",
	]
	_keys_ = __slots__[:6]
	_disp_ = __slots__
	_defaults_ = [
				[16, 32, 64, 128, 256, -1],
				[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
				0.1, 1.05, [1],
				True,
	]
	_types_ = [
				(int,),
				(float, ),
				float, float, (int, ),
				bool,
	]

	def __init__(self):
		super().__init__()

		# set aspect ratios list
		self.aspect_ratios = [[2, 3]] * len(self.output_strides) if self.aspect_ratios is None else self.aspect_ratios

		# set step list
		if isinstance(self.steps, int):
			self.steps = [self.steps] * len(self.output_strides)
		elif isinstance(self.steps, list) and len(self.steps) <= len(self.output_strides):
			self.steps = self.steps + [1] * (len(self.output_strides) - len(self.steps))
		else:
			Logger.error(
				"--AnchorConfigure.SSD.step should be either a list of ints with the same length as "
				"the output strides OR an integer"
			)

		# set out_strides_aspect_ratio dict
		self.output_strides_aspect_ratio = {k: v for k, v in zip(self.output_strides, self.aspect_ratios)}
		self.num_anchors_per_out_stride = [2 + 2 * len(ar) for os, ar in self.output_strides_aspect_ratio.items()]

		# set sizes
		scales = np.linspace(self.min_scale_ratio, self.max_scale_ratio, len(self.output_strides) + 1)
		self.sizes = dict()
		for i, s in enumerate(self.output_strides):
			self.sizes[s] = {
				"min": scales[i],
				"max": (scales[i] * scales[i + 1]) ** 0.5,
				"step": self.steps[i],
			}

	# return [num_priors, 4] for given out_stride
	# num_priors = 6 * num_points
	@torch.no_grad()
	def _generate_anchors(
		self, height: int, width: int,
		output_stride: int, device: Optional[str] = "cpu",
		*args, **kwargs
	) -> Tensor:
		min_size_h = self.sizes[output_stride]["min"]
		min_size_w = self.sizes[output_stride]["min"]

		max_size_h = self.sizes[output_stride]["max"]
		max_size_w = self.sizes[output_stride]["max"]
		aspect_ratio = self.output_strides_aspect_ratio[output_stride]

		step = max(1, self.sizes[output_stride]["step"])

		default_anchors_ctr = []

		start_step = max(0, step // 2)

		# Note that feature maps are in [N C H W] format
		for y, x in product(
			range(start_step, height, step), range(start_step, width, step)
		):

			# [x, y, w, h] format
			cx = (x + 0.5) / width
			cy = (y + 0.5) / height

			# small box size
			default_anchors_ctr.append([cx, cy, min_size_w, min_size_h])

			# big box size
			default_anchors_ctr.append([cx, cy, max_size_w, max_size_h])

			# change h/w ratio of the small sized box based on aspect ratios
			for ratio in aspect_ratio:
				ratio = ratio ** 0.5
				default_anchors_ctr.extend(
					[
						[cx, cy, min_size_w * ratio, min_size_h / ratio],
						[cx, cy, min_size_w / ratio, min_size_h * ratio],
					]
				)

		default_anchors_ctr = torch.tensor(
			default_anchors_ctr, dtype=torch.float, device=device
		)
		if self.clip:
			default_anchors_ctr = torch.clamp(default_anchors_ctr, min=0.0, max=1.0)

		return default_anchors_ctr
