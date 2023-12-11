from torch import nn, Tensor
import torch
import torch.nn.functional as F

from utils.type_utils import Par
from utils.entry_utils import Entry

from . import BaseLayer, ConvLayer2D
from model.classification.cvnets.mobilenetv1 import SeparableConv

from typing import Union, Tuple, Type, List, Dict


@Entry.register_entry(Entry.Layer)
class FeaturePyramidNetwork(BaseLayer):
	"""
	This class implements the `Feature Pyramid Network <https://arxiv.org/abs/1612.03144>`_ module for object detection.

	Args:
		in_channels (List[int]): List of channels at different output strides
		output_strides (List[int]): Feature maps from these output strides will be used in FPN
		out_channels (int): Output channels

	"""

	__slots__ = ["in_channels_list", "out_strides_list", "out_channels"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [None, None, None]
	_types_ = [(int,), (int, ), int]

	def __init__(
		self,
		in_channels: List[int] = None,
		output_strides: List[int] = None,
		out_channels: int = None,
	) -> None:
		super().__init__(**Par.purify(locals()))

		if isinstance(in_channels, int):
			in_channels = [in_channels]
		if isinstance(output_strides, int):
			output_strides = [output_strides]

		assert len(in_channels) == len(output_strides), f"len(in_channels) != len(out_strides), i.e. {len(in_channels)} != {len(output_strides)}"

		self.proj_layers = nn.ModuleDict()
		self.nxn_convs = nn.ModuleDict()

		for os, in_channel in zip(output_strides, in_channels):
			proj_layer = ConvLayer2D(
				in_channels=in_channel,
				out_channels=out_channels,
				kernel_size=1,
				bias=False,
				use_norm=True,
				use_act=False,
			)
			nxn_conv = ConvLayer2D(
				in_channels=out_channels,
				out_channels=out_channels,
				kernel_size=3,
				bias=False,
				use_norm=True,
				use_act=False,
			)

			self.proj_layers.add_module(name="os_{}".format(os), module=proj_layer)
			self.nxn_convs.add_module(name="os_{}".format(os), module=nxn_conv)

		self.num_fpn_layers = len(in_channels)
		self.out_channels = out_channels
		self.output_strides = output_strides

		# init weights
		from layer.utils.init_params_util import InitParaUtil
		from .normalization import BaseNorm
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				InitParaUtil.initialize_conv_layer(m, init_method="xavier_uniform")
			elif isinstance(m, BaseNorm):
				InitParaUtil.initialize_norm_layers(m)

	def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
		assert len(x) == self.num_fpn_layers

		# dictionary to store results for fpn
		fpn_out_dict = {f"os_{os}": None for os in self.output_strides}

		# process the last output stride
		os_key = f"os_{self.output_strides[-1]}"
		prev_x = self.proj_layers[os_key](x[os_key])
		prev_x = self.nxn_convs[os_key](prev_x)
		fpn_out_dict[os_key] = prev_x

		remaining_output_strides = self.output_strides[:-1]

		# bottom-up processing
		for os in remaining_output_strides[::-1]:
			os_key = f"os_{os}"
			# 1x1 conv
			curr_x = self.proj_layers[os_key](x[os_key])
			# up_sampling
			prev_x = F.interpolate(prev_x, size=curr_x.shape[-2:], mode="nearest")
			# add
			prev_x = curr_x + prev_x
			prev_x = self.nxn_convs[os_key](prev_x)
			fpn_out_dict[os_key] = prev_x

		return fpn_out_dict

	def profile(self, x: Dict[str, Tensor]) -> (Dict[str, Tensor], float, float):
		params, macs = 0.0, 0.0

		# dictionary to store results for fpn
		fpn_out_dict = {f"os_{os}": None for os in self.output_strides}

		# process the last output stride
		os_key = f"os_{self.output_strides[-1]}"
		prev_x, p, m = self.proj_layers[os_key].profile(x=x[os_key])  # module_profile(module=self.proj_layers[os_key], x=x[os_key])
		params += p
		macs += m

		prev_x, p, m = self.nxn_convs[os_key].profile(x=prev_x)  # module_profile(module=self.nxn_convs[os_key], x=prev_x)
		params += p
		macs += m

		fpn_out_dict[os_key] = prev_x

		remaining_output_strides = self.output_strides[:-1]

		for os in remaining_output_strides[::-1]:
			# 1x1 conv
			os_key = f"os_{os}"
			curr_x, p, m = self.proj_layers[os_key].profile(x=x[os_key])  # module_profile(module=self.proj_layers[os_key], x=x[os_key])
			params += p
			macs += m

			# up_sampling
			prev_x = F.interpolate(prev_x, size=curr_x.shape[-2:], mode="nearest")
			# add
			prev_x = curr_x + prev_x
			prev_x, p, m = self.nxn_convs[os_key].profile(x=prev_x)  # module_profile(module=self.nxn_convs[os_key], x=prev_x)
			params += p
			macs += m

			fpn_out_dict[os_key] = prev_x

		return fpn_out_dict, params, macs


@Entry.register_entry(Entry.Layer)
class SSDHead(BaseLayer):
	"""
	This class defines the `SSD object detection Head <https://arxiv.org/abs/1512.02325>`_

	Args:
		in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
		n_anchors (int): Number of anchors
		n_classes (int): Number of classes in the dataset
		n_coordinates (Optional[int]): Number of coordinates. Default: 4 (x, y, w, h)
		proj_channels (Optional[int]): Number of projected channels. If `-1`, then projection layer is not used
		kernel_size (Optional[int]): Kernel size in convolutional layer. If kernel_size=1, then standard
			point-wise convolution is used. Otherwise, separable convolution is used
		step (Optional[int]): stride for feature map. If stride > 1, then feature map is sampled at this rate
			and predictions made on fewer pixels as compared to the input tensor. Default: 1
	"""

	__slots__ = ["in_channels", "n_anchors", "n_classes", "n_coordinates", "proj_channels", "kernel_size", "step"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1280, 6, 1000, 4, -1, 3, 1]
	_types_ = [int, int, int, int, int, int, int]

	def __init__(
			self,
			in_channels: int = None,
			n_anchors: int = None,
			n_classes: int = None,
			n_coordinates: int = None,
			proj_channels: int = None,
			kernel_size: int = None,
			step: int = None,
	):
		super().__init__(**Par.purify(locals()))
		if proj_channels != -1 and proj_channels != in_channels and kernel_size > 1:
			self.proj_layer = ConvLayer2D(
				in_channels=in_channels,
				out_channels=proj_channels,
				kernel_size=1,
				stride=1,
				groups=1,
				add_bias=False,
				use_norm=True,
				use_act=True,
			)
			in_channels = proj_channels

		conv_fn: Type[Union[ConvLayer2D, SeparableConv]] = ConvLayer2D if kernel_size == 1 else SeparableConv
		if kernel_size > 1 and step > 1:
			kernel_size = max(kernel_size, step if step % 2 != 0 else step + 1)
		self.loc_cls_layer = conv_fn(
			in_channels=in_channels,
			out_channels=n_anchors * (n_coordinates + n_classes),
			kernel_size=kernel_size,
			stride=1,
			groups=1,
			bias=True,
			use_norm=False,
			use_act=False,
		)

		self.kernel_size = kernel_size

		from layer.utils.init_params_util import InitParaUtil
		for layer in self.modules():
			if isinstance(layer, nn.Conv2d):
				InitParaUtil.initialize_conv_layer(module=layer, init_method="xavier_uniform")

	def _sample_fm(self, x: Tensor) -> Tensor:
		height, width = x.shape[-2:]
		device = x.device
		start_step = max(0, self.step // 2)
		indices_h = torch.arange(
			start=start_step,
			end=height,
			step=self.step,
			dtype=torch.int64,
			device=device,
		)
		indices_w = torch.arange(
			start=start_step,
			end=width,
			step=self.step,
			dtype=torch.int64,
			device=device,
		)

		x_sampled = torch.index_select(x, dim=-1, index=indices_w)
		x_sampled = torch.index_select(x_sampled, dim=-2, index=indices_h)
		return x_sampled

	def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		batch_size = x.shape[0]

		if self.proj_layer is not None:
			x = self.proj_layer(x)

		# [B x C x H x W] --> [B x Anchors * (coordinates + classes) x H x W]
		x = self.loc_cls_layer(x)

		if self.step > 1:
			x = self._sample_fm(x)

		# [B x Anchors * (coordinates + classes) x H x W] --> [B x H x W x Anchors * (coordinates + classes)]
		x = x.permute(0, 2, 3, 1)
		# [B x H x W x Anchors * (coordinates + classes)] --> [B x H*W*Anchors x (coordinates + classes)]
		x = x.contiguous().view(batch_size, -1, self.n_coordinates + self.n_classes)

		# [B x H*W*Anchors x (coordinates + classes)] --> [B x H*W*Anchors x coordinates], [B x H*W*Anchors x classes]
		box_locations, box_classes = torch.split(
			x, [self.n_coordinates, self.n_classes], dim=-1
		)
		return box_locations, box_classes

	def profile(self, x: Tensor):
		params = macs = 0.0

		if self.proj_layer is not None:
			x, p, m = self.proj_layer.profile(x)
			params += p
			macs += m

		x, p, m = self.loc_cls_layer.profile(x)
		params += p
		macs += m

		return x, params, macs
