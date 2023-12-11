from . import BaseLayer, ExtensionLayer

from torch import nn, Tensor
import torch
import torch.nn.functional as F

from utils.type_utils import Par, Prf
from utils.entry_utils import Entry

from .extensions import ConvLayer2D
from .builtins import Dropout2D, UpSample, AdaptiveAvgPool2d
from .utils import LayerUtil
from layer.utils.init_params_util import InitParaUtil
from model.classification.cvnets.mobilenetv1 import SeparableConv


from typing import Dict, Tuple, Optional


class BaseSegmentationHead(BaseLayer):
	"""
	Base class for segmentation heads
	"""

	def __init__(self, seg_model):
		super().__init__()

		self.seg_model = seg_model
		enc_conf = self.seg_model.encoder.model_conf_dict

		if self.seg_model.use_tail:
			self.enc_tail_out = enc_conf["tail"]["out"]
		self.enc_l5_out = enc_conf["layer5"]["out"]
		self.enc_l4_out = enc_conf["layer4"]["out"]
		self.enc_l3_out = enc_conf["layer3"]["out"]
		self.enc_l2_out = enc_conf["layer2"]["out"]
		self.enc_l1_out = enc_conf["layer2"]["out"]

		self.aux_head = None
		if self.seg_model.use_aux_head:
			inner_channels = max(int(self.enc_l4_channels // 4), 128)
			self.aux_head = nn.Sequential(
				ConvLayer2D(
					in_channels=self.enc_l4_channels,
					out_channels=inner_channels,
					kernel_size=3,
					stride=1,
					use_norm=True,
					use_act=True,
					bias=False,
					groups=1,
				),
				Dropout2D(self.aux_dropout),
				ConvLayer2D(
					in_channels=inner_channels,
					out_channels=self.n_classes,
					kernel_size=1,
					stride=1,
					use_norm=False,
					use_act=False,
					bias=True,
					groups=1,
				),
			)

			self.upsample_seg_out = None
			if self.seg_model.output_stride > 1:
				self.upsample_seg_out = UpSample(
					scale_factor=self.seg_model.output_stride,
					mode="bilinear",
					align_corners=True,
				)

	def forward_aux_head(self, enc_out: Dict) -> Tensor:
		aux_out = self.aux_head(enc_out["out_l4"])
		return aux_out

	def forward_seg_head(self, enc_out: Dict) -> Tensor:
		raise NotImplementedError

	def forward(self, enc_out: Dict, **kwargs) -> Tensor or Tuple[Tensor, Tensor]:
		out = self.forward_seg_head(enc_out=enc_out)

		if self.upsample_seg_out is not None:
			# resize the mask based on given size
			mask_size = kwargs.get("orig_size", None)
			if mask_size is not None:
				self.upsample_seg_out.scale_factor = None
				self.upsample_seg_out.size = mask_size

			out = self.upsample_seg_out(out)

		if self.aux_head is not None and self.training:
			aux_out = self.forward_aux_head(enc_out=enc_out)
			return out, aux_out
		return out

	def get_trainable_parameters(
		self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False
	):
		param_list = LayerUtil.split_parameters(
			named_parameters=self.named_parameters,
			weight_decay=weight_decay,
			no_decay_bn_filter_bias=no_decay_bn_filter_bias,
		)
		return param_list, [self.lr_multiplier] * len(param_list)


@Entry.register_entry(Entry.Layer)
class DeeplabV3(BaseSegmentationHead):
	__slots__ = ["atrous_rates", "out_channels", "is_sep_conv", "dropout"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [[6, 12, 18], 256, False, 0.1]
	_types_ = [(int, ), int, bool, float]

	def __init__(self, seg_model):
		super().__init__(seg_model=seg_model)

		self.aspp = nn.Sequential()
		aspp_in_channels = (
			self.enc_l5_out if not self.use_tail else self.enc_tail_out
		)
		self.aspp.add_module(
			name="aspp_layer",
			module=ASPP(
				in_channels=aspp_in_channels,
				out_channels=self.out_channels,
				atrous_rates=self.atrous_rates,
				is_sep_conv=self.is_sep_conv,
				dropout=self.dropout,
			),
		)

		self.classifier = ConvLayer2D(
			in_channels=self.out_channels,
			out_channels=self.n_classes,
			kernel_size=1,
			stride=1,
			use_norm=False,
			use_act=False,
			add_bias=True,
		)

		InitParaUtil.initialize_weights(self)

	def forward_seg_head(self, enc_out: Dict) -> Tensor:
		# low resolution features
		x = enc_out["out_tail"] if self.tail else enc_out["out_layer5"]
		# ASPP featues
		x = self.aspp(x)
		# classify
		x = self.classifier(x)
		return x

	def profile(self, enc_out: Dict) -> Tuple[Tensor, float, float]:
		# Note: Model profiling is for reference only and may contain errors.
		# It relies heavily on the user to implement the underlying functions accurately.

		params, macs = 0.0, 0.0

		if self.use_tail:
			x, p, m = Prf.profile_list(module_list=self.aspp, x=enc_out["out_tail"])
		else:
			x, p, m = Prf.profile_list(module_list=self.aspp, x=enc_out["out_layer5"])
		params += p
		macs += m

		out, p, m = Prf.profile_list(module_list=self.classifier, x=x)
		params += p
		macs += m

		print(
			"{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
				self.__class__.__name__,
				"Params",
				round(params / 1e6, 3),
				"MACs",
				round(macs / 1e6, 3),
			)
		)
		return out, params, macs


@Entry.register_entry(Entry.Layer)
class PSPNet(BaseSegmentationHead):
	__slots__ = ["out_channels", "pool_sizes", "dropout"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [512, [1, 2, 3, 6], 0.1]
	_types_ = [int, (int, ), float]

	def __init__(self, seg_model):
		super().__init__(seg_model=seg_model)

		psp_in_channels = (
			self.enc_l5_out if not self.use_tail else self.enc_tail_out
		)
		self.psp_layer = PSP(
			in_channels=psp_in_channels,
			out_channels=self.out_channels,
			pool_sizes=self.pool_sizes,
			dropout=self.dropout,
		)
		self.classifier = ConvLayer2D(
			in_channels=self.out_channels,
			out_channels=self.n_classes,
			kernel_size=1,
			stride=1,
			use_norm=False,
			use_act=False,
			bias=True,
		)

		InitParaUtil.initialize_weights(self)

	def forward_seg_head(self, enc_out: Dict) -> Tensor:
		# low resolution features
		x = enc_out["out_tail"] if self.use_tail else enc_out["out_layer5"]

		# Apply PSP layer
		x = self.psp_layer(x)

		out = self.classifier(x)

		return out

	def profile(self, enc_out: Dict) -> Tuple[Tensor, float, float]:
		# Note: Model profiling is for reference only and may contain errors.
		# It relies heavily on the user to implement the underlying functions accurately.

		params, macs = 0.0, 0.0

		if self.use_tail:
			x, p, m = Prf.profile_list(module_list=self.psp_layer, x=enc_out["out_tail"])
		else:
			x, p, m = Prf.profile_list(module_list=self.psp_layer, x=enc_out["out_layer5"])
		params += p
		macs += m

		out, p, m = Prf.profile_list(module_list=self.classifier, x=x)
		params += p
		macs += m

		print(
			"{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
				self.__class__.__name__,
				"Params",
				round(params / 1e6, 3),
				"MACs",
				round(macs / 1e6, 3),
			)
		)
		return out, params, macs


@Entry.register_entry(Entry.Layer)
class ASPP(BaseLayer):
	"""
	ASPP module defined in DeepLab papers, `here <https://arxiv.org/abs/1606.00915>`_ and `here <https://arxiv.org/abs/1706.05587>`_

	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
		atrous_rates (Tuple[int]): atrous rates for different branches.
		is_sep_conv (Optional[bool]): Use separable convolution instead of standaard conv. Default: False
		dropout (Optional[float]): Apply dropout. Default is 0.0

	Shape:
		- Input: :math:`(N, C_{in}, H, W)`
		- Output: :math:`(N, C_{out}, H, W)`
	"""
	__slots__ = ["in_channels", "out_channels", "atrous_rates", "is_sep_conv", "dropout"]
	_disp_ = __slots__
	# _defaults_ = [None, None, None, False, 0.0]

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			atrous_rates: Tuple[int],
			is_sep_conv: bool = False,
			dropout: float = 0.0,
	):
		super().__init__(**Par.purify(locals()))

		# setup modules, 1 general convolution, 3 dilated convolutions, 1 global pooling
		modules = [
			ConvLayer2D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=1,
				use_norm=True,
				use_act=True,
			)
		]
		aspp_layer = ASPPSeparableConv if is_sep_conv else ASPPConv
		assert len(atrous_rates) == 3
		modules.extend(
			[
				aspp_layer(
					in_channels=in_channels,
					out_channels=out_channels,
					dilation=rate,
				)
				for rate in atrous_rates
			]
		)
		modules.append(
			ASPPPooling(in_channels=in_channels, out_channels=out_channels)
		)
		self.convs = nn.ModuleList(modules)

		# setup project layer, concat above 5 convolutions and map to out_channels
		self.project = ConvLayer2D(
			in_channels=5 * out_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=1,
			use_norm=True,
			use_act=True,
		)

		# setup dropout layer
		self.dropout_layer = Dropout2D(p=dropout)

	def forward(self, x: Tensor) -> Tensor:
		out = []
		for conv in self.convs:
			out.append(conv(x))
		out = torch.cat(out, dim=1)
		out = self.project(out)
		out = self.dropout_layer(out)
		return out

	def profile(self, x: Tensor) -> (Tensor, float, float):
		params, macs = 0.0, 0.0
		res = []
		for c in self.convs:
			out, p, m = c.profile(x=x)
			params += p
			macs += m
			res.append(out)
		res = torch.cat(res, dim=1)

		out, p, m = self.project.profile(x=res)
		params += p
		macs += m
		return out, params, macs


@Entry.register_entry(Entry.Layer)
class ASPPConv(ConvLayer2D):
	"""
	Convolution with a dilation  for the ASPP module
	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
		dilation (int): Dilation rate

	Shape:
		- Input: :math:`(N, C_{in}, H, W)`
		- Output: :math:`(N, C_{out}, H, W)`
	"""

	def __init__(
		self, in_channels: int, out_channels: int, dilation: int,
	) -> None:
		super().__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=1,
			use_norm=True,
			use_act=True,
			dilation=dilation,
		)

	def adjust_atrous_rate(self, rate: int) -> None:
		"""This function allows to adjust the dilation rate"""
		self.block.conv_layer.dilation = rate
		# padding is the same here
		# see ConvLayer2D to see the method for computing padding
		self.block.conv_layer.padding = rate   # keep resolution unchanged


@Entry.register_entry(Entry.Layer)
class ASPPSeparableConv(SeparableConv):
	"""
	Separable Convolution with a dilation for the ASPP module
	Args:
		opts: command-line arguments
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
		dilation (int): Dilation rate

	Shape:
		- Input: :math:`(N, C_{in}, H, W)`
		- Output: :math:`(N, C_{out}, H, W)`
	"""

	def __init__(
		self, opts, in_channels: int, out_channels: int, dilation: int
	) -> None:
		super().__init__(
			opts=opts,
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=1,
			dilation=dilation,
			use_norm=True,
			use_act=True,
		)

	def adjust_atrous_rate(self, rate: int) -> None:
		"""This function allows to adjust the dilation rate"""
		self.dw_conv.layer.conv.dilation = rate
		# padding is the same here
		# see ConvLayer to see the method for computing padding
		self.dw_conv.layer.conv.padding = rate  # keep resolution unchanged


@Entry.register_entry(Entry.Layer)
class ASPPPooling(ExtensionLayer):
	"""
	ASPP pooling layer
	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`

	Shape:
		- Input: :math:`(N, C_{in}, H, W)`
		- Output: :math:`(N, C_{out}, H, W)`
	"""
	__slots__ = ["in_channels", "out_channels"]
	_disp_ = __slots__

	# defaults=[None, None]

	def __init__(
		self, in_channels: int, out_channels: int,
	) -> None:
		super().__init__(in_channels=in_channels, out_channels=out_channels)

		self.block = nn.Sequential()
		self.block.add_module(
			name="global_pool", module=AdaptiveAvgPool2d(output_size=1)
		)
		self.block.add_module(
			name="conv_1x1",
			module=ConvLayer2D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=1,
				use_norm=True,
				use_act=True,
			),
		)

	def forward(self, x: Tensor) -> Tensor:
		x_size = x.shape[-2:]
		x = self.block(x)
		x = F.interpolate(x, size=x_size, mode="bilinear", align_corners=False)  # recover size
		return x

	def profile(self, x: Tensor) -> Tuple[Tensor, float, float]:
		out, params, macs = super().profile(x=x)
		out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)  # recover size
		return out, params, macs


@Entry.register_entry(Entry.Layer)
class PSP(ExtensionLayer):
	"""
	This class defines the Pyramid Scene Parsing module in the `PSPNet paper <https://arxiv.org/abs/1612.01105>`_

	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
		pool_sizes Optional[Tuple[int, ...]]: List or Tuple of pool sizes. Default: (1, 2, 3, 6)
		dropout (Optional[float]): Apply dropout. Default is 0.0
	"""
	__slots__ = [
		"in_channels", "out_channels", "pool_sizes", "dropout",
		"reduction_dim", "channels_after_concat"
	]
	_disp_ = __slots__

	# defaults=[None, None, (1, 2, 3, 6), 0.0]

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			pool_sizes: Optional[Tuple[int, ...]] = (1, 2, 3, 6),
			dropout: Optional[float] = 0.0,
	):
		super().__init__(**Par.purify(locals()))

		reduction_dim = in_channels // len(pool_sizes)
		reduction_dim = (reduction_dim // 16) * 16
		channels_after_concat = (reduction_dim * len(pool_sizes)) + in_channels

		self.psp_branches = nn.ModuleList(
			[
				nn.Sequential(
					AdaptiveAvgPool2d(output_size=(ps, ps)),
					ConvLayer2D(
						in_channels=in_channels,
						out_channels=reduction_dim,
						kernel_size=1,
						bias=False,
						use_norm=True,
						use_act=True,
					),
				)
				for ps in pool_sizes
			]
		)

		self.fusion = nn.Sequential(
			ConvLayer2D(
				in_channels=channels_after_concat,
				out_channels=out_channels,
				kernel_size=3,
				stride=1,
				use_norm=True,
				use_act=True,
			),
			Dropout2D(p=dropout),
		)

	def forward(self, x: Tensor) -> Tensor:
		x_size = x.shape[2:]
		out = [x] + [
			F.interpolate(
				input=psp_branch(x), size=x_size, mode="bilinear", align_corners=True
			)
			for psp_branch in self.psp_branches
		]
		out = torch.cat(out, dim=1)
		out = self.fusion(out)
		return out

	def profile_module(
		self, x: Tensor
	) -> Tuple[Tensor, float, float]:
		params, macs = 0.0, 0.0
		res = [x]
		input_size = x.size()
		for psp_branch in self.psp_branches:
			out, p, m = Prf.profile_list(module_list=psp_branch, x=x)
			out = F.interpolate(
				out, input_size[2:], mode="bilinear", align_corners=True
			)
			params += p
			macs += m
			res.append(out)
		res = torch.cat(res, dim=1)

		res, p, m = Prf.profile_list(module_list=self.fusion, x=res)
		return res, params + p, macs + m
