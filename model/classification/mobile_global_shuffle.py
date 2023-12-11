from data.transform.builtins import Normalize, RandomCropResize, AutoAugment, RandAugment, ColorJitter, RandomErasing
from data.transform.extensions import RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Compose
from data.transform.torch import RandomMixup, RandomCutmix
from data.sampler.batch_sampler import BatchSampler, BatchSamplerDDP
from data.dataset.classification.food import FoodDataset

from layer.activation.builtins import ReLU, ReLU6
from layer.normalization.builtins import BatchNorm2d

from loss.extensions import CrossEntropy
from optim.builtins import SGD
from scheduler.extensions import Cosine

from metrics.stats import Statistics

import torch
from torch import Tensor, nn

from layer import ConvLayer2D, LinearLayer, GlobalPool, Dropout
from utils.entry_utils import Entry
from utils.type_utils import Par, Dct
from utils.math_utils import Math
from . import BaseEncoder

from model.classification.cvnets.mobilenetv1 import SeparableConv
from model.classification.cvnets.mobilenetv2 import InvertedResidual
from model.classification.cvnets.mobilenetv3 import InvertedResidualSE

from typing import Optional, Union, Tuple, Dict


def get_configuration() -> Dict:

	config = {
		# an example
		"example_layer": [
			{
				"out_channels": 64,
				"kernel_size": 3,
				"stride": 1,
				"loc0": {"block": "v0", "groups": 1},
				"loc1": {"block": "v1", "force_map_channels": True},
				"loc2": {"block": "v2", "expand_ratio": 1.0},
				'gbl': {"block": "v3", "expand_ratio": 1.0, "use_hs": False, "use_se": False, "h_groups": 16, "w_groups": 16},
				'agt': {"type": "channel", "size": (3, 128, 128), "use_linear": False,
						"bias": False, "use_norm": True, "use_act": True, "act_name": "relu",
						"use_dropout": False, "dropout": 0.0},
			},
			...
		],

		"image_channels": 3,
		"head_out_channels": 16,
		"last_channels": 1280,

		# 128x128
		"layer1": [
			{
				"kernel_size": 3,
				"out_channels": 16,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 1},
				"gbl": {"block": "v2", "expand_ratio": 1, "h_groups": 16, "w_groups": 16},
			},
		],

		# 128x128
		"layer2": [
			{
				"kernel_size": 3,
				"out_channels": 24,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 4},
				"gbl": {"block": "v2", "expand_ratio": 4, "h_groups": 16, "w_groups": 16},
			},
			# 64x64
			{
				"kernel_size": 3,
				"out_channels": 24,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 8, "w_groups": 8},
			},
		],

		# 64x64
		"layer3": [
			{
				"kernel_size": 3,
				"out_channels": 40,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 8, "w_groups": 8},
			},
			# 32x32
			{
				"kernel_size": 3,
				"out_channels": 40,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 4, "w_groups": 4},
			},
			# 32x32
			{
				"kernel_size": 3,
				"out_channels": 40,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 4, "w_groups": 4},
			},
		],

		# 32x32
		"layer4": [
			{
				"kernel_size": 3,
				"out_channels": 80,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 6},
				"gbl": {"block": "v2", "expand_ratio": 6, "h_groups": 4, "w_groups": 4},
			},
			# 16x16
			{
				"kernel_size": 3,
				"out_channels": 80,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 2.5},
				"gbl": {"block": "v2", "expand_ratio": 2.5, "h_groups": 4, "w_groups": 4},
			},
			# 16x16
			{
				"kernel_size": 3,
				"out_channels": 80,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 2.3},
				"gbl": {"block": "v2", "expand_ratio": 2.3, "h_groups": 4, "w_groups": 4},
			},

		],

		# 16x16
		"layer5": [
			{
				"kernel_size": 3,
				"out_channels": 160,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 6},
				"gbl": {"block": "v2", "expand_ratio": 6, "h_groups": 4, "w_groups": 4},
			},

		],
	}
	return config


@Entry.register_entry(Entry.Layer)
class ShuffleConv(ConvLayer2D):
	__slots__ = ["h_groups", "w_groups"]
	_disp_ = __slots__

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			h_groups: int,
			w_groups: int,
			kernel_size: Optional[Union[int, Tuple[int, int]]] = 1,
			stride: Optional[Union[int, Tuple[int, int]]] = 1,
			dilation: Optional[Union[int, Tuple[int, int]]] = 1,
			groups: Optional[int] = 1,
			add_bias: Optional[bool] = False,
			padding_mode: Optional[str] = "zeros",
			use_norm: Optional[bool] = True,
			norm_layer: Optional[Union[str, type, object]] = None,
			use_act: Optional[bool] = True,
			act_layer: Optional[Union[str, type, object]] = None,
	):
		self.h_groups = h_groups
		self.w_groups = w_groups

		super().__init__(**Dct.pop_keys(Par.purify(locals()), ["h_groups", "w_groups"]))

	def forward(self, x: Tensor) -> Tensor:
		y = x
		crop_width = x.size(-1)
		crop_height = x.size(-2)
		if self.h_groups > 1 and self.w_groups > 1:
			h_index = self.shuffle_index(crop_height, self.h_groups)
			w_index = self.shuffle_index(crop_width, self.w_groups)
			y = x[:, :, h_index, :][:, :, :, w_index]
		elif self.h_groups > 1:
			h_index = self.shuffle_index(crop_height, self.h_groups)
			y = x[:, :, h_index, :]
		elif self.w_groups > 1:
			w_index = self.shuffle_index(crop_width, self.w_groups)
			y = x[:, :, :, w_index]

		return self.block(y)

	@staticmethod
	def shuffle_index(len: int, groups: int):
		if len // groups < 2:
			return torch.tensor([i for i in range(len)])
		elif len % groups == 0:
			return torch.tensor([i for i in range(len)]).view(groups, -1).t().flatten(0, -1)
		elif len % groups % 2 == 0:
			return torch.hstack([
				torch.tensor([i for i in range(len % groups // 2)]),
				torch.tensor([i for i in range(len % groups // 2, len - len % groups // 2)]).view(groups, -1).t().flatten(0, -1),
				torch.tensor([i for i in range(len - len % groups // 2, len)])
			])
		else:
			return torch.hstack([
				torch.tensor([i for i in range(len % groups // 2)]),
				torch.tensor([i for i in range(len % groups // 2, len - len % groups // 2 - 1)]).view(groups, -1).t().flatten(0, -1),
				torch.tensor([i for i in range(len - len % groups // 2 - 1, len)])
			])


@Entry.register_entry(Entry.Layer)
class ShuffleSeparableConv(SeparableConv):
	__slots__ = ["h_groups", "w_groups"]
	_disp_ = __slots__

	def __init__(
		self,
		h_groups: int,
		w_groups: int,
		in_channels: int,
		out_channels: int,
		kernel_size: Optional[Union[int, Tuple[int, int]]] = 3,
		stride: Optional[Union[int, Tuple[int, int]]] = 1,
		dilation: Optional[Union[int, Tuple[int, int]]] = 1,
		add_bias: Optional[bool] = False,
		padding_mode: Optional[str] = "zeros",
		use_norm: Optional[bool] = True,
		norm_layer: Optional[Union[str, type, object]] = None,
		use_act: Optional[bool] = True,
		act_layer: Optional[Union[str, type, object]] = None,
		force_map_channels: Optional[bool] = True
	) -> None:
		self.h_groups = h_groups
		self.w_groups = w_groups

		super().__init__(**Dct.pop_keys(Par.purify(locals()), ["h_groups", "w_groups"]))

		self.dw_conv = ShuffleConv(
			in_channels=in_channels,
			out_channels=in_channels,
			h_groups=h_groups,
			w_groups=w_groups,
			kernel_size=kernel_size,
			stride=stride,
			dilation=dilation,
			groups=in_channels,
			add_bias=False,
			padding_mode=padding_mode,
			use_norm=True,
			use_act=False,
		)


@Entry.register_entry(Entry.Layer)
class ShuffleInvertedResidual(InvertedResidual):
	__slots__ = ["h_groups", "w_groups"]
	_disp_ = __slots__

	def __init__(
		self,
		h_groups: int,
		w_groups: int,
		in_channels: int,
		out_channels: int,
		stride: int,
		expand_ratio: Union[int, float],
		dilation: int = 1,
		skip_connection: Optional[bool] = True
	) -> None:
		self.h_groups = h_groups
		self.w_groups = w_groups

		super().__init__(**Dct.pop_keys(Par.purify(locals()), ["h_groups", "w_groups"]))

		tar_idx = 0 if expand_ratio == 1 else 1

		self.block[tar_idx] = ShuffleConv(
			in_channels=self.hidden_dim,
			out_channels=self.hidden_dim,
			h_groups=h_groups,
			w_groups=w_groups,
			kernel_size=3,
			stride=stride,
			dilation=dilation,
			groups=self.hidden_dim,
			use_act=True,
			use_norm=True
		)


@Entry.register_entry(Entry.Layer)
class ShuffleInvertedResidualSE(InvertedResidualSE):
	def __init__(
			self,
			h_groups: int,
			w_groups: int,
			in_channels: int,
			out_channels: int,
			expand_ratio: int,
			kernel_size: Union[int, Tuple[int, ...]] = 3,
			stride: Union[int, Tuple[int, ...]] = 1,
			dilation: Union[int, Tuple[int, ...]] = 1,
			use_hs: bool = False,
			use_se: bool = False,
			norm_layer: str = None,
			act_layer: str = None
	):
		self.h_groups = h_groups
		self.w_groups = w_groups

		super().__init__(**Dct.pop_keys(Par.purify(locals()), ["h_groups", "w_groups"]))

		tar_idx = 0 if expand_ratio == 1 else 2

		self.block[tar_idx] = ShuffleConv(
			in_channels=self.hidden_dim,
			out_channels=self.hidden_dim,
			h_groups=h_groups,
			w_groups=w_groups,
			kernel_size=kernel_size,
			stride=stride,
			dilation=dilation,
			groups=self.hidden_dim,
			use_norm=True,
			use_act=False
		)


@Entry.register_entry(Entry.ClassificationModel)
class MobileShuffleV2(BaseEncoder):
	__slots__ = ["width_multiplier"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0]
	_types_ = [float]

	configure_file = "config/classification/food172/mobile_shuffle.yaml"

	def __init__(self) -> None:
		super().__init__()

		if self.classifier_dropout == 0.0 or self.classifier_dropout is None:
			val = round(0.2 * self.width_multiplier, 3)
			self.classifier_dropout = Math.bound_fn(min_val=0.0, max_val=0.2, value=val)

		# get net structure configuration
		self.cfg = get_configuration()

		# create configuration dict
		self.model_conf_dict: dict = dict()

		# build net
		self._make_head_layer()
		self._make_body_layers()
		self._make_tail_layers()
		self._make_classifier_layer()

	def _make_head_layer(self):
		image_channels = self.cfg.get("image_channels", 3)
		self.cur_channels = self.cfg.get("head_out_channels", 16)

		self.head = ConvLayer2D(
			in_channels=image_channels,
			out_channels=self.cur_channels,
			kernel_size=3,
			stride=2,
			use_norm=True,
			use_act=True,
		)

		self.model_conf_dict["head"] = {"in": image_channels, "out": self.cur_channels}

	def _make_body_layers(self):
		self.layer_1, out_channels = self._make_layer(
			configure=self.cfg["layer1"],
			input_channel=self.cur_channels,
		)
		self.model_conf_dict["layer1"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_2, out_channels = self._make_layer(
			configure=self.cfg["layer2"],
			input_channel=self.cur_channels,
		)
		self.model_conf_dict["layer2"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_3, out_channels = self._make_layer(
			configure=self.cfg["layer3"],
			input_channel=self.cur_channels,
		)
		self.model_conf_dict["layer3"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_4, out_channels = self._make_layer(
			configure=self.cfg["layer4"],
			input_channel=self.cur_channels,
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_5, out_channels = self._make_layer(
			configure=self.cfg["layer5"],
			input_channel=self.cur_channels,
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

	def _make_tail_layers(self):
		last_channels = self.cfg.get("last_channels", 1024)
		last_channels = Math.make_divisible(
			last_channels * max(1.0, self.width_multiplier), self.round_nearest
		)

		self.tail = ConvLayer2D(
			in_channels=self.cur_channels,
			out_channels=last_channels,
			kernel_size=1,
			stride=1,
			use_act=True,
			use_norm=True,
		)

		self.model_conf_dict["tail"] = {"in": self.cur_channels, "out": last_channels}
		self.cur_channels = last_channels

	def _make_classifier_layer(self):

		self.classifier = nn.Sequential()
		self.classifier.add_module(
			name="global_pool", module=GlobalPool(pool_type=self.global_pool, keep_dim=False)
		)

		if 0.0 < self.classifier_dropout < 1.0:
			self.classifier.add_module(
				name="classifier_dropout", module=Dropout(p=self.classifier_dropout)
			)
		self.classifier.add_module(
			name="classifier_fc",
			module=LinearLayer(
				in_features=self.cur_channels, out_features=self.n_classes, bias=True
			),
		)

		self.model_conf_dict["cls"] = {"in": self.cur_channels, "out": self.n_classes}
		self.cur_channels = self.n_classes

	def _make_layer(
		self,
		configure,
		input_channel: int,
		dilate: Optional[bool] = False,
	) -> Tuple[nn.Module, int]:
		from model.classification.mobile_aggregate import LocalGlobalAggregate

		# prev_dilation = self.dilation
		block_seq = nn.Sequential()
		count = 0

		for cfg_block in configure:
			out_channels_cfg = cfg_block.get("out_channels")
			stride = cfg_block.get("stride", 1)
			cfg_block["out_channels"] = Math.make_divisible(out_channels_cfg * self.width_multiplier, self.round_nearest)
			if dilate and count == 0:
				self.dilation *= stride
				stride = 1
				cfg_block["stride"] = 1

			block = LocalGlobalAggregate(
				in_channels=input_channel,
				cfg=cfg_block,
			)
			block_seq.add_module(
				name=f"ms_s_{stride}_idx-{count}",
				module=block
			)
			count += 1
			input_channel = cfg_block["out_channels"]

		return block_seq, input_channel
