from model.classification import BaseEncoder
from layer.utils import InitParaUtil
from layer import ExtensionLayer

import math
from utils.math_utils import Math
from utils.entry_utils import Entry
from utils.type_utils import Par
from torch import nn

from layer import ConvLayer2D, Identity, GlobalPool, Dropout, LinearLayer

from typing import Union, Tuple, Any


@Entry.register_entry(Entry.Layer)
class SeparableConv(ExtensionLayer):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size", "stride", "dilation",
		"add_bias", "padding_mode",
		"use_norm", "norm_layer", "use_act", "act_layer", "force_map_channels"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 1, False, "zeros", True, None, True, None, True]

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Union[int, Tuple[int, ...]] = 3,
			stride: Union[int, Tuple[int, ...]] = 1,
			dilation: Union[int, Tuple[int, ...]] = 1,
			add_bias: bool = False,
			padding_mode: str = "zeros",
			use_norm: bool = True,
			norm_layer: Union[str, type, object] = None,
			use_act: bool = True,
			act_layer: Union[str, type, object] = None,
			force_map_channels: bool = True
	):
		super().__init__(**Par.purify(locals()))

		self.block = nn.Sequential()

		self.dw_conv = None
		self.pw_conv = None

		self.dw_conv = ConvLayer2D(
			in_channels=in_channels,
			out_channels=in_channels,
			kernel_size=kernel_size,
			stride=stride,
			dilation=dilation,
			add_bias=False,
			groups=in_channels,
			padding_mode=padding_mode,
			use_norm=True,
			norm_layer=norm_layer,
			use_act=False,
		)
		self.block.add_module("dw_conv", self.dw_conv)

		if in_channels != out_channels or force_map_channels:
			self.pw_conv = ConvLayer2D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=1,
				dilation=1,
				groups=1,
				add_bias=add_bias,
				padding_mode=padding_mode,
				use_norm=use_norm,
				norm_layer=norm_layer,
				use_act=use_act,
				act_layer=act_layer,
			)
			self.block.add_module("pw_conv", self.pw_conv)

	def forward(self, x):
		x = self.dw_conv(x)
		if self.pw_conv:
			x = self.pw_conv(x)
		return x


def get_configuration(width_mult):

	def scale_channels(in_channels):
		return Math.make_divisible(int(math.ceil(in_channels * width_mult)), 16)

	config = {
		# 224x224
		"conv1_out": scale_channels(32),
		# 112x112
		"layer1": {
			"out_channels": scale_channels(64),
			"stride": 1,
			"repeat": 1
		},
		"layer2": {
			"out_channels": scale_channels(128),
			"stride": 2,
			"repeat": 1,
		},
		# 56x56
		"layer3": {
			"out_channels": scale_channels(256),
			"stride": 2,
			"repeat": 1,
		},
		# 28x28
		"layer4": {
			"out_channels": scale_channels(512),
			"stride": 2,
			"repeat": 5,
		},
		# 14x14
		"layer5": {
			"out_channels": scale_channels(1024),
			"stride": 2,
			"repeat": 1,
		},
		# 7x7
	}
	return config


@Entry.register_entry(Entry.ClassificationModel)
class MobileNetV1(BaseEncoder):
	__slots__ = ["width_multiplier", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1.0, ]
	_types_ = [float, ]

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

		image_channels = 3
		num_classes = self.n_classes
		classifier_dropout = Math.bound_fn(min_val=0.0, max_val=0.1, value=round(0.1 * self.width_multiplier, 3))

		cfg = get_configuration(self.width_multiplier)

		input_channels = cfg["conv1_out"]
		self.head = ConvLayer2D(
			in_channels=image_channels,
			out_channels=input_channels,
			kernel_size=3,
			stride=2,
			use_norm=True,
			use_act=True,
		)
		self.model_conf_dict["head"] = {"in": image_channels, "out": input_channels}

		self.body = nn.Sequential()

		layer_1, out_channels = self._make_layer(
			mv1_config=cfg["layer1"], input_channel=input_channels
		)
		self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer1", module=layer_1)

		layer_2, out_channels = self._make_layer(
			mv1_config=cfg["layer2"], input_channel=input_channels
		)
		self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer2", module=layer_2)

		layer_3, out_channels = self._make_layer(
			mv1_config=cfg["layer3"], input_channel=input_channels
		)
		self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer3", module=layer_3)

		layer_4, out_channels = self._make_layer(
			mv1_config=cfg["layer4"],
			input_channel=input_channels,
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer4", module=layer_4)

		layer_5, out_channels = self._make_layer(
			mv1_config=cfg["layer5"],
			input_channel=input_channels,
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer5", module=layer_5)

		self.tail = Identity()
		self.model_conf_dict["tail"] = {
			"in": input_channels,
			"out": input_channels,
		}

		pool_type = self.global_pool
		self.classifier = nn.Sequential()
		self.classifier.add_module(
			name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
		)
		if 0.0 < classifier_dropout < 1.0:
			self.classifier.add_module(
				name="classifier_dropout", module=Dropout(p=classifier_dropout)
			)
		self.classifier.add_module(
			name="classifier_fc",
			module=LinearLayer(
				in_features=input_channels, out_features=num_classes, add_bias=True
			),
		)
		self.model_conf_dict["cls"] = {"in": input_channels, "out": num_classes}

		# weight initialization
		InitParaUtil.initialize_weights(self)

	def _make_layer(self, mv1_config, input_channel, dilate=False):
		prev_dilation = self.dilation
		mv1_block = nn.Sequential()

		out_channels = mv1_config.get("out_channels")
		stride = mv1_config.get("stride", 1)
		n_repeat = mv1_config.get("repeat", 0)

		if stride == 2:
			if dilate:
				self.dilation *= stride
				stride = 1

			mv1_block.add_module(
				name="mv1_block_head",
				module=SeparableConv(
					in_channels=input_channel,
					out_channels=out_channels,
					kernel_size=3,
					stride=stride,
					use_norm=True,
					use_act=True,
					dilation=prev_dilation,
				),
			)
			input_channel = out_channels

		for i in range(n_repeat):
			mv1_block.add_module(
				name=f"mv1_block_{i}",
				module=SeparableConv(
					in_channels=input_channel,
					out_channels=out_channels,
					kernel_size=3,
					stride=1,
					use_norm=True,
					use_act=True,
					dilation=self.dilation,
				),
			)
			input_channel = out_channels

		return mv1_block, input_channel

	def __repr__(self):
		return self._repr_by_line() + "\n" + nn.Module.__repr__(self)
