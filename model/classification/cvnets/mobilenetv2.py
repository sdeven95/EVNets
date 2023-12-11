from model.classification import BaseEncoder
from utils.math_utils import Math
from utils.entry_utils import Entry
from layer.utils import InitParaUtil

from layer import ConvLayer2D, LinearLayer, GlobalPool, Dropout
from layer import ExtensionLayer

from torch import nn
from utils.type_utils import Par

from typing import Union, Tuple


@Entry.register_entry(Entry.Layer)
class InvertedResidual(ExtensionLayer):
	__slots__ = [
		"in_channels", "out_channels", "expand_ratio", "stride", "dilation",
		"skip_connection", "norm_layer", "act_layer"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 1, True, None, None]

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			expand_ratio: float,
			stride: Union[int, Tuple[int, ...]] = 1,
			dilation: Union[int, Tuple[int, ...]] = 1,
			skip_connection: bool = True,
			norm_layer: str = None,
			act_layer: str = None
	):
		super().__init__(**Par.purify(locals()))

		assert self.stride in [1, 2], "Stride must be 1 or 2 for InvertedResidual Module"
		self.hidden_dim = Math.make_divisible(int(round(in_channels * expand_ratio)), 8)

		self.block = nn.Sequential()
		if self.expand_ratio != 1:
			self.block.add_module(
				name="exp_1x1",
				module=ConvLayer2D(
					in_channels=in_channels,
					out_channels=self.hidden_dim,
					kernel_size=1,
					use_norm=True,
					norm_layer=norm_layer,
					use_act=True,
					act_layer=act_layer
				)
			)
		self.block.add_module(
			name="conv_3x3",
			module=ConvLayer2D(
				in_channels=self.hidden_dim,
				out_channels=self.hidden_dim,
				kernel_size=3,
				stride=stride,
				dilation=dilation,
				groups=self.hidden_dim,
				use_norm=True,
				norm_layer=norm_layer,
				use_act=True,
				act_layer=act_layer
			)
		)
		self.block.add_module(
			name="red_1x1",
			module=ConvLayer2D(
				in_channels=self.hidden_dim,
				out_channels=out_channels,
				kernel_size=1,
				use_norm=True,
				norm_layer=norm_layer,
				use_act=False,
			)
		)

		self.use_res_connect = stride == 1 and in_channels == out_channels and skip_connection

	def forward(self, x):
		return x + self.block(x) if self.use_res_connect else self.block(x)


def get_configuration():
	mobilenetv2_config = {
		# 112x112
		"layer1": {
			"expansion_ratio": 1,
			"out_channels": 16,
			"num_blocks": 1,
			"stride": 1,
		},
		"layer2": {
			"expansion_ratio": 6,
			"out_channels": 24,
			"num_blocks": 2,
			"stride": 2,
		},
		# 56x56
		"layer3": {
			"expansion_ratio": 6,
			"out_channels": 32,
			"num_blocks": 3,
			"stride": 2,
		},
		# 28x28
		"layer4": {
			"expansion_ratio": 6,
			"out_channels": 64,
			"num_blocks": 4,
			"stride": 2,
		},
		# 14x14
		"layer4_a": {
			"expansion_ratio": 6,
			"out_channels": 96,
			"num_blocks": 3,
			"stride": 1,
		},
		# 14x14
		"layer5": {
			"expansion_ratio": 6,
			"out_channels": 160,
			"num_blocks": 3,
			"stride": 2,
		},
		# 7x7
		"layer5_a": {
			"expansion_ratio": 6,
			"out_channels": 320,
			"num_blocks": 1,
			"stride": 1,
		},
		# 7x7
	}
	return mobilenetv2_config


@Entry.register_entry(Entry.ClassificationModel)
class MobileNetV2(BaseEncoder):
	"""
	This class defines the `MobileNetv2 architecture <https://arxiv.org/abs/1801.04381>`_
	"""
	__slots__ = ["width_multiplier", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1.0, ]
	_types_ = [float, ]

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

		width_mult = self.width_multiplier
		num_classes = self.n_classes

		cfg = get_configuration()

		image_channels = 3
		input_channels = 32
		last_channel = 1280
		classifier_dropout = self.classifier_dropout
		if classifier_dropout == 0.0 or classifier_dropout is None:
			val = round(0.2 * width_mult, 3)
			classifier_dropout = Math.bound_fn(min_val=0.0, max_val=0.2, value=val)

		last_channel = Math.make_divisible(
			last_channel * max(1.0, width_mult), self.round_nearest
		)

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
			mv2_config=cfg["layer1"],
			width_mult=width_mult,
			input_channel=input_channels,
		)
		self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer1", module=layer_1)

		layer_2, out_channels = self._make_layer(
			mv2_config=cfg["layer2"],
			width_mult=width_mult,
			input_channel=input_channels,
		)
		self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer2", module=layer_2)

		layer_3, out_channels = self._make_layer(
			mv2_config=cfg["layer3"],
			width_mult=width_mult,
			input_channel=input_channels,
		)
		self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer3", module=layer_3)

		layer_4, out_channels = self._make_layer(
			mv2_config=[cfg["layer4"], cfg["layer4_a"]],
			width_mult=width_mult,
			input_channel=input_channels,
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer4", module=layer_4)

		layer_5, out_channels = self._make_layer(
			mv2_config=[cfg["layer5"], cfg["layer5_a"]],
			width_mult=width_mult,
			input_channel=input_channels,
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer5", module=layer_5)

		self.tail = ConvLayer2D(
			in_channels=input_channels,
			out_channels=last_channel,
			kernel_size=1,
			stride=1,
			use_act=True,
			use_norm=True,
		)
		self.model_conf_dict["tail"] = {
			"in": input_channels,
			"out": last_channel,
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
				in_features=last_channel, out_features=num_classes, add_bias=True
			),
		)

		self.model_conf_dict["cls"] = {"in": last_channel, "out": num_classes}

		# weight initialization
		InitParaUtil.initialize_weights(self)

	def _make_layer(self, mv2_config, width_mult, input_channel, dilate=False):
		prev_dilation = self.dilation
		mv2_block = nn.Sequential()
		count = 0

		if isinstance(mv2_config, dict):
			mv2_config = [mv2_config]

		for cfg in mv2_config:
			t = cfg.get("expansion_ratio")
			c = cfg.get("out_channels")
			n = cfg.get("num_blocks")
			s = cfg.get("stride")

			output_channel = Math.make_divisible(c * width_mult, self.round_nearest)

			for block_idx in range(n):
				stride = s if block_idx == 0 else 1
				block_name = "mv2_block_{}".format(count)
				if dilate and count == 0:
					self.dilation *= stride
					stride = 1

				layer = InvertedResidual(
					in_channels=input_channel,
					out_channels=output_channel,
					stride=stride,
					expand_ratio=t,
					dilation=prev_dilation if count == 0 else self.dilation,
				)
				mv2_block.add_module(name=block_name, module=layer)
				count += 1
				input_channel = output_channel
		return mv2_block, input_channel

	def __repr__(self):
		return self._repr_by_line() + "\n" + nn.Module.__repr__(self)
