from model.classification import BaseEncoder
from utils.math_utils import Math
from utils.entry_utils import Entry
from utils.type_utils import Par

from torch import nn

from layer import ConvLayer2D, Dropout, LinearLayer, GlobalPool, AdaptiveAvgPool2d
from layer.utils import InitParaUtil
from layer import ExtensionLayer

from typing import Union, Tuple


@Entry.register_entry(Entry.Layer)
class SqueezeExcitation(ExtensionLayer):
	__slots__ = ["in_channels", "squeeze_factor", "scale_fn_name", "act_layer"]
	_disp_ = __slots__

	def __init__(
			self,
			in_channels: int,
			squeeze_factor: int,
			scale_fn_name: str = "Sigmoid",
			act_layer: str = None,
	):
		super().__init__(**Par.purify(locals()))

		squeeze_channels = max(Math.make_divisible(in_channels // squeeze_factor, 8), 32)

		self.block = nn.Sequential()
		self.block.add_module(name="global_pool", module=AdaptiveAvgPool2d())

		self.fc1 = ConvLayer2D(
			in_channels=in_channels,
			out_channels=squeeze_channels,
			kernel_size=1,
			stride=1,
			add_bias=True,
			use_norm=False,
			use_act=True,
			act_layer=act_layer
		)
		self.block.add_module(name="fc1", module=self.fc1)

		self.fc2 = ConvLayer2D(
			in_channels=squeeze_channels,
			out_channels=in_channels,
			kernel_size=1,
			stride=1,
			add_bias=True,
			use_norm=False,
			use_act=False,
		)
		self.block.add_module(name="fc2", module=self.fc2)

		if scale_fn_name == "Sigmoid":
			act_fn = nn.Sigmoid()
		elif scale_fn_name == "Hardsigmoid":
			self.act_fn = nn.Hardswish()
		else:
			raise NotImplementedError(f"Activation layer {self.scale_fn_name} is not implemented")
		self.block.add_module(name="scale_act", module=nn.Sigmoid())

	def forward(self, x):
		return x * self.block(x)

	def profile(self, x):
		block_x, paras, macs = super().profile(x)
		x = x * self.block(x)
		b, c, h, w = x.size()
		macs += b * c * h * w
		return x, paras, macs


@Entry.register_entry(Entry.Layer)
class InvertedResidualSE(ExtensionLayer):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"expand_ratio", "stride", "dilation",
		"use_hs", "use_se", "norm_layer", "act_layer"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 3, 1, 1, False, False, None, None]

	def __init__(
			self,
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
		super().__init__(**Par.purify(locals()))

		if self.use_hs:
			act_fn = nn.Hardswish()
		else:
			act_fn = nn.ReLU()

		self.block = nn.Sequential()
		self.hidden_dim = Math.make_divisible(int(round(in_channels * expand_ratio)), 8)
		if self.expand_ratio != 1:
			self.block.add_module(
				name="exp_1x1",
				module=ConvLayer2D(
					in_channels=in_channels,
					out_channels=self.hidden_dim,
					kernel_size=1,
					use_norm=True,
					norm_layer=norm_layer,
					use_act=False,
				)
			)
			self.block.add_module(name="act_fn_1", module=act_fn)

		self.block.add_module(
			name="conv_3x3",
			module=ConvLayer2D(
				in_channels=self.hidden_dim,
				out_channels=self.hidden_dim,
				kernel_size=kernel_size,
				stride=stride,
				dilation=dilation,
				groups=self.hidden_dim,
				use_norm=True,
				norm_layer=norm_layer,
				use_act=False,
			)
		)
		self.block.add_module(name="act_fn_2", module=act_fn)

		if self.use_se:
			self.block.add_module(
				name="se",
				module=SqueezeExcitation(
					in_channels=self.hidden_dim,
					squeeze_factor=4,
					scale_fn_name="Hardsigmoid",
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

		self.use_res_connect = stride == 1 and in_channels == out_channels

	def forward(self, x):
		return x + self.block(x) if self.use_res_connect else self.block(x)


def get_configuration(mv3_mode):

	mv3_mode = mv3_mode.lower()
	mv3_config = dict()
	if mv3_mode == "small":
		# kernel_size, expansion_factor, in_channels, use_se, use_hs, stride
		mv3_config["layer_1"] = [[3, 1, 16, True, False, 2]]

		mv3_config["layer_2"] = [[3, 4.5, 24, False, False, 2]]

		mv3_config["layer_3"] = [[3, 3.67, 24, False, False, 1]]

		mv3_config["layer_4"] = [
			[5, 4, 40, True, True, 2],
			[5, 6, 40, True, True, 1],
			[5, 6, 40, True, True, 1],
			[5, 3, 48, True, True, 1],
			[5, 3, 48, True, True, 1],
		]

		mv3_config["layer_5"] = [
			[5, 6, 96, True, True, 2],
			[5, 6, 96, True, True, 1],
			[5, 6, 96, True, True, 1],
		]
		mv3_config["last_channels"] = 1024
	elif mv3_mode == "large":
		mv3_config["layer_1"] = [[3, 1, 16, False, False, 1]]
		# kernel_size, expansion_factor, in_channels, use_se, use_hs, stride
		mv3_config["layer_2"] = [
			[3, 4, 24, False, False, 2],
			[3, 3, 24, False, False, 1],
		]

		mv3_config["layer_3"] = [
			[5, 3, 40, True, False, 2],
			[5, 3, 40, True, False, 1],
			[5, 3, 40, True, False, 1],
		]

		mv3_config["layer_4"] = [
			[3, 6, 80, False, True, 2],
			[3, 2.5, 80, False, True, 1],
			[3, 2.3, 80, False, True, 1],
			[3, 2.3, 80, False, True, 1],
			[3, 6, 112, True, True, 1],
			[3, 6, 112, True, True, 1],
		]

		mv3_config["layer_5"] = [
			[5, 6, 160, True, True, 2],
			[5, 6, 160, True, True, 1],
			[5, 6, 160, True, True, 1],
		]
		mv3_config["last_channels"] = 1280

	return mv3_config


@Entry.register_entry(Entry.ClassificationModel)
class MobileNetV3(BaseEncoder):
	"""
	This class implements the `MobileNetv3 architecture <https://arxiv.org/abs/1905.02244>`_
	"""
	__slots__ = ["mode", "width_multiplier"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["large", 1.0]
	_types_ = [str, float]

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		num_classes = self.n_classes
		classifier_dropout = self.classifier_dropout
		if classifier_dropout == 0.0 or classifier_dropout is None:
			val = round(0.2 * self.width_multiplier, 3)
			classifier_dropout = Math.bound_fn(min_val=0.0, max_val=0.2, value=val)

		image_channels = 3
		input_channels = Math.make_divisible(16 * self.width_multiplier, 8)

		mv3_config = get_configuration(self.mode)

		self.head = nn.Sequential()
		self.head.add_module(
			name="conv_3x3_bn",
			module=ConvLayer2D(
				in_channels=image_channels,
				out_channels=input_channels,
				kernel_size=3,
				stride=2,
				use_norm=True,
				use_act=False,
			),
		)
		self.head.add_module(
			name="act", module=Entry.get_entity(Entry.Activation, entry_name="Hardswish", inplace=True)
		)
		self.model_conf_dict["head"] = {"in": image_channels, "out": input_channels}

		self.body = nn.Sequential()

		layer_1, out_channels = self._make_layer(
			mv3_config=mv3_config["layer_1"],
			width_mult=self.width_multiplier,
			input_channel=input_channels,
		)
		self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer1", module=layer_1)

		layer_2, out_channels = self._make_layer(
			mv3_config=mv3_config["layer_2"],
			width_mult=self.width_multiplier,
			input_channel=input_channels,
		)
		self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer2", module=layer_2)

		layer_3, out_channels = self._make_layer(
			mv3_config=mv3_config["layer_3"],
			width_mult=self.width_multiplier,
			input_channel=input_channels,
		)
		self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer3", module=layer_3)

		layer_4, out_channels = self._make_layer(
			mv3_config=mv3_config["layer_4"],
			width_mult=self.width_multiplier,
			input_channel=input_channels,
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer4", module=layer_4)

		layer_5, out_channels = self._make_layer(
			mv3_config=mv3_config["layer_5"],
			width_mult=self.width_multiplier,
			input_channel=input_channels,
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
		input_channels = out_channels
		self.body.add_module(name="layer5", module=layer_5)

		self.tail = nn.Sequential()
		out_channels = 6 * input_channels
		self.tail.add_module(
			name="conv_1x1",
			module=ConvLayer2D(
				in_channels=input_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=1,
				use_act=False,
				use_norm=True,
			),
		)
		self.tail.add_module(
			name="act", module=Entry.get_entity(Entry.Activation, entry_name="Hardswish", inplace=True)
		)
		self.model_conf_dict["tail"] = {
			"in": input_channels,
			"out": out_channels,
		}

		pool_type = self.global_pool
		last_channels = mv3_config["last_channels"]
		self.classifier = nn.Sequential()
		self.classifier.add_module(
			name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
		)
		self.classifier.add_module(
			name="fc1",
			module=LinearLayer(
				in_features=out_channels, out_features=last_channels, add_bias=True
			),
		)
		self.classifier.add_module(
			name="act", module=Entry.get_entity(Entry.Activation, entry_name="Hardswish", inplace=True)
		)
		if 0.0 < classifier_dropout < 1.0:
			self.classifier.add_module(
				name="classifier_dropout", module=Dropout(p=classifier_dropout)
			)
		self.classifier.add_module(
			name="classifier_fc",
			module=LinearLayer(
				in_features=last_channels, out_features=num_classes, add_bias=True
			),
		)

		self.model_conf_dict["cls"] = {"in": 6 * input_channels, "out": num_classes}

		# weight initialization
		InitParaUtil.initialize_weights(self)

	def _make_layer(self, mv3_config, width_mult, input_channel, dilate=False):
		prev_dilation = self.dilation
		mv3_block = nn.Sequential()
		count = 0

		for i in range(len(mv3_config)):
			for kernel_size, expansion_factor, in_channels, use_se, use_hs, stride in [
				mv3_config[i]
			]:
				output_channel = Math.make_divisible(
					in_channels * width_mult, self.round_nearest
				)

				if dilate and count == 0:
					self.dilation *= stride
					stride = 1

				layer = InvertedResidualSE(
					in_channels=input_channel,
					out_channels=output_channel,
					kernel_size=kernel_size,
					stride=stride,
					expand_ratio=expansion_factor,
					dilation=prev_dilation if count == 0 else self.dilation,
					use_hs=use_hs,
					use_se=use_se,
				)
				mv3_block.add_module(
					name="mv3_s_{}_idx_{}".format(stride, count), module=layer
				)
				count += 1
				input_channel = output_channel
		return mv3_block, input_channel

	def __repr__(self):
		return self._repr_by_line() + "\n" + nn.Module.__repr__(self)
