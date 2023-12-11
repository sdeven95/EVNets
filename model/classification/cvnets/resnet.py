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

from model.classification import BaseEncoder
from layer.builtins import Dropout, Identity, LinearLayer
from layer.extensions import ConvLayer2D, GlobalPool

from layer.utils import InitParaUtil
from layer import ExtensionLayer
from utils.logger import Logger
from utils.entry_utils import Entry

from torch import nn
from typing import Dict, Tuple, Optional
from utils.type_utils import Prf, Par


@Entry.register_entry(Entry.Layer)
class BasicResNetBlock(ExtensionLayer):
	__slots__ = ["in_channels", "mid_channels", "out_channels", "stride", "dilation", "dropout"]
	_disp_ = __slots__

	expansion: int = 1

	def __init__(
		self,
		in_channels: int,
		mid_channels: int,
		out_channels: int,
		stride: Optional[int] = 1,
		dilation: Optional[int] = 1,
		dropout: Optional[float] = 0.0,
	) -> None:
		super().__init__(**Par.purify(locals()))

		cbr_1 = ConvLayer2D(
			in_channels=in_channels,
			out_channels=mid_channels,
			kernel_size=3,
			stride=stride,
			dilation=dilation,
			use_norm=True,
			use_act=True,
		)
		cb_2 = ConvLayer2D(
			in_channels=mid_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=1,
			use_norm=True,
			use_act=False,
			dilation=dilation,
		)

		self.block = nn.Sequential()
		self.block.add_module(name="conv_batch_act_1", module=cbr_1)
		self.block.add_module(name="conv_batch_2", module=cb_2)
		if 0.0 < dropout < 1.0:
			self.block.add_module(name="dropout", module=Dropout(p=dropout))

		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.down_sample = Identity()
		if stride == 2:
			self.down_sample = ConvLayer2D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=stride,
				use_norm=True,
				use_act=False,
			)

		self.final_act = Entry.get_entity(Entry.Activation, num_parameters=out_channels)

	def forward(self, x):
		out = self.block(x)
		res = self.down_sample(x)
		out = out + res
		return self.final_act(out)

	def profile(self, x):
		out, n_paras, n_macs = Prf.profile_list(self.block, x=x)
		res, n_paras_down, n_macs_down = Prf.profile_list(self.down_sample, x=x)
		return out, n_paras + n_paras_down, n_macs + n_macs_down


@Entry.register_entry(Entry.Layer)
class BottleneckResNetBlock(ExtensionLayer):
	"""
	This class defines the Basic block in the `ResNet model <https://arxiv.org/abs/1512.03385>`_
	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
		mid_channels (int): :math:`C_{mid}` from an expected tensor of size :math:`(N, C_{mid}, H_{out}, W_{out})`
		out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
		stride (Optional[int]): Stride for convolution. Default: 1
		dilation (Optional[int]): Dilation for convolution. Default: 1
		dropout (Optional[float]): Dropout after second convolution. Default: 0.0

	Shape:
		- Input: :math:`(N, C_{in}, H_{in}, W_{in})`
		- Output: :math:`(N, C_{out}, H_{out}, W_{out})`

	"""
	__slots__ = ["in_channels", "mid_channels", "out_channels", "stride", "dilation", "dropout"]
	_disp_ = __slots__

	# defaults = [None, None, None, 1, 1, 0.0]

	expansion: int = 4

	def __init__(
		self,
		in_channels: int,
		mid_channels: int,
		out_channels: int,
		stride: Optional[int] = 1,
		dilation: Optional[int] = 1,
		dropout: Optional[float] = 0.0,
	) -> None:
		super().__init__(**Par.purify(locals()))

		cbr_1 = ConvLayer2D(
			in_channels=in_channels,
			out_channels=mid_channels,
			kernel_size=1,
			stride=1,
			use_norm=True,
			use_act=True,
		)
		cbr_2 = ConvLayer2D(
			in_channels=mid_channels,
			out_channels=mid_channels,
			kernel_size=3,
			stride=stride,
			use_norm=True,
			use_act=True,
			dilation=dilation,
		)
		cb_3 = ConvLayer2D(
			in_channels=mid_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=1,
			use_norm=True,
			use_act=False,
		)
		self.block = nn.Sequential()
		self.block.add_module(name="conv_batch_act_1", module=cbr_1)
		self.block.add_module(name="conv_batch_act_2", module=cbr_2)
		self.block.add_module(name="conv_batch_3", module=cb_3)
		if 0.0 < dropout < 1.0:
			self.block.add_module(name="dropout", module=Dropout(p=dropout))

		self.down_sample = Identity()
		if stride == 2:
			self.down_sample = ConvLayer2D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=stride,
				use_norm=True,
				use_act=False,
			)
		elif in_channels != out_channels:
			self.down_sample = ConvLayer2D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=1,
				use_norm=True,
				use_act=False,
			)

		self.final_act = Entry.get_entity(Entry.Activation, num_parameters=out_channels)

	def forward(self, x):
		out = self.block(x)
		res = self.down_sample(x)
		out = out + res
		return self.final_act(out)

	def profile(self, x):
		out, n_paras, n_macs = Prf.profile_list(self.block, x=x)
		res, n_paras_down, n_macs_down = Prf.profile_list(self.down_sample, x=x)
		return out, n_paras + n_paras_down, n_macs + n_macs_down


def get_configuration(depth) -> Dict:
	resnet_config = dict()

	if depth == 18:
		# 112 * 112 => 112 * 112
		resnet_config["layer2"] = {
			"num_blocks": 2,
			"mid_channels": 64,
			"block_type": "basic",
			"stride": 1,
		}
		# 56 * 56 => 28 * 28
		resnet_config["layer3"] = {
			"num_blocks": 2,
			"mid_channels": 128,
			"block_type": "basic",
			"stride": 2,
		}
		# 28 * 28 => 14 * 14
		resnet_config["layer4"] = {
			"num_blocks": 2,
			"mid_channels": 256,
			"block_type": "basic",
			"stride": 2,
		}

		# 14 * 14 => 7 * 7
		resnet_config["layer5"] = {
			"num_blocks": 2,
			"mid_channels": 512,
			"block_type": "basic",
			"stride": 2,
		}
	elif depth == 34:
		resnet_config["layer2"] = {
			"num_blocks": 3,
			"mid_channels": 64,
			"block_type": "basic",
			"stride": 1,
		}
		resnet_config["layer3"] = {
			"num_blocks": 4,
			"mid_channels": 128,
			"block_type": "basic",
			"stride": 2,
		}
		resnet_config["layer4"] = {
			"num_blocks": 6,
			"mid_channels": 256,
			"block_type": "basic",
			"stride": 2,
		}
		resnet_config["layer5"] = {
			"num_blocks": 3,
			"mid_channels": 512,
			"block_type": "basic",
			"stride": 2,
		}
	elif depth == 50:
		resnet_config["layer2"] = {
			"num_blocks": 3,
			"mid_channels": 64,
			"block_type": "bottleneck",
			"stride": 1,
		}
		resnet_config["layer3"] = {
			"num_blocks": 4,
			"mid_channels": 128,
			"block_type": "bottleneck",
			"stride": 2,
		}
		resnet_config["layer4"] = {
			"num_blocks": 6,
			"mid_channels": 256,
			"block_type": "bottleneck",
			"stride": 2,
		}
		resnet_config["layer5"] = {
			"num_blocks": 3,
			"mid_channels": 512,
			"block_type": "bottleneck",
			"stride": 2,
		}
	elif depth == 101:
		resnet_config["layer2"] = {
			"num_blocks": 3,
			"mid_channels": 64,
			"block_type": "bottleneck",
			"stride": 1,
		}
		resnet_config["layer3"] = {
			"num_blocks": 4,
			"mid_channels": 128,
			"block_type": "bottleneck",
			"stride": 2,
		}
		resnet_config["layer4"] = {
			"num_blocks": 23,
			"mid_channels": 256,
			"block_type": "bottleneck",
			"stride": 2,
		}
		resnet_config["layer5"] = {
			"num_blocks": 3,
			"mid_channels": 512,
			"block_type": "bottleneck",
			"stride": 2,
		}
	else:
		Logger.error(
			"ResNet models are supported with depths of 18, 34, 50 and 101. Please specify depth using "
			"--model.classification.resnet.depth flag. Got: {}".format(depth)
		)
	return resnet_config


@Entry.register_entry(Entry.ClassificationModel)
class ResNet(BaseEncoder):
	"""
	This class implements the `ResNet architecture <https://arxiv.org/pdf/1512.03385.pdf>`_

	.. note::
		Our ResNet implementation is different from the original implementation in two ways:
		1. First 7x7 strided conv is replaced with 3x3 strided conv
		2. MaxPool operation is replaced with another 3x3 strided depth-wise conv
	"""
	__slots__ = ["depth", "dropout"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [50, 0.0]
	_types_ = [int, float]

	configure_file = "config/classification/food172/resnet.yaml"

	def __init__(self) -> None:
		super().__init__()

		image_channels = 3
		input_channels = 64

		cfg = get_configuration(depth=self.depth)

		self.model_conf_dict = dict()

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

		layer_1 = ConvLayer2D(
			in_channels=input_channels,
			out_channels=input_channels,
			kernel_size=3,
			stride=2,
			use_norm=True,
			use_act=True,
			groups=input_channels,
		)
		self.model_conf_dict["layer1"] = {"in": input_channels, "out": input_channels}
		self.body.add_module(name="layer1", module=layer_1)

		layer_2, out_channels = self._make_layer(
			in_channels=input_channels, layer_config=cfg["layer2"]
		)
		self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
		self.body.add_module(name="layer2", module=layer_2)
		input_channels = out_channels

		layer_3, out_channels = self._make_layer(
			in_channels=input_channels, layer_config=cfg["layer3"]
		)
		self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
		self.body.add_module(name="layer3", module=layer_3)
		input_channels = out_channels

		layer_4, out_channels = self._make_layer(
			in_channels=input_channels,
			layer_config=cfg["layer4"],
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
		self.body.add_module(name="layer4", module=layer_4)
		input_channels = out_channels

		layer_5, out_channels = self._make_layer(
			in_channels=input_channels,
			layer_config=cfg["layer5"],
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
		self.body.add_module(name="layer5", module=layer_5)
		input_channels = out_channels

		self.tail = Identity()
		self.model_conf_dict["tail"] = {"in": input_channels, "out": input_channels}

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
				in_features=input_channels, out_features=self.n_classes, add_bias=True
			),
		)

		self.model_conf_dict["cls"] = {"in": input_channels, "out": self.n_classes}

		# weight initialization
		InitParaUtil.initialize_weights(self)

	def _make_layer(
		self,
		in_channels: int,
		layer_config: Dict,
		dilate: bool = False,
	) -> Tuple[nn.Sequential, int]:
		block_type = (
			BottleneckResNetBlock
			if layer_config.get("block_type", "bottleneck").lower() == "bottleneck"
			else BasicResNetBlock
		)
		mid_channels = layer_config.get("mid_channels")
		num_blocks = layer_config.get("num_blocks", 2)
		stride = layer_config.get("stride", 1)

		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1

		out_channels = block_type.expansion * mid_channels

		block = nn.Sequential()
		block.add_module(
			name="block_0",
			module=block_type(
				in_channels=in_channels,
				mid_channels=mid_channels,
				out_channels=out_channels,
				stride=stride,
				dilation=previous_dilation,
				dropout=self.dropout,
			),
		)

		for block_idx in range(1, num_blocks):
			block.add_module(
				name="block_{}".format(block_idx),
				module=block_type(
					in_channels=out_channels,
					mid_channels=mid_channels,
					out_channels=out_channels,
					stride=1,
					dilation=self.dilation,
					dropout=self.dropout,
				),
			)

		return block, out_channels
