import random
from torch import nn, Tensor
import torch

from utils.type_utils import Dct, Par
from utils.entry_utils import Entry
from .builtins import Conv2d, Conv3d
from utils.math_utils import Math
from utils import Util

from . import ExtensionLayer

from typing import Union, Any, List, Tuple, Optional


@Entry.register_entry(Entry.Layer)
class GlobalPool(ExtensionLayer):
	__slots__ = ["pool_type", "keep_dim"]
	_disp_ = __slots__

	pool_types = ["mean", "rms", "abs"]

	def __init__(self, pool_type: str = "mean", keep_dim: bool = False):
		super().__init__(pool_type=pool_type, keep_dim=keep_dim)
		if self.pool_type not in self.pool_types:
			raise KeyError(f"Don't support global pool type [{self.pool_type}]")

	def forward(self, x):
		if x.dim() == 4:
			dims = [-2, -1]
		elif x.dim() == 5:
			dims = [-3, -2, -1]
		else:
			raise NotImplementedError("Currently 2D and 3D global pooling supported")

		if self.pool_type == "rms":
			x = x ** 2
			x = torch.mean(x, dims, keepdim=self.keep_dim)
			x = x ** -0.5
		elif self.pool_type == "abs":
			x = torch.mean(abs(x), dims, keepdim=self.keep_dim)
		else:
			x = torch.mean(x, dims, keepdim=self.keep_dim)
		return x


# @Entry.register_entry(Entry.Layer)
# class NormActLayer(ExtensionLayer):
# 	def __init__(self, norm_layer=None, act_layer=None, **kwargs):
# 		super().__init__(**kwargs)
#
# 		self.block = nn.Sequential()
# 		if not norm_layer or isinstance(norm_layer, str):
# 			norm_layer = Entry.get_entity(Entry.Norm, entry_name=norm_layer, **kwargs)
# 		self.block.add_module(name="norm", module=norm_layer)
# 		self.normalization = norm_layer.__class__.__name__
#
# 		if not act_layer or isinstance(act_layer, str):
# 			act_layer = Entry.get_entity(Entry.Activation, entry_name=act_layer, **kwargs)
# 		self.block.add_module(name="act", module=act_layer)
# 		self.activation = act_layer.__class__.__name__
#
# 	def forward(self, x):
# 		return self.block(x)


@Entry.register_entry(Entry.Layer)
class ConvLayer2D(ExtensionLayer):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "dilation", "groups", "add_bias", "padding_mode",
		"use_norm", "norm_layer", "use_act", "act_layer",
		"padding"
	]
	_disp_ = __slots__

	# defaults = [
	# 	3, 3, 3,
	# 	1, 1, 1, False, "zeros",
	# 	True, None, True, None]

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		dilation: Union[int, tuple] = 1,
		groups: int = 1,
		add_bias: bool = False,
		padding_mode: str = "zeros",  # TODO: refine this type
		use_norm: bool = True,
		norm_layer: Union[str, type, object] = None,
		use_act: bool = True,
		act_layer: Union[str, type, object] = None,
	):
		super().__init__(**Par.purify(locals()))

		assert not in_channels % groups, f"Input channels are not divided by groups, {in_channels}%{groups}!=0"
		assert not out_channels % groups, f"Output channels are not divided by groups, {out_channels}%{groups}!=0"

		if use_norm:
			self.norm_layer = Entry.get_entity_instance(Entry.Norm, entry_name=norm_layer, num_features=out_channels)
			if "Batch" in self.norm_layer.__class__.__name__:
				assert not add_bias, "Don't use bias when using batch normalization layer."
			elif "Layer" in self.norm_layer.__class__.__name__:
				assert add_bias, "Set bias to true when using layer normalization layer. "

		if use_act:
			self.act_layer = Entry.get_entity_instance(Entry.Activation, entry_name=self.act_layer)

		self.kernel_size = Util.pair(kernel_size)
		self.stride = Util.pair(stride)
		self.dilation = Util.pair(dilation)

		self.padding = (
			int((self.kernel_size[0] - 1) / 2) * self.dilation[0],
			int((self.kernel_size[1] - 1) / 2) * self.dilation[1],
		)

		after_pop = Dct.pop_keys(Par.purify(locals()), ["use_norm", "norm_layer", "use_act", "act_layer"])
		after_pop = Dct.update_key_values(target=after_pop, source=self.__dict__)
		after_pop["padding"] = self.padding
		self.conv_layer = Conv2d(**after_pop)

		self.block = nn.Sequential()
		self.block.add_module(name="conv", module=self.conv_layer)
		if self.use_norm:
			self.block.add_module(name="norm", module=self.norm_layer)
		if self.use_act:
			self.block.add_module(name="act", module=self.act_layer)

	def forward(self, x):
		return self.block(x)

	def profile(self, x):
		x, paras, macs = self.conv_layer.profile(x)
		if self.use_norm:
			x, paras_, macs_ = self.norm_layer.profile(x)
			paras += paras_
			# macs += macs_
		if self.use_act:
			x, paras_, macs_ = self.act_layer.profile(x)
			paras += paras_
			# macs += macs_

		return x, paras, macs


@Entry.register_entry(Entry.Layer)
class ConvLayer3D(ExtensionLayer):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size", "stride", "dilation",
		"groups", "add_bias", "padding_mode",
		"use_norm", "norm_layer", "use_act", "act_layer"
	]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 1, 1, False, "zeros", True, None, True, None]

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		dilation: Union[int, tuple] = 1,
		groups: int = 1,
		add_bias: bool = False,
		padding_mode: str = "zeros",  # TODO: refine this type
		use_norm: bool = True,
		norm_layer: Union[str, type, object] = None,
		use_act: bool = True,
		act_layer: Union[str, type, object] = None
	):
		super().__init__(**Par.purify(locals()))

		assert not in_channels % groups, f"Input channels are not divided by groups, {in_channels}%{groups}!=0"
		assert not out_channels % groups, f"Output channels are not divided by groups, {out_channels}%{groups}!=0"

		if use_norm:
			self.norm_layer = Entry.get_entity_instance(Entry.Norm, entry_name=norm_layer, num_featrues=out_channels)
			if "Batch" in self.norm_layer.__class__.__name__:
				assert not add_bias, "Don't use bias when using batch normalization layer."
			elif "Layer" in self.norm_layer.__class__.__name__:
				assert add_bias, "Set add_bias to true when using layer normalization layer. "

		if use_act:
			self.act_layer = Entry.get_entity_instance(Entry.Activation, entry_name=act_layer)

		self.kernel_size = Util.triple(kernel_size)
		self.stride = Util.triple(stride)
		self.dilation = Util.triple(dilation)

		self.padding = tuple(int((self.kernel_size[i] - 1) / 2) * self.dilation[i] for i in range(3))

		after_pop = Dct.pop_keys(Par.purify(locals()), ["use_norm", "norm_layer", "use_act", "act_layer"])
		after_pop = Dct.update_key_values(target=after_pop, source=self.__dict__)
		after_pop["padding"] = self.padding
		self.conv_layer = Conv3d(**after_pop)

		self.block = nn.Sequential()
		self.block.add_module(name="conv", module=self.conv_layer)
		if self.use_norm:
			self.block.add_module(name="norm", module=self.act_layer)
		if self.use_act:
			self.block.add_module(name="act", module=self.act_layer)

	def forward(self, x):
		return self.block(x)

	def profile(self, x):
		x, paras, macs = self.conv_layer.profile(x)
		if self.use_norm:
			x, paras_, macs_ = self.norm_layer.profile(x)
			paras += paras_
			macs += macs_
		if self.use_act:
			x, paras_, macs_ = self.act_layer.profile(x)
			paras += paras_
			macs += macs_

		return x, paras, macs


@Entry.register_entry(Entry.Layer)
class RandomApply(ExtensionLayer):
	"""
	This layer randomly applies a list of modules during training.

	Args:
		module_list (List): List of modules
		keep_p (Optional[float]): Keep P modules from the list during training. Default: 0.8 (or 80%)
	"""
	__slots__ = ["keep_p", "module_list", "keep_k", "module_indexes"]
	_disp_ = __slots__[:1]

	# defaults = [0.8]

	def __init__(
			self,
			module_list: List,
			keep_p: Optional[float] = 0.8
	) -> None:
		super().__init__(keep_p=keep_p)
		self.module_list = module_list
		n_modules = len(module_list)
		self.module_indexes = [i for i in range(1, n_modules)]
		k = int(round(n_modules * keep_p))
		self.keep_k = Math.bound_fn(min_val=1, max_val=n_modules, value=k)

	def forward(self, x: Tensor) -> Tensor:
		if self.training:
			indexes = [0] + sorted(random.sample(self.module_indexes, k=self.keep_k))
			for idx in indexes:
				x = self.module_list[idx](x)
		else:
			for layer in self.module_list:
				x = layer(x)
		return x

	def profile(self, x) -> Tuple[Tensor, float, float]:
		params, macs = 0.0, 0.0
		for layer in self.module_list:
			x, p, m = layer.profile(x)
			params += p
			macs += m
		return x, params, macs

	def __repr__(self):
		format_string = "{}(apply_k (N={})={}, ".format(
			self.__class__.__name__, len(self.module_list), self.keep_k
		)
		for layer in self.module_list:
			format_string += "\n\t {}".format(layer)
		format_string += "\n)"
		return format_string
