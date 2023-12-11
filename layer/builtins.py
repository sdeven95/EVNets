from torch import nn
from utils.type_utils import Par
from utils.entry_utils import Entry
import math
from . import BuiltinLayer


from typing import Union, Tuple, Sequence


class Identity(nn.Identity):
	pass


@Entry.register_entry(Entry.Layer)
class LinearLayer(BuiltinLayer, nn.Linear):
	__slots__ = ["in_features", "out_features", "add_bias"]
	_disp_ = __slots__

	def profile(self, x):
		out, paras, macs = super().profile(x)
		macs = paras
		return out, paras, macs

	def __init__(
		self,
		in_features: int,
		out_features: int,
		add_bias: bool = True
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class Conv1d(BuiltinLayer, nn.Conv1d):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "padding", "dilation", "groups", "add_bias", "padding_mode"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 0, 1, 1, True, "zeros"]

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		padding: Union[int, tuple] = 0,
		dilation: Union[int, tuple] = 1,
		groups: int = 1,
		add_bias: bool = True,
		padding_mode: str = "zeros",  # TODO: refine this type
	) -> None:
		super().__init__(**Par.purify(locals()))

	def profile(self, x):
		assert x.dim() == 3, f"Conv1d requires 3-dimensional input (BxCxL). Provided input has shape: {x.size()}"

		b, in_c, in_l = x.size()
		assert in_c == self.in_channels, f"inc[{in_c}]!=in_channels[{self.in_channels}]"

		k_s: int = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
		d_s: int = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
		s_s: int = self.stride[0] if isinstance(self.stride, tuple) else self.stride
		p_s: int = self.padding[0] if isinstance(self.padding, tuple) else self.padding

		out_l = math.floor((in_l + 2 * p_s - d_s * (k_s - 1) - 1)/s_s + 1) * 1.0

		macs = k_s * self.in_channels * self.out_channels * out_l
		macs /= self.groups

		if self.add_bias:
			macs += self.out_channels * out_l

		out, paras, _ = super().profile(x)

		return out, paras, macs


@Entry.register_entry(Entry.Layer)
class Conv2d(BuiltinLayer, nn.Conv2d):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "padding", "dilation", "groups", "add_bias", "padding_mode"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 0, 1, 1, True, "zeros"]

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		padding: Union[int, tuple] = 0,
		dilation: Union[int, tuple] = 1,
		groups: int = 1,
		add_bias: bool = True,
		padding_mode: str = "zeros",  # TODO: refine this type
	) -> None:
		super().__init__(**Par.purify(locals()))

	def profile(self, x):
		assert x.dim() == 4, f"Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {x.size()}"

		b, in_c, in_h, in_w = x.size()
		assert in_c == self.in_channels, f"inc[{in_c}]!=in_channels[{self.in_channels}]"

		if isinstance(self.kernel_size, int):
			self.kernel_size = (self.kernel_size, self.kernel_size)
		if isinstance(self.dilation, int):
			self.dilation = (self.dilation, self.dilation)
		if isinstance(self.stride, int):
			self.stride = (self.stride, self.stride)
		if isinstance(self.padding, int):
			self.padding = (self.padding, self.padding)

		k_h, k_w = self.kernel_size
		d_h, d_w = self.dilation
		s_h, s_w = self.stride
		p_h, p_w = self.padding

		out_h = math.floor((in_h + 2 * p_h - d_h * (k_h - 1) - 1)/s_h + 1)
		out_w = math.floor((in_w + 2 * p_w - d_w * (k_w - 1) - 1)/s_w + 1)

		macs = (k_h * k_w) * (self.in_channels * self.out_channels) * (out_h * out_w) * 1.0
		macs /= self.groups

		if self.add_bias:
			macs += self.out_channels * out_h * out_w

		out, paras, _ = super().profile(x)

		return out, paras, macs


@Entry.register_entry(Entry.Layer)
class Conv3d(BuiltinLayer, nn.Conv3d):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "padding", "dilation", "groups", "add_bias", "padding_mode"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 0, 1, 1, True, "zeros"]

	def profile(self, x):
		assert x.dim() == 5, f"Conv3d requires 5-dimensional input (BxCxDxHxW). Provided input has shape: {x.size()}"

		b, in_c, in_d, in_h, in_w = x.size()
		assert in_c == self.in_channels, f"inc[{in_c}]!=in_channels[{self.in_channels}]"

		if isinstance(self.kernel_size, int):
			self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
		if isinstance(self.dilation, int):
			self.dilation = (self.dilation, self.dilation, self.dilation)
		if isinstance(self.stride, int):
			self.stride = (self.stride, self.stride, self.stride)
		if isinstance(self.padding, int):
			self.padding = (self.padding, self.padding, self.padding)

		k_d, k_h, k_w = self.kernel_size
		d_d, d_h, d_w = self.dilation
		s_d, s_h, s_w = self.stride
		p_d, p_h, p_w = self.padding

		out_d = math.floor((in_d + 2 * p_d - d_d * (k_d - 1) - 1)/s_d + 1)
		out_h = math.floor((in_h + 2 * p_h - d_h * (k_h - 1) - 1)/s_h + 1)
		out_w = math.floor((in_w + 2 * p_w - d_w * (k_w - 1) - 1)/s_w + 1)

		macs = (k_d * k_h * k_w) * (self.in_channels * self.out_channels) * (out_d * out_h * out_w) * 1.0
		macs /= self.groups

		if self.add_bias:
			macs += self.out_channels * out_d * out_h * out_w

		out, paras, _ = super().profile(x)

		return out, paras, macs

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		padding: Union[int, tuple] = 0,
		dilation: Union[int, tuple] = 1,
		groups: int = 1,
		add_bias: bool = True,
		padding_mode: str = "zeros",  # TODO: refine this type
	) -> None:
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class ConvTranspose1d(BuiltinLayer, nn.ConvTranspose1d):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "padding", "dilation", "groups", "add_bias", "padding_mode", "output_padding"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 0, 0, 1, True, "zeros", None]

	def profile(self, x):
		assert x.dim() == 3, f"Conv1d requires 3-dimensional input (BxCxL). Provided input has shape: {x.size()}"

		b, in_c, in_l = x.size()
		assert in_c == self.in_channels, f"in_c[{in_c}] != in_channels[{self.in_channels}]"

		k_l = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
		s_l = self.stride[0] if isinstance(self.stride, tuple) else self.stride
		d_l = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
		p_l = self.padding[0] if isinstance(self.padding, tuple) else self.padding
		op_l = self.output_padding[0] if isinstance(self.output_padding, tuple) else self.output_padding

		out_l = (in_l - 1) * s_l - 2 * p_l + d_l * (k_l - 1) + op_l + 1

		macs = k_l * (self.in_channels * self.out_channels) * out_l * 1.0
		macs /= self.groups

		if self.add_bias:
			macs += self.out_channels * out_l

		out, paras, _ = super().profile(x)

		return out, paras, macs

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		padding: Union[int, tuple] = 0,
		dilation: Union[int, tuple] = 1,
		output_padding: Union[int, tuple] = 0,
		groups: int = 1,
		add_bias: bool = True,
		padding_mode: str = "zeros",
	) -> None:
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class ConvTranspose2d(BuiltinLayer, nn.ConvTranspose2d):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "padding", "dilation", "groups", "add_bias", "padding_mode", "output_padding"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 0, 0, 1, True, "zeros", None]

	def profile(self, x):
		assert x.dim() == 4, f"Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {x.size()}"

		if isinstance(self.kernel_size, int):
			self.kernel_size = (self.kernel_size, self.kernel_size)
		if isinstance(self.stride, int):
			self.stride = (self.stride, self.stride)
		if isinstance(self.dilation, int):
			self.dilation = (self.dilation, self.dilation)
		if isinstance(self.padding, int):
			self.padding = (self.padding, self.padding)
		if isinstance(self.output_padding, int):
			self.output_padding = (self.output_padding, self.output_padding)

		b, in_c, in_h, in_w = x.size()
		assert in_c == self.in_channels, f"in_c[{in_c}] != in_channels[{self.in_channels}]"

		k_h, k_w = self.kernel_size
		s_h, s_w = self.stride
		d_h, d_w = self.dilation
		p_h, p_w = self.padding
		op_h, op_w = self.output_padding

		out_h = (in_h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
		out_w = (in_w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1

		macs = (k_h * k_w) * (self.in_channels * self.out_channels) * (out_h * out_w) * 1.0
		macs /= self.groups

		if self.add_bias:
			macs += self.out_channels * out_h * out_w

		out, paras, _ = super().profile(x)

		return out, paras, macs

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		padding: Union[int, tuple] = 0,
		output_padding: Union[int, tuple] = 0,
		groups: int = 1,
		add_bias: bool = True,
		dilation: Union[int, tuple] = 1,
		padding_mode: str = "zeros",
	) -> None:
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class ConvTranspose3d(BuiltinLayer, nn.ConvTranspose3d):
	__slots__ = [
		"in_channels", "out_channels", "kernel_size",
		"stride", "padding", "dilation", "groups", "add_bias", "padding_mode", "output_padding"]
	_disp_ = __slots__

	# defaults = [3, 3, 3, 1, 0, 1, 1, True, "zeros", None]

	def profile(self, x):
		assert x.dim() == 5, f"Conv2d requires 5-dimensional input (BxCxDxHxW). Provided input has shape: {x.size()}"

		if isinstance(self.kernel_size, int):
			self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
		if isinstance(self.stride, int):
			self.stride = (self.stride, self.stride, self.stride)
		if isinstance(self.dilation, int):
			self.dilation = (self.dilation, self.dilation, self.dilation)
		if isinstance(self.padding, int):
			self.padding = (self.padding, self.padding, self.padding)
		if isinstance(self.output_padding, int):
			self.output_padding = (self.output_padding, self.output_padding, self.output_padding)

		b, in_c, in_d, in_h, in_w = x.size()
		assert in_c == self.in_channels, f"in_c[{in_c}] != in_channels[{self.in_channels}]"

		k_d, k_h, k_w = self.kernel_size
		s_d, s_h, s_w = self.stride
		d_d, d_h, d_w = self.dilation
		p_d, p_h, p_w = self.padding
		op_d, op_h, op_w = self.output_padding

		out_d = (in_d - 1) * s_d - 2 * p_d + d_d * (k_d - 1) + op_d + 1
		out_h = (in_h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
		out_w = (in_w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1

		macs = (k_d * k_h * k_w) * (self.in_channels * self.out_channels) * (out_d * out_h * out_w) * 1.0
		macs /= self.groups

		if self.add_bias:
			macs += self.out_channels * out_d * out_h * out_w

		out, paras, _ = super().profile(x)

		return out, paras, macs

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: Union[int, tuple] = 3,
		stride: Union[int, tuple] = 1,
		padding: Union[int, tuple] = 0,
		output_padding: Union[int, tuple] = 0,
		groups: int = 1,
		add_bias: bool = True,
		dilation: Union[int, tuple] = 1,
		padding_mode: str = "zeros",
	) -> None:
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class MaxPool2d(BuiltinLayer, nn.MaxPool2d):
	__slots__ = ["kernel_size", "stride", "padding", "dilation"]
	_disp_ = __slots__

	# defaults = [3, 2, 1],

	def __init__(
			self,
			kernel_size: Union[int, Sequence] = 2,
			stride: Union[int, Sequence] = 2,
			padding: Union[int, Sequence] = 0,
			dilation: Union[int, Sequence] = 1,
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class AvgPool2d(BuiltinLayer, nn.AvgPool2d):
	__slots__ = ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"]
	_disp_ = __slots__

	# defaults = [3, None, 0, False, True, None]

	def __init__(
		self,
		kernel_size: Union[int, Tuple] = 2,
		stride: Union[int, Tuple] = 2,
		padding: Union[int, Tuple] = 0,
		ceil_mode: bool = False,
		count_include_pad: bool = True,
		divisor_override: int = None
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Layer)
class AdaptiveAvgPool2d(BuiltinLayer, nn.AdaptiveAvgPool2d):
	__slots__ = ["output_size", ]
	_disp_ = __slots__

	# defaults = [1, ],

	def __init__(self, output_size: Union[int, Tuple[int, ...]] = 1):
		super().__init__(output_size=output_size)


@Entry.register_entry(Entry.Layer)
class Dropout(BuiltinLayer, nn.Dropout):
	__slots__ = ["p", "inplace"]
	_disp_ = __slots__

	# defaults = [0.5, False]

	def __init__(self, p: float = 0.5, inplace: bool = False):
		super().__init__(p=p, inplace=inplace)


@Entry.register_entry(Entry.Layer)
class Dropout2D(BuiltinLayer, nn.Dropout2d):
	__slots__ = ["p", "inplace"]
	_disp_ = __slots__

	# defaults = [0.5, False]

	def __init__(self, p: float = 0.5, inplace: bool = False):
		super().__init__(p=p, inplace=inplace)


@Entry.register_entry(Entry.Layer)
class Flatten(BuiltinLayer, nn.Flatten):
	__slots__ = ["start_dim", "end_dim"]
	_disp_ = __slots__

	# _defaults_ = [1, -1]

	def __init__(self, start_dim: int = 1, end_dim: int = -1):
		super().__init__(start_dim=start_dim, end_dim=end_dim)


@Entry.register_entry(Entry.Layer)
class PixelShuffle(BuiltinLayer, nn.PixelShuffle):
	__slots__ = ["upscale_factor", ]
	_disp_ = __slots__

	# _defaults = [3]

	def __init__(self, upscale_factor: int) -> None:
		super().__init__(upscale_factor=upscale_factor)


@Entry.register_entry(Entry.Layer)
class Softmax(BuiltinLayer, nn.Softmax):
	__slots__ = ["dim", ]
	_disp_ = __slots__

	# _defaults_ = [None]

	def __init__(self, dim: int = None):
		super().__init__(dim=dim)


@Entry.register_entry(Entry.Layer)
class UpSample(BuiltinLayer, nn.Upsample):
	__slots__ = ["size", "scale_factor", "mode", "align_corners", "recompute_scale_factor"]
	_disp_ = __slots__

	# _defaults_ [None, None, "nearest", None, None]

	def __init__(
			self,
			size: Union[int, Tuple[int, ...]] = None,
			scale_factor: Union[float, Tuple[float, ...]] = None,
			mode: str = "nearest",
			align_corners: bool = None,
			recompute_scale_factor: bool = None) -> None:
		super().__init__(**Par.purify(locals()))
