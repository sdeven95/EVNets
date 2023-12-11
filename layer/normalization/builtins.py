from torch import nn

from . import BaseNorm
from utils.type_utils import Cfg, Par, Dsp
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Norm)
class BatchNorm2d(BaseNorm, nn.BatchNorm2d):
	def profile(self, x):
		assert x.dim() in (3, 4), f"input tensor shape of InstanceNorm2d should be [N, C, H, W] or [C, H, W], but got {x.dim()}"
		output, paras, macs = super().profile(x)
		c, h, w = 1, 1, 1
		if x.dim() == 3:
			c, h, w = x.shape
		elif x.dim() == 4:
			_, c, h, w = x.shape

		macs += c * h * w * (4 if self.affine else 2)

		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class BatchNorm1d(BaseNorm, nn.BatchNorm1d):
	def profile(self, x):
		assert x.dim() in (2, 3), f"input tensor shape of BatchNorm1d should be [N, C] or [N, C, L], but got {x.dim()}"
		output, paras, macs = super().profile(x)
		c, ll = 1, 1
		if x.dim() == 2:
			_, c = x.shape
		elif x.dim() == 3:
			_, c, ll = x.shape

		macs += c * ll * (4 if self.affine else 2)

		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class BatchNorm3d(BaseNorm, nn.BatchNorm3d):
	def profile(self, x):
		assert x.dim() in (4, 5), f"input tensor shape of InstanceNorm3d should be [N, C, D, H, W] or [C, D, H, W], but got {x.dim()}"
		output, paras, macs = super().profile(x)
		c, d, h, w = 1, 1, 1, 1
		if x.dim() == 4:
			c, d, h, w = x.shape
		elif x.dim() == 5:
			_, c, d, h, w = x.shape

		macs += c * d * h * w * (4 if self.affine else 2)

		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class SyncBatchNorm(BaseNorm, nn.SyncBatchNorm):
	pass


@Entry.register_entry(Entry.Norm)
class InstanceNorm2d(BaseNorm, nn.InstanceNorm2d):
	def profile(self, x):
		assert x.dim() in (3, 4), f"input tensor shape of InstanceNorm2d should be [N, C, H, W] or [C, H, W], but got {x.dim()}"
		output, paras, macs = super().profile(x)
		c, h, w = 1, 1, 1
		if x.dim() == 3:
			c, h, w = x.shape
		elif x.dim() == 4:
			_, c, h, w = x.shape

		macs += c * h * w * (4 if self.affine else 2)

		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class InstanceNorm1d(BaseNorm, nn.InstanceNorm1d):
	def profile(self, x):
		assert x.dim() in (2, 3), f"input tensor shape of InstanceNorm1d should be [N, C, L] or [C, L], but got {x.dim()}"
		output, paras, macs = super().profile(x)
		c, ll = 1, 1
		if x.dim() == 2:
			c, ll = x.shape
		elif x.dim() == 3:
			_, c, ll = x.shape

		macs += c * ll * (4 if self.affine else 2)
		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class InstanceNorm3d(BaseNorm, nn.InstanceNorm3d):
	def profile(self, x):
		assert x.dim() in (4, 5), f"input tensor shape of InstanceNorm3d should be [N, C, D, H, W] or [C, D, H, W], but got {x.dim()}"
		output, paras, macs = super().profile(x)
		c, d, h, w = 1, 1, 1, 1
		if x.dim() == 4:
			c, d, h, w = x.shape
		elif x.dim() == 5:
			_, c, d, h, w = x.shape

		macs += c * d * h * w * (4 if self.affine else 2)

		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class GroupNorm(Cfg, Dsp, nn.GroupNorm):
	__slots__ = ["num_groups", "num_channels", "eps", "affine"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [None, None, 1e-5, True]
	_types_ = [int, int, float, bool]

	_cfg_path_ = Entry.Norm

	def __init__(
			self,
			num_groups: int = None,
			num_channels: int = None,
			eps: float = None,
			affine: bool = None
	):
		para_dict = Par.purify(locals())
		super().__init__(**para_dict)
		Par.init_helper(self, para_dict, nn.GroupNorm)

	def profile(self, x):
		assert x.dim() == 4, f"GroupNorm only support input [N C H W], but got dimension {x.dim()}"
		n, c, h, w = x.shape
		output, paras, macs = super().profile(x)
		num_elements = c * h * w
		# calc mean
		macs += num_elements + self.num_groups
		# calc std
		macs += 2 * num_elements + self.num_groups * 2
		# calc normalization value
		macs += num_elements * 2
		# affine
		if self.affine:
			macs += num_elements * 2

		return output, paras, macs


@Entry.register_entry(Entry.Norm)
class LayerNorm2D(GroupNorm):
	def __init__(
			self,
			num_channels: int = None,
			eps: float = None,
			affine: bool = None
	):
		super().__init__(num_groups=1, num_channels=num_channels, eps=eps, affine=affine)


@Entry.register_entry(Entry.Norm)
class LayerNorm(Cfg, Dsp, nn.LayerNorm):
	__slots__ = ["normalized_shape", "eps", "elementwise_affine"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [None, 1e-5, True]
	_types_ = [(int,), float, bool]

	_cfg_path_ = Entry.Norm

	def __init__(
			self,
			normalized_shape: tuple = None,
			eps: float = None,
			elementwise_affine: bool = None
	):
		para_dict = Par.purify(locals())
		super().__init__(**para_dict)
		Par.init_helper(self, para_dict, nn.LayerNorm)

	def profile(self, x):
		assert 2 <= x.dim() <= 4, f"input dim should between 2 and 4, but got {x.dim()}"
		n, c, h, w = 1, 1, 1, 1
		if x.dim() == 2:
			n, c = x.shape
		elif x.dim() == 3:
			n, c, h = x.shape
		elif x.dim() == 4:
			n, c, h, w = x.shape

		output, paras, macs = super().profile(x)
		num_elements = c * h * w
		# calc mean
		macs += num_elements + 1
		# calc std
		macs += 2 * num_elements + 2
		# calc normalization
		macs += 2 * num_elements
		# affine
		if self.elementwise_affine:
			macs += num_elements * 2

		return output, paras, macs
