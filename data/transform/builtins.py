from . import BaseTransform, INTERPOLATION_MODE_MAP
from torchvision import transforms as T
from torchvision.transforms import functional as F
from data.constants import DataConstants
import numpy as np
from typing import Sequence, Dict, Union, Optional, List
from utils.type_utils import Par
from utils.entry_utils import Entry


def _crop_fn(data: dict, top: int, left: int, height: int, width: int) -> dict:
	"""Helper function for cropping"""
	img = data["image"]
	data["image"] = F.crop(img, top=top, left=left, height=height, width=width)

	if "mask" in data:
		mask = data.pop("mask")
		data["mask"] = F.crop(mask, top=top, left=left, height=height, width=width)

	if "box_coordinates" in data:
		boxes = data.pop("box_coordinates")

		area_before_cropping = (boxes[..., 2] - boxes[..., 0]) * (
			boxes[..., 3] - boxes[..., 1]
		)

		# box part on cropped image
		boxes[..., 0::2] = np.clip(boxes[..., 0::2] - left, a_min=0, a_max=left + width)
		boxes[..., 1::2] = np.clip(boxes[..., 1::2] - top, a_min=0, a_max=top + height)

		area_after_cropping = (boxes[..., 2] - boxes[..., 0]) * (
			boxes[..., 3] - boxes[..., 1]
		)

		# remained percent after cropping
		area_ratio = area_after_cropping / (area_before_cropping + 1)

		# keep the boxes whose area is at least 20% of the area before cropping
		keep = area_ratio >= 0.2

		box_labels = data.pop("box_labels")

		data["box_coordinates"] = boxes[keep]
		data["box_labels"] = box_labels[keep]

	if "instance_mask" in data:
		assert "instance_coords" in data

		instance_masks = data.pop("instance_mask")
		data["instance_mask"] = F.crop(
			instance_masks, top=top, left=left, height=height, width=width
		)

		instance_coords = data.pop("instance_coords")
		instance_coords[..., 0::2] = np.clip(
			instance_coords[..., 0::2] - left, a_min=0, a_max=left + width
		)
		instance_coords[..., 1::2] = np.clip(
			instance_coords[..., 1::2] - top, a_min=0, a_max=top + height
		)
		data["instance_coords"] = instance_coords

	return data


def _resize_fn(
	data: Dict,
	size: Union[Sequence, int],
	interpolation: Optional[T.InterpolationMode or str] = T.InterpolationMode.BILINEAR,
) -> Dict:
	"""Helper function for resizing"""
	img = data["image"]

	w, h = F.get_image_size(img)

	if isinstance(size, Sequence) and len(size) == 2:
		size_h, size_w = size[0], size[1]
	elif isinstance(size, int):
		if (w <= h and w == size) or (h <= w and h == size):
			return data

		if w < h:
			size_h = int(size * h / w)
			size_w = size
		else:
			size_w = int(size * w / h)
			size_h = size
	else:
		raise TypeError(
			"Supported size args are int or tuple of length 2. Got inappropriate size arg: {}".format(
				size
			)
		)

	if isinstance(interpolation, str):
		interpolation = INTERPOLATION_MODE_MAP[interpolation]

	data["image"] = F.resize(
		img=img, size=[size_h, size_w], interpolation=interpolation
	)

	if "mask" in data:
		mask = data.pop("mask")
		resized_mask = F.resize(
			img=mask,
			size=[size_h, size_w],
			interpolation=T.InterpolationMode.NEAREST
		)
		data["mask"] = resized_mask

	if "box_coordinates" in data:
		boxes = data.pop("box_coordinates")
		boxes[:, 0::2] *= 1.0 * size_w / w
		boxes[:, 1::2] *= 1.0 * size_h / h
		data["box_coordinates"] = boxes

	if "instance_mask" in data:
		assert "instance_coords" in data

		instance_masks = data.pop("instance_mask")

		resized_instance_masks = F.resize(
			img=instance_masks,
			size=[size_h, size_w],
			interpolation=T.InterpolationMode.NEAREST,
		)
		data["instance_mask"] = resized_instance_masks

		instance_coords = data.pop("instance_coords")
		instance_coords = instance_coords.astype(np.float)
		instance_coords[..., 0::2] *= 1.0 * size_w / w
		instance_coords[..., 1::2] *= 1.0 * size_h / h
		data["instance_coords"] = instance_coords

	return data


@Entry.register_entry(Entry.Transform)
class Normalize(BaseTransform, T.Normalize):
	__slots__ = ["enable", "mean", "std", "inplace"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, DataConstants.IMAGENET_DEFAULT_MEAN, DataConstants.IMAGENET_DEFAULT_STD, False]
	_types_ = [bool, (float, ), (float, ), bool]

	def __init__(self, mean=None, std=None, inplace=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Transform)
class RandomCropResize(BaseTransform, T.RandomResizedCrop):
	__slots__ = ["enable", "size", "scale", "ratio", "interpolation", ]  #  "antialias"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, [128, 128], [0.08, 1.0], [3.0 / 4.0, 4.0 / 3.0], "bilinear", ]  #  "warn"],
	_types_ = [bool, (int, ), (float, ), (float, ), str, ]  # str]

	def __init__(
		self,
		size: Sequence[int] = None,
		scale: Sequence[float] = None,
		ratio: Sequence[float] = None,
		interpolation: str = None,
		# antialias: Optional[Union[str, bool]] = None,
	):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data):
		img = data["image"]
		i, j, h, w = super().get_params(img=img, scale=self.scale, ratio=self.ratio)
		data = _crop_fn(data=data, top=i, left=j, height=h, width=w)
		return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)


@Entry.register_entry(Entry.Transform)
class AutoAugment(BaseTransform, T.AutoAugment):
	__slots__ = ["enable", "policy", "interpolation", "fill"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, "imagenet", "bilinear", None],
	_types_ = [bool, str, str, (float, )]

	def __init__(
		self,
		policy: str = None,
		interpolation: str = None,
		fill: Optional[List[float]] = None,
	) -> None:
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Transform)
class RandAugment(BaseTransform, T.RandAugment):
	__slots__ = ["enable", "num_ops", "magnitude", "num_magnitude_bins", "interpolation", "fill"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 2, 9, 31, "nearest", None]
	_types_ = [bool, int, int, int, str, (float, )]

	def __init__(
		self,
		num_ops: int = None,
		magnitude: int = None,
		num_magnitude_bins: int = None,
		interpolation: str = None,
		fill: Optional[List[float]] = None,
	) -> None:
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Transform)
class ColorJitter(BaseTransform, T.ColorJitter):
	__slots__ = ["enable", "brightness", "contrast", "saturation", "hue"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 0, 0, 0, 0]
	_types_ = [bool, float, float, float, float]

	def __init__(
		self,
		brightness: Union[float, Sequence[float]] = None,
		contrast: Union[float, Sequence[float]] = None,
		saturation: Union[float, Sequence[float]] = None,
		hue: Union[float, Sequence[float]] = None,
	) -> None:
		super().__init__(**Par.purify(locals()))


# randomly select a region in a tensor and erases its pixel
@Entry.register_entry(Entry.Transform)
class RandomErasing(BaseTransform, T.RandomErasing):
	__slots__ = ["enable", "p", "scale", "ratio", "value", "inplace"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 0.5, [0.02, 0.33], [0.3, 3.3], 0, False]
	_types_ = [bool, float, (float, ), (float, ), float, bool]

	def __init__(
			self,
			p: float = None,
			scale: Sequence[float] = None,
			ratio: Sequence[float] = None,
			value: int = None,
			inplace: bool = None):
		super().__init__(**Par.purify(locals()))
