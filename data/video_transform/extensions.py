import random
import math
import torch
from torch.nn import functional as F

from typing import Sequence, Dict, Union, Tuple, List, Optional

from utils.logger import Logger
from utils.entry_utils import Entry

from . import VideoBaseTransform
from ..transform.utils import TsUtil


SUPPORTED_PYTORCH_INTERPOLATIONS = ["nearest", "bilinear", "bicubic"]


def _check_rgb_video_tensor(clip):
	if not isinstance(clip, torch.FloatTensor) or clip.dim() != 4:
		Logger.error(
			"Video clip is either not an instance of FloatTensor or it is not a 4-d tensor (NCHW or CNHW)"
		)


# crop image and mask
def _crop_fn(data: Dict, i: int, j: int, h: int, w: int):
	img = data["image"]
	if not isinstance(img, torch.Tensor) and img.dim() != 4:
		Logger.error(
			"Cropping requires 4-d tensor of shape NCHW or CNHW. Got {}-dimensional tensor".format(
				img.dim()
			)
		)

	crop_image = img[..., i: i + h, j: j + w]
	data["image"] = crop_image

	mask = data.get("mask", None)
	if mask is not None:
		crop_mask = mask[..., i: i + h, j: j + w]
		data["mask"] = crop_mask
	return data


# resize image and mask
# if pass int as a size to keep aspect ratio
def _resize_fn(data: Dict, size: Union[Sequence, int], interpolation: Optional[str] = "bilinear"):
	img = data["image"]

	# if target height size and width size are both given
	if isinstance(size, Sequence) and len(size) == 2:
		size_h, size_w = size[0], size[1]
	# if int parameter is given
	elif isinstance(size, int):
		# if image size is less than target size, return original image
		h, w = img.shape[-2:]
		if (w <= h and w == size) or (h <= w and h == size):
			return data

		# keep image aspect ratio
		if w < h:
			size_h = int(size * h / w)
			size_w = size
		else:
			size_w = int(size * w / h)
			size_h = size
	else:
		raise TypeError(f"Supported size args are int or tuple of length 2. Got inappropriate size arg: {size}")

	# resize image by interpolation
	img = F.interpolate(
		input=img,
		size=(size_w, size_h),
		mode=interpolation,
		align_corners=True if interpolation != "nearest" else None,
	)
	data["image"] = img

	# resize mask by "nearest" interpolation
	mask = data.get("mask", None)
	if mask is not None:
		mask = F.interpolate(input=mask, size=(size_w, size_h), mode="nearest")
		data["mask"] = mask

	return data


@Entry.register_entry(Entry.VideoTransform)
class ToTensor(VideoBaseTransform):
	"""
	This method converts an image into a tensor.

	.. note::
		We do not perform any mean-std normalization. If mean-std normalization is desired, please modify this class.
	"""
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]

	def __call__(self, data: Dict) -> Dict:
		# [C, N, H, W]
		clip = data["image"]
		# convert numpy format to torch tensor
		if not isinstance(clip, torch.Tensor):
			clip = torch.from_numpy(clip)
		clip = clip.float()

		# normalize between 0 and 1
		clip = torch.div(clip, 255.0)
		data["image"] = clip
		return data


@Entry.register_entry(Entry.VideoTransform)
class Compose(VideoBaseTransform):
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]

	def __init__(self, video_transforms: List) -> None:
		super().__init__()
		self.video_transforms = video_transforms

	def __call__(self, data: Dict) -> Dict:
		for t in self.video_transforms:
			data = t(data)
		return data

	def __repr__(self) -> str:
		transform_str = ", ".join("\n\t\t\t" + str(t) for t in self.video_transforms)
		repr_str = "{}({})".format(self.__class__.__name__, transform_str)
		return repr_str


@Entry.register_entry(Entry.VideoTransform)
class Resize(VideoBaseTransform):
	"""
	This class implements resizing operation.

	.. note::
	Two possible modes for resizing.
	1. Resize while maintaining aspect ratio. To enable this option, pass int as a size
	2. Resize to a fixed size. To enable this option, pass a tuple of height and width as a size
	"""
	__slots__ = ["enable", "size", "interpolation"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, None, "bilinear"]
	_types_ = [bool, (int,), str]

	def __init__(
			self,
			size: Sequence or int = None,
			interpolation: str = None
	) -> None:
		super().__init__(size=size, interpolation=interpolation)

	def __call__(self, data: Dict) -> Dict:
		return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)


@Entry.register_entry(Entry.VideoTransform)
class CenterCrop(VideoBaseTransform):
	"""
	This class implements center cropping method.

	.. note::
		This class assumes that the input size is greater than or equal to the desired size.
	"""
	__slots__ = ["enable", "size"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, None]
	_types_ = [bool, str]

	def __init__(self, size: Sequence or int) -> None:
		super().__init__()
		if isinstance(size, Sequence) and len(size) == 2:
			self.height, self.width = size[0], size[1]
		elif isinstance(size, Sequence) and len(size) == 1:
			self.height = self.width = size[0]
		elif isinstance(size, int):
			self.height = self.width = size
		else:
			Logger.error("Scale should be either an int or tuple of ints")

	def __call__(self, data: Dict) -> Dict:
		height, width = data["image"].shape[-2:]
		i = (height - self.height) // 2
		j = (width - self.width) // 2
		return _crop_fn(data=data, i=i, j=j, h=self.height, w=self.width)


@Entry.register_entry(Entry.VideoTransform)
class RandomCrop(VideoBaseTransform):
	"""
	This method randomly crops a video area.

	.. note::
		This class assumes that the input video size is greater than or equal to the desired size.
	"""
	__slots__ = ["enable", "size"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, None]
	_types_ = [bool, (int, )]

	def __init__(self, size: Sequence or int = None) -> None:
		super().__init__(size=size)
		self.size = TsUtil.setup_size(size=self.size)

	# get random position
	def get_params(self, height, width) -> Tuple[int, int, int, int]:
		th, tw = self.size

		if width == tw and height == th:
			return 0, 0, height, width

		i = random.randint(0, height - th)
		j = random.randint(0, width - tw)
		return i, j, th, tw

	def __call__(self, data: Dict) -> Dict:
		clip = data["image"]
		height, width = clip.shape[-2:]
		i, j, h, w = self.get_params(height=height, width=width)
		return _crop_fn(data=data, i=i, j=j, h=h, w=w)


@Entry.register_entry(Entry.VideoTransform)
class RandomHorizontalFlip(VideoBaseTransform):
	"""
	This class implements random horizontal flipping method
	"""
	__slots__ = ["enable", "p"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 0.5]
	_types_ = [bool, float]

	def __call__(self, data: Dict) -> Dict:
		if random.random() <= self.p:
			clip = data["image"]
			clip = torch.flip(clip, dims=[-1])
			data["image"] = clip

			mask = data.get("mask", None)
			if mask is not None:
				mask = torch.flip(mask, dims=[-1])
				data["mask"] = mask

		return data


@Entry.register_entry(Entry.VideoTransform)
class RandomCropResize(VideoBaseTransform):
	"""
	This class crops a random portion of an image and resize it to a given size.
	note: first cropping, then resizing
	"""
	__slots__ = ["enable", "interpolation", "scale", "aspect_ratio"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, "bilinear", (0.08, 1.0), (3.0 / 4.0, 4.0 / 3.0)]
	_types_ = [bool, str, (int,), (float, )]

	def __init__(self, size: Sequence or int = None) -> None:
		super().__init__(size=size)
		self.size = TsUtil.setup_size(size=self.size)

		# check scale
		if not isinstance(self.scale, Sequence) or (
			isinstance(self.scale, Sequence)
			and len(self.scale) != 2
			and 0.0 <= self.scale[0] < self.scale[1]
		):
			Logger.error(
				"--video-augmentation.random-resized-crop.scale should be a tuple of length 2 "
				"such that 0.0 <= scale[0] < scale[1]. Got: {}".format(self.scale)
			)

		# check ratio
		if not isinstance(self.ratio, Sequence) or (
			isinstance(self.ratio, Sequence)
			and len(self.ratio) != 2
			and 0.0 < self.ratio[0] < self.ratio[1]
		):
			Logger.error(
				"--video-augmentation.random-resized-crop.aspect-ratio should be a tuple of length 2 "
				"such that 0.0 < ratio[0] < ratio[1]. Got: {}".format(self.ratio)
			)

		self.ratio = (round(self.ratio[0], 3), round(self.ratio[1], 3))

	# get random position for cropping image
	# scale for selecting random area, i.e. tar_area = ori_area * random_scale
	# ratio for generation a random number r, then w = sqrt(tar_area * r) and h = sqrt(tar_area / r)
	def get_params(self, height: int, width: int) -> (int, int, int, int):
		area = height * width  # original area
		for _ in range(10):
			target_area = random.uniform(*self.scale) * area  # random area

			# for getting h = 2, set scale to (1.0, 1.0),
			# for getting w > h, set scale to (3, 3) ( greater than one),
			# for getting w < h, set scale to (0.75, 0.75) ( less than one)
			# default to (0.75, 0.75)
			# aspect ratio = exp( random(log(ratio[0]), log(ratio[1]) )
			# ratio = (0.1, 0.9) for generating a random number using uniform
			log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))  # (log 3/4, log 3/4)
			aspect_ratio = math.exp(random.uniform(*log_ratio))   # 3/4

			w = int(round(math.sqrt(target_area * aspect_ratio)))   # sqrt(target_area * 3/4)
			h = int(round(math.sqrt(target_area / aspect_ratio)))   # sqrt(target_area / 3/4)

			# if we get right w and h successfully, return random position
			if 0 < w <= width and 0 < h <= height:
				i = random.randint(0, height - h)
				j = random.randint(0, width - w)
				return i, j, h, w

		# if we can't get right position successfully, just use center crop
		# Fallback to central crop
		in_ratio = (1.0 * width) / height  # original ratio ( w/h )
		# get right h and w
		if in_ratio < min(self.ratio):
			w = width
			h = int(round(w / min(self.ratio)))
		elif in_ratio > max(self.ratio):
			h = height
			w = int(round(h * max(self.ratio)))
		else:  # whole image
			w = width
			h = height
		# calculate position
		i = (height - h) // 2
		j = (width - w) // 2
		return i, j, h, w

	def __call__(self, data: Dict) -> Dict:
		clip = data["image"]
		_check_rgb_video_tensor(clip=clip)

		height, width = clip.shape[-2:]

		# crop image by random position
		i, j, h, w = self.get_params(height=height, width=width)
		data = _crop_fn(data=data, i=i, j=j, h=h, w=w)

		# resize cropped image
		return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

	def __repr__(self) -> str:
		return "{}(scale={}, ratio={}, interpolation={})".format(
			self.__class__.__name__, self.scale, self.ratio, self.interpolation
		)


@Entry.register_entry(Entry.VideoTransform)
class RandomShortSizeResizeCrop(VideoBaseTransform):
	"""
	This class first randomly resizes the input video such that shortest side is between specified minimum and
	maximum values, adn then crops a desired size video.
	note: first resizing, then cropping

	.. note::
		This class assumes that the video size after resizing is greater than or equal to the desired size.
	"""
	__slots__ = ["enable", "interpolation", "short_side_min", "short_side_max", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, "bilinear", None, None]
	_types_ = [bool, str, int, int]

	def __init__(self, size: Sequence or int = None) -> None:
		super().__init__(size=size)

	# get random position
	def get_params(self, height, width) -> Tuple[int, int, int, int]:
		th, tw = self.size

		if width == tw and height == th:
			return 0, 0, height, width

		i = random.randint(0, height - th)
		j = random.randint(0, width - tw)
		return i, j, th, tw

	def __call__(self, data: Dict) -> Dict:
		short_size = random.randint(self.short_side_min, self.short_side_max)
		# resize the video so that shorter side is short_size
		data = _resize_fn(data, size=short_size, interpolation=self.interpolation)

		clip = data["image"]
		_check_rgb_video_tensor(clip=clip)
		height, width = clip.shape[-2:]
		i, j, h, w = self.get_params(height=height, width=width)
		# crop the video
		return _crop_fn(data=data, i=i, j=j, h=h, w=w)
