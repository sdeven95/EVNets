import torch

from . import BaseTransform
import random
import math
import copy

from torchvision import transforms as T
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import numpy as np
from typing import Dict, List, Sequence, Union, Optional

from .builtins import _crop_fn, _resize_fn
from .utils import TsUtil
from utils.type_utils import Par
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Transform)
class RandomHorizontalFlip(BaseTransform):
	__slots__ = ["enable", "p"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 0.5]
	_types_ = [bool, float]

	def __call__(self, data: Dict) -> Dict:
		if random.random() <= self.p:
			img = data["image"]
			width, height = F.get_image_size(img)
			data["image"] = F.hflip(img)

			if "mask" in data:
				mask = data.pop("mask")
				data["mask"] = F.hflip(mask)

			if "box_coordinates" in data:
				boxes = data.pop("box_coordinates")
				boxes[..., 0::2] = width - boxes[..., 2::-2]
				data["box_coordinates"] = boxes

			if "instance_mask" in data:
				assert "instance_coords" in data

				instance_coords = data.pop("instance_coords")
				instance_coords[..., 0::2] = width - instance_coords[..., 2::-2]
				data["instance_coords"] = instance_coords

				instance_masks = data.pop("instance_mask")
				data["instance_mask"] = F.hflip(instance_masks)
		return data

	def __init__(self, p: float = None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Transform)
class RandomRotate(BaseTransform):
	__slots__ = ["enable", "angle", "mask_fill"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 10.0, 0.0]
	_types_ = [bool, float, float]

	def __call__(self, data: Dict) -> Dict:

		data_keys = list(data.keys())
		if "box_coordinates" in data_keys or "instance_mask" in data_keys:
			raise KeyError(f"{self.__class__.__name__} supports only images and masks")

		rand_angle = random.uniform(-self.angle, self.angle)
		img = data.pop("image")
		data["image"] = F.rotate(
			img,
			angle=rand_angle,
			interpolation=F.InterpolationMode.BILINEAR,
			fill=0.0
		)
		if "mask" in data:
			mask = data.pop("mask")
			data["mask"] = F.rotate(
				mask,
				angle=rand_angle,
				interpolation=F.InterpolationMode.NEAREST,
				fill=self.mask_fill,
			)

		return data

	def __init__(
			self,
			angle: float = None,
			mask_fill: float = None,
	):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Transform)
class Resize(BaseTransform):
	__slots__ = ["enable", "size", "interpolation"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, [128, 128], "bilinear"]
	_types_ = [bool, (int, ), str]

	def __init__(
			self,
			size: Union[int, Sequence] = None,
			interpolation: str = None,
	):
		super().__init__(**Par.purify(locals()))
		self.maintain_aspect_ratio = True if isinstance(self.size, int) else False

	def __call__(self, data):
		return _resize_fn(data, size=self.size, interpolation=self.interpolation)


@Entry.register_entry(Entry.Transform)
class CustomCrop(BaseTransform):
	__slots__ = ["enable", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, ]
	_types_ = [bool, ]

	def __call__(self, data: Dict) -> Dict:
		# img_width, img_height = F.get_image_size(data["image"])
		crop_width, crop_height = data["crop_width"], data["crop_height"]
		top, left = data["top"], data["left"]
		return _crop_fn(data=data, top=top, left=left, height=crop_height, width=crop_width)


@Entry.register_entry(Entry.Transform)
class CenterCrop(BaseTransform):
	__slots__ = ["enable", "size"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, [128, 128]]
	_types_ = [bool, (int, )]

	def __init__(self, size: tuple = None):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data: Dict) -> Dict:
		width, height = F.get_image_size(data["image"])
		h, w = self.size if isinstance(self.size, Sequence) else (self.size, self.size)
		i = (height - h) // 2
		j = (width - w) // 2

		return _crop_fn(data=data, top=i, left=j, height=h, width=w)


# randomly resize image in a specified ratio range, defaults to (0.5, 2.0)
@Entry.register_entry(Entry.Transform)
class RandomResize(BaseTransform):
	__slots__ = ["enable", "max_scale_long_edge", "max_scale_short_edge", "min_ratio", "max_ratio", "interpolation"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, None, None, 0.5, 2.0, "bilinear"]
	_types_ = [bool, int, int, float, float, str]

	def __init__(
			self,
			max_scale_long_edge: int = None,
			max_scale_short_edge: int = None,
			min_ratio: float = None,
			max_ratio: float = None,
			interpolation: str = None
	):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data: Dict) -> Dict:
		random_ratio = random.uniform(self.min_ratio, self.max_ratio)

		# compute the size
		width, height = F.get_image_size(data["image"])
		if self.max_scale_long_edge is not None:
			min_hw = min(height, width)
			max_hw = max(height, width)
			scale_factor = (
				min(
					self.max_scale_long_edge / max_hw,
					self.max_scale_short_edge / min_hw,
				)
				* random_ratio
			)
			# resize while maintaining aspect ratio
			new_size = int(math.ceil(height * scale_factor)), int(
				math.ceil(width * scale_factor)
			)
		else:
			new_size = int(math.ceil(height * random_ratio)), int(
				math.ceil(width * random_ratio)
			)
		# new_size should be a tuple of height and width
		return _resize_fn(data, size=new_size, interpolation=self.interpolation)


# random resize the image such that shortest side is between specified minimum and maximum values
@Entry.register_entry(Entry.Transform)
class RandomShortSizeResize(BaseTransform):
	__slots__ = ["enable", "short_side_min", "short_size_max", "interpolation", "max_img_dim"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 64, 128, "bicubic", 96]
	_types_ = [bool, int, int, str, int]

	def __init__(
			self,
			short_side_min: int = None,
			short_size_max: int = None,
			interpolation: str = None,
			max_img_dim: int = None
	):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data: Dict) -> Dict:
		short_side = random.randint(self.short_side_min, self.short_side_max)
		img_w, img_h = data["image"].size
		scale = min(
			short_side / min(img_h, img_w), self.max_img_dim / max(img_h, img_w)
		)
		img_w = int(img_w * scale)
		img_h = int(img_h * scale)
		data = _resize_fn(data, size=(img_h, img_w), interpolation=self.interpolation)
		return data


# randomly blur the input image by GaussianBlur
@Entry.register_entry(Entry.Transform)
class RandomGaussianBlur(BaseTransform):
	__slots__ = ["enable", "p"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 0.5]
	_types_ = [bool, float]

	def __init__(self, p: float = None):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data: Dict) -> Dict:
		if random.random() < self.p:
			img = data.pop("image")
			# radius is the standard deviation of the gaussian kernel
			img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
			data["image"] = img
		return data


@Entry.register_entry(Entry.Transform)
class RandomCrop(BaseTransform):
	"""
	This method randomly crops an image area.

	.. note::
		If the size of input image is smaller than the desired crop size, the input image is first resized
		while maintaining the aspect ratio and then cropping is performed.
	"""
	__slots__ = [
		"enable", "size", "ignore_idx",
		"seg_class_max_ratio", "pad_if_needed", "mask_fill",
	]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
				False, None, 255,
				None, False, 255
			]
	_types_ = [
				bool, (int,), int,
				float, bool, int
			]

	def __init__(
			self,
			size: Union[Sequence, int] = None,
			ignore_idx: Optional[int] = None,
	):
		super().__init__(**Par.purify(locals()))
		self.height, self.width = TsUtil.setup_size(size=size)
		self.if_needed_fn = (
			self._pad_if_needed if self.pad_if_needed else self._resize_if_needed
		)

	# get a random-start-positions crop (target_h * target_w) from the image(img_h * img_w)
	@staticmethod
	def get_params(img_h, img_w, target_h, target_w):
		if img_w == target_w and img_h == target_h:
			return 0, 0, img_h, img_w

		i = random.randint(0, max(0, img_h - target_h))
		j = random.randint(0, max(0, img_w - target_w))
		return i, j, target_h, target_w

	# get a random-start-positions crop based on specified box by adding offset
	@staticmethod
	def get_params_from_box(boxes, img_h, img_w):
		# x, y, w, h
		offset = random.randint(20, 50)
		start_x = max(0, int(round(np.min(boxes[..., 0]))) - offset)
		start_y = max(0, int(round(np.min(boxes[..., 1]))) - offset)
		end_x = min(int(round(np.max(boxes[..., 2]))) + offset, img_w)
		end_y = min(int(round(np.max(boxes[..., 3]))) + offset, img_h)

		return start_y, start_x, end_y - start_y, end_x - start_x

	# get a valid random corp with objects inside
	def get_params_from_mask(self, data, i, j, h, w):
		img_w, img_h = F.get_image_size(data["image"])
		for _ in range(self.num_repeats):
			temp_data = _crop_fn(
				data=copy.deepcopy(data), top=i, left=j, height=h, width=w
			)
			class_labels, cls_count = np.unique(
				np.array(temp_data["mask"]), return_counts=True
			)
			valid_cls_count = cls_count[class_labels != self.ignore_idx]

			if valid_cls_count.size == 0:
				continue

			# compute the ratio of segmentation class with max. pixels to total pixels.
			# If the ratio is less than seg_class_max_ratio, then exit the loop
			total_valid_pixels = np.sum(valid_cls_count)
			max_valid_pixels = np.max(valid_cls_count)
			ratio = max_valid_pixels / total_valid_pixels

			# is it wrong???  ratio > self.seg_class_max_ratio ???
			if len(cls_count) > 1 and ratio < self.seg_class_max_ratio:
				break
			# get a random-start-position crop
			i, j, h, w = self.get_params(
				img_h=img_h, img_w=img_w, target_h=self.height, target_w=self.width
			)
		return i, j, h, w

	# resize to target size
	def _resize_if_needed(self, data: Dict) -> Dict:
		img = data["image"]

		w, h = F.get_image_size(img)
		# resize while maintaining the aspect ratio
		new_size = min(h + max(0, self.height - h), w + max(0, self.width - w))

		return _resize_fn(
			data, size=new_size, interpolation=T.InterpolationMode.BILINEAR
		)

	# pad to target size
	def _pad_if_needed(self, data: Dict) -> Dict:
		img = data.pop("image")

		w, h = F.get_image_size(img)
		new_h = h + max(self.height - h, 0)
		new_w = w + max(self.width - w, 0)

		pad_img = Image.new(img.mode, (new_w, new_h), color=0)
		pad_img.paste(img, (0, 0))
		data["image"] = pad_img

		if "mask" in data:
			mask = data.pop("mask")
			pad_mask = Image.new(mask.mode, (new_w, new_h), color=self.mask_fill)
			pad_mask.paste(mask, (0, 0))
			data["mask"] = pad_mask

		return data

	def __call__(self, data: Dict) -> Dict:
		# box_info
		# if box is defined, confines data to box inside (actually to a random box)
		if "box_coordinates" in data:
			boxes = data.get("box_coordinates")
			# crop the relevant area
			image_w, image_h = F.get_image_size(data["image"])
			box_i, box_j, box_h, box_w = self.get_params_from_box(
				boxes, image_h, image_w
			)
			data = _crop_fn(data, top=box_i, left=box_j, height=box_h, width=box_w)

		# resize data size to target size
		data = self.if_needed_fn(data)

		# get a random crop position
		img_w, img_h = F.get_image_size(data["image"])
		i, j, h, w = self.get_params(
			img_h=img_h, img_w=img_w, target_h=self.height, target_w=self.width
		)

		# get a valid random crop position
		if (
			"mask" in data
			and self.seg_class_max_ratio is not None
			and self.seg_class_max_ratio < 1.0
		):
			i, j, h, w = self.get_params_from_mask(data=data, i=i, j=j, h=h, w=w)

		data = _crop_fn(data=data, top=i, left=j, height=h, width=w)
		return data


@Entry.register_entry(Entry.Transform)
class ToTensor(BaseTransform):
	__slots__ = ["dtype", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["float", ]
	_types_ = [str, ]

	def __init__(self, dtype:str = None):
		super().__init__(**Par.purify(locals()))
		self.img_dtype = torch.float
		if self.dtype in ["half", "float16"]:
			self.img_dtype = torch.float16

	def __call__(self, data: Dict) -> Dict:
		# HWC --> CHW
		img = data["image"]

		if F._is_pil_image(img):
			# convert PIL image to tensor
			img = F.pil_to_tensor(img).contiguous()

		data["image"] = img.to(dtype=self.img_dtype).div(255.0)

		if "mask" in data:
			mask = data.pop("mask")
			mask = np.array(mask)
			if len(mask.shape) > 2 and mask.shape[-1] > 1:
				mask = np.ascontiguousarray(mask.transpose([2, 0, 1]))
			data["mask"] = torch.as_tensor(mask, dtype=torch.long)

		if "box_coordinates" in data:
			boxes = data.pop("box_coordinates")
			data["box_coordinates"] = torch.as_tensor(boxes, dtype=torch.float)

		if "box_labels" in data:
			box_labels = data.pop("box_labels")
			data["box_labels"] = torch.as_tensor(box_labels)

		if "instance_mask" in data:
			assert "instance_coords" in data
			instance_masks = data.pop("instance_mask")
			data["instance_mask"] = instance_masks.to(dtype=torch.long)

			instance_coords = data.pop("instance_coords")
			data["instance_coords"] = torch.as_tensor(
				instance_coords, dtype=torch.float
			)
		return data


# apply a list of transform in a sequential fashion
@Entry.register_entry(Entry.Transform)
class Compose(BaseTransform):
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]

	def __init__(self, img_transforms: List, **kwargs):
		super().__init__(**kwargs)
		self.img_transforms = img_transforms

	def __call__(self, data):
		for t in self.img_transforms:
			data = t(data)
		return data

	def __repr__(self) -> str:
		filter_list = filter(lambda t: not isinstance(t, Identity), self.img_transforms)
		transform_str = ", ".join("\n\t" + str(t) for t in filter_list)
		repr_str = "{}({}\n)".format(self.__class__.__name__, transform_str)
		return repr_str


# apply a list of all or few transform in a random order
@Entry.register_entry(Entry.Transform)
class RandomOrder(BaseTransform):
	__slots__ = ["enable", "apply_k_factor"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 1.0]
	_types_ = [bool, float]

	def __init__(self, img_transforms: List, apply_k_factor: float = None):
		super().__init__(apply_k_factor=apply_k_factor)
		self.img_transforms = img_transforms
		self.keep_t = int(math.ceil(len(self.img_transforms) * self.apply_k_factor))

	def __call__(self, data: Dict) -> Dict:
		random.shuffle(self.transforms)
		for t in self.transforms[: self.keep_t]:
			data = t(data)
		return data


@Entry.register_entry(Entry.Transform)
class SSDCropping(BaseTransform):
	__slots__ = ["enable", "iou_thresholds", "n_trials", "min_aspect_ratio", "max_aspect_ratio"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 40, 0.5, 2.0]
	_types_ = [bool, (float, ), int, float, float]

	def __init__(
			self,
			iou_thresholds: List[float] = None,
			n_trials: int = None,
			min_aspect_ratio: float = None,
			max_aspect_ratio: float = None,
	):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data: Dict) -> Dict:
		if "box_coordinates" in data:
			boxes = data["box_coordinates"]

			# guard against no boxes
			if boxes.shape[0] == 0:
				return data

			image = data["image"]
			labels = data["box_labels"]
			width, height = F.get_image_size(image)

			while True:
				# randomly choose a mode
				min_jaccard_overalp = random.choice(self.iou_thresholds)
				if min_jaccard_overalp == 0.0:
					return data

				for _ in range(self.n_trials):
					new_w = int(random.uniform(0.3 * width, width))
					new_h = int(random.uniform(0.3 * height, height))

					aspect_ratio = new_h / new_w
					if not (
						self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
					):
						continue

					left = int(random.uniform(0, width - new_w))
					top = int(random.uniform(0, height - new_h))

					# convert to integer rect x1,y1,x2,y2
					rect = np.array([left, top, left + new_w, top + new_h])

					# calculate IoU (jaccard overlap) b/t the cropped and gt boxes
					ious = TsUtil.jaccard_numpy(boxes, rect)

					# is min and max overlap constraint satisfied? if not try again
					if ious.max() < min_jaccard_overalp:
						continue

					# keep overlap with gt box IF center in sampled patch
					centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5

					# mask in all gt boxes that above and to the left of centers
					m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

					# mask in all gt boxes that under and to the right of centers
					m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

					# mask in that both m1 and m2 are true
					mask = m1 * m2

					# have any valid boxes? try again if not
					if not mask.any():
						continue

					# if image size is too small, try again
					if (rect[3] - rect[1]) < 100 or (rect[2] - rect[0]) < 100:
						continue

					# cut the crop from the image
					image = F.crop(image, top=top, left=left, width=new_w, height=new_h)

					# take only matching gt boxes
					current_boxes = boxes[mask, :].copy()

					# take only matching gt labels
					current_labels = labels[mask]

					# should we use the box left and top corner or the crop's
					current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
					# adjust to crop (by substracting crop's left,top)
					current_boxes[:, :2] -= rect[:2]

					current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
					# adjust to crop (by substracting crop's left,top)
					current_boxes[:, 2:] -= rect[:2]

					data["image"] = image
					data["box_labels"] = current_labels
					data["box_coordinates"] = current_boxes

					if "mask" in data:
						mask = data.pop("mask")
						data["mask"] = F.crop(
							mask, top=top, left=left, width=new_w, height=new_h
						)

					if "instance_mask" in data:
						assert "instance_coords" in data
						instance_masks = data.pop("instance_mask")
						data["instance_mask"] = F.crop(
							instance_masks,
							top=top,
							left=left,
							width=new_w,
							height=new_h,
						)

						instance_coords = data.pop("instance_coords")
						# should we use the box left and top corner or the crop's
						instance_coords[..., :2] = np.maximum(
							instance_coords[..., :2], rect[:2]
						)
						# adjust to crop (by substracting crop's left,top)
						instance_coords[..., :2] -= rect[:2]

						instance_coords[..., 2:] = np.minimum(
							instance_coords[..., 2:], rect[2:]
						)
						# adjust to crop (by substracting crop's left,top)
						instance_coords[..., 2:] -= rect[:2]
						data["instance_coords"] = instance_coords

					return data
		return data


@Entry.register_entry(Entry.Transform)
class PhotometricDistort(BaseTransform):
	__slots__ = ["enable", "p", "alpha_min", "alpha_max", "beta_min", "beta_max", "gamma_min", "gamma_max", "delta_min", "delta_max"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 0.5, 0.5, 1.5, 0.875, 1.125, 0.5, 1.5, -0.05, 0.05]
	_types_ = [bool, float, float, float, float, float, float, float, float, float]

	def __init__(
			self,
			p: float = None,
			alpha_min: float = None,
			alpha_max: float = None,
			beta_min: float = None,
			beta_max: float = None,
			gamma_min: float = None,
			gamma_max: float = None,
			delta_min: float = None,
			delta_max: float = None
		):
		super().__init__(**Par.purify(locals()))

		self.contrast = T.ColorJitter(contrast=(self.alpha_min, self.alpha_max))
		self.brightness = T.ColorJitter(brightness=(self.beta_min, self.beta_max))
		self.saturation = T.ColorJitter(saturation=(self.gamma_min, self.gamma_max))
		self.hue = T.ColorJitter(hue=(self.delta_min, self.delta_max))

	def _apply_transformations(self, image):
		r = np.random.rand(7)

		if r[0] < self.p:
			image = self.brightness(image)

		contrast_before = r[1] < self.p
		if contrast_before and r[2] < self.p:
			image = self.contrast(image)

		if r[3] < self.p:
			image = self.saturation(image)

		if r[4] < self.p:
			image = self.hue(image)

		if not contrast_before and r[5] < self.p:
			image = self.contrast(image)

		if r[6] < self.p and image.mode != "L":
			# Only permute channels for RGB images
			# [H, W, C] format
			image_np = np.asarray(image)
			n_channels = image_np.shape[2]
			image_np = image_np[..., np.random.permutation(range(n_channels))]
			image = Image.fromarray(image_np)
		return image

	def __call__(self, data: Dict) -> Dict:
		image = data.pop("image")
		data["image"] = self._apply_transformations(image)
		return data


@Entry.register_entry(Entry.Transform)
class BoxPercentCoords(BaseTransform):
	__slots__ = ["enable"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False]
	_types_ = [bool]

	def __call__(self, data):
		if "box_coordinates" in data:
			boxes = data.pop("box_coordinates")
			image = data["image"]
			width, height = F.get_image_size(image)

			boxes = boxes.astype(np.float)

			boxes[..., 0::2] /= width
			boxes[..., 1::2] /= height
			data["box_coordinates"] = boxes

		return data


@Entry.register_entry(Entry.Transform)
class InstanceProcessor(BaseTransform):
	__slots__ = ["enable", "instance_size", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 16, ]
	_types_ = [bool, int, ]

	def __init__(self, instance_size: int = None):
		super().__init__(instance_size=instance_size)
		self.instance_size = TsUtil.setup_size(self.instance_size)

	def __call__(self, data):

		if "instance_mask" in data:
			assert "instance_coords" in data
			instance_masks = data.pop("instance_mask")
			instance_coords = data.pop("instance_coords")
			instance_coords = instance_coords.astype(np.int)

			valid_boxes = (instance_coords[..., 3] > instance_coords[..., 1]) & (
				instance_coords[..., 2] > instance_coords[..., 0]
			)
			instance_masks = instance_masks[valid_boxes]
			instance_coords = instance_coords[valid_boxes]

			num_instances = instance_masks.shape[0]

			resized_instances = []
			for i in range(num_instances):
				# format is [N, H, W]
				instance_m = instance_masks[i]
				box_coords = instance_coords[i]

				instance_m = F.crop(
					instance_m,
					top=box_coords[1],
					left=box_coords[0],
					height=box_coords[3] - box_coords[1],
					width=box_coords[2] - box_coords[0],
				)
				# need to unsqueeze and squeeze to make F.resize work
				instance_m = F.resize(
					instance_m.unsqueeze(0),
					size=self.instance_size,
					interpolation=T.InterpolationMode.NEAREST,
				).squeeze(0)
				resized_instances.append(instance_m)

			if len(resized_instances) == 0:
				resized_instances = torch.zeros(
					size=(1, self.instance_size[0], self.instance_size[1]),
					dtype=torch.long,
				)
				instance_coords = np.array(
					[[0, 0, self.instance_size[0], self.instance_size[1]]]
				)
			else:
				resized_instances = torch.stack(resized_instances, dim=0)

			data["instance_mask"] = resized_instances
			data["instance_coords"] = instance_coords.astype(np.float)
		return data


@Entry.register_entry(Entry.Transform)
class Identity(BaseTransform):
	__slots__ = ["nothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]

	def __call__(self, data):
		return data
