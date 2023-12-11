from . import BaseTransform
from utils.type_utils import Par
from utils.entry_utils import Entry

from torchvision.transforms import functional as F
import torch.nn.functional as F_torch

import torch
import math


# Copied from PyTorch Torchvision
# batch * alpha + shuffle batch(dim=0) * (1 - alpha)
# i.e. construct image pairs and merge two images' data and labels to generate one image sample
@Entry.register_entry(Entry.Transform)
class RandomMixup(BaseTransform):
	__slots__ = ["enable", "alpha", "p", "inplace", "num_classes"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 1.0, 0.5, False, None]
	_types_ = [bool, float, float, bool, int]

	def __init__(
			self,
			alpha: float = None,
			p: float = None,
			inplace: bool = None,
			num_classes: int = None
	):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data):
		if torch.rand(1).item() >= self.p:
			return data

		image_tensor, target_tensor = data.pop("image"), data.pop("label")

		# check tensor dimensions and dtypes
		assert image_tensor.ndim == 4, f"Batch ndim should be 4. Got {image_tensor.ndim}"
		assert target_tensor.ndim == 1, f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
		assert image_tensor.is_floating_point(), f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
		assert target_tensor.dtype == torch.int64, f"Target dtype should be torch.int64. Got {target_tensor.dtype}"

		if not self.inplace:
			image_tensor = image_tensor.clone()
			target_tensor = target_tensor.clone()

		# convert target_tensor by one hot
		if target_tensor.ndim == 1:
			target_tensor = F_torch.one_hot(
				target_tensor, num_classes=self.num_classes
			).to(dtype=image_tensor.dtype)

		# It's faster to roll the batch by one instead of shuffling it to create image pairs
		batch_rolled = image_tensor.roll(1, 0)
		target_rolled = target_tensor.roll(1, 0)

		# Implemented as on mixup paper, page 3.
		lambda_param = float(
			torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
		)
		batch_rolled.mul_(1.0 - lambda_param)
		image_tensor.mul_(lambda_param).add_(batch_rolled)

		target_rolled.mul_(1.0 - lambda_param)
		target_tensor.mul_(lambda_param).add_(target_rolled)

		data["image"] = image_tensor
		data["label"] = target_tensor

		return data


@Entry.register_entry(Entry.Transform)
# for each image in batch, copy a random region from paired image and merge label with pair image's label simultaneously
class RandomCutmix(BaseTransform):
	__slots__ = ["enable", "alpha", "p", "inplace", "num_classes"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [False, 1.0, 0.5, False, None]
	_types_ = [bool, float, float, bool, int]

	def __init__(
			self,
			alpha: float = None,
			p: float = None,
			inplace: bool = None,
			num_classes: int = None
	):
		super().__init__(**Par.purify(locals()))

	def __call__(self, data):
		if torch.rand(1).item() >= self.p:
			return data

		image_tensor, target_tensor = data.pop("image"), data.pop("label")

		# check tensor dimensions and dtypes
		assert image_tensor.ndim == 4, f"Batch ndim should be 4. Got {image_tensor.ndim}"
		assert target_tensor.ndim == 1, f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
		assert image_tensor.is_floating_point(), f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
		assert target_tensor.dtype == torch.int64, f"Target dtype should be torch.int64. Got {target_tensor.dtype}"

		if not self.inplace:
			image_tensor = image_tensor.clone()
			target_tensor = target_tensor.clone()

		# Convert target tensor by one hot
		if target_tensor.ndim == 1:
			target_tensor = F_torch.one_hot(
				target_tensor, num_classes=self.num_classes
			).to(dtype=image_tensor.dtype)

		# It's faster to roll the batch by one instead of shuffling it to create image pairs
		batch_rolled = image_tensor.roll(1, 0)
		target_rolled = target_tensor.roll(1, 0)

		# Implemented as on cutmix paper, page 12 (with minor corrections on typos).
		lambda_param = float(
			torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
		)

		W, H = F.get_image_size(image_tensor)

		# get a random point
		r_x = torch.randint(W, (1,))
		r_y = torch.randint(H, (1,))

		# get two random distances along width & height respectively, about 0.5Width & 0.5Height
		r = 0.5 * math.sqrt(1.0 - lambda_param)  # around 0.5
		r_w_half = int(r * W)  # about half of width
		r_h_half = int(r * H)  # about half of height

		x1 = int(torch.clamp(r_x - r_w_half, min=0))  # random left
		y1 = int(torch.clamp(r_y - r_h_half, min=0))  # random top
		x2 = int(torch.clamp(r_x + r_w_half, max=W))  # random right
		y2 = int(torch.clamp(r_y + r_h_half, max=H))  # random bottom

		image_tensor[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]  # copy one region from paired image
		lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

		# mix label
		target_rolled.mul_(1.0 - lambda_param)
		target_tensor.mul_(lambda_param).add_(target_rolled)

		data["image"] = image_tensor
		data["label"] = target_tensor

		return data
