from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry
from utils import Util

from common import Dev

from torch.utils.data import Dataset
from utils.tensor_utils import Tsr
from PIL import Image
from typing import Union
import io
import psutil
import torch
import numpy as np
import cv2


class BaseImageDataset(Cfg, Dsp, Dataset):
	__slots__ = [
				"root_train", "train_index_file", "train_index_offset",
				"root_val", "val_index_file", "val_index_offset",
				"root_eval", "eval_index_file", "eval_index_offset",
				"workers", "dali_workers", "persistent_workers", "pin_memory", "prefetch_factor",
				"img_dtype", "cache_images_on_ram", "cache_limit",
				"decode_data_on_gpu", "sampler_type",
				"is_training", "is_evaluation",
				"n_records", "n_classes", "root",
			]
	_cfg_path_ = Entry.Dataset
	_keys_ = __slots__[:-3]
	_defaults_ = [
				"", "", 0,
				"", "", 0,
				"", "", 0,
				None, -1, False, False, 2,
				"float", False, 80.0,
				False, "batch",
				True, False,
			]
	_types_ = [
				str, str, int,
				str, str, int,
				str, str, int,
				int, int, bool, bool, int,
				str, bool, float,
				bool, str,
				bool, bool,
			]

	_disp_ = __slots__

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		Dataset.__init__(self)

		self.root = self.root_train if self.is_training else (self.root_eval if self.is_evaluation else self.root_val)

		device = Util.get_device()
		use_cuda = self.decode_data_on_gpu and "cuda" in device.type
		if use_cuda:
			assert not self.pin_memory, \
				"For loading images on GPU, --dataset.pin-memory should be disabled."
		self.device = device if use_cuda else torch.device("cpu")

		self.cached_data = dict() if self.cache_images_on_ram and self.is_training else None
		if self.cached_data:
			assert self.persistent_workers, \
				"For caching, --dataset.persistent-workers should be enabled."

	def __getitem__(self, item):
		raise NotImplementedError

	def _get_training_transforms(self, *args, **kwargs):
		raise NotImplementedError

	def _get_validation_transforms(self, *args, **kwargs):
		raise NotImplementedError

	def _get_evaluation_transforms(self, *args, **kwargs):
		raise NotImplementedError

	def read_image_pil(self, path):
		def convert_to_rgb(inp_data: Union[str, io.BytesIO]):
			try:
				rgb_img = Image.open(inp_data).convert("RGB")
			except Exception as e:
				print("\nImage File: {}\nError:{}\n".format(inp_data, repr(e)))
				rgb_img = None
			return rgb_img

		if self.cached_data is not None:
			# code for caching data on RAM
			used_memory = float(psutil.virtual_memory().percent)

			# if image is in cache data, get it from cache
			if path in self.cached_data:
				img_byte = self.cached_data[path]
			elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
				# image is not present in cache and RAM usage is less than the threshold, add to cache
				with open(path, "rb") as bin_file:
					bin_file_data = bin_file.read()
					img_byte = io.BytesIO(bin_file_data)
					self.cached_data[path] = img_byte
			else:
				# read image to RAM
				with open(path, "rb") as bin_file:
					bin_file_data = bin_file.read()
					img_byte = io.BytesIO(bin_file_data)  # in-memory data
			img = convert_to_rgb(img_byte)
		else:
			img = convert_to_rgb(path)

		return img

	@staticmethod
	def read_mask_pil(path):
		try:
			mask = Image.open(path)
			assert mask.mode == "L", f"Mask mode should be L. Got: {mask.mode}"
			return mask
		except Exception:
			return None

	@staticmethod
	def read_mask_opencv(path: str):
		return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	@staticmethod
	def convert_mask_to_tensor(mask):
		mask = np.array(mask)
		if len(mask.shape) > 2 and mask.shape[-1] > 1:
			mask = np.ascontiguousarray(mask.transpose([2, 0, 1]))
		return torch.as_tensor(mask, dtype=torch.long)

	@staticmethod
	def adjust_mask_value():
		return 0

	def to_device(self, x):
		return Tsr.move_to_device(x, device=self.device)


# from .classification.food import FoodDataset
# from .classification.imagenet import ImagenetDataset
# from .detection.coco_ssd import COCODetectionSSD
# from .segmentation.ade20k import ADE20KDataset
# from .segmentation.coco_segmentation import COCODataset
# from .segmentation.pascal_voc import PascalVOCDataset
# from .video_classification.kinetics import KineticsDataset
