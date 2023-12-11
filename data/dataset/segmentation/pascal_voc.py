import os
import numpy as np

from .. import BaseImageDataset
from utils.type_utils import Opt
from utils.entry_utils import Entry

import data.transform as T

from typing import Optional, List, Tuple, Dict


@Entry.register_entry(Entry.Dataset)
class PascalVOCDataset(BaseImageDataset):
	"""
	Dataset class for the PASCAL VOC 2012 dataset

	The structure of PASCAL VOC dataset should be something like this: ::

		pascal_voc/VOCdevkit/VOC2012/Annotations
		pascal_voc/VOCdevkit/VOC2012/JPEGImages
		pascal_voc/VOCdevkit/VOC2012/SegmentationClass
		pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug_Visualization
		pascal_voc/VOCdevkit/VOC2012/ImageSets
		pascal_voc/VOCdevkit/VOC2012/list
		pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug
		pascal_voc/VOCdevkit/VOC2012/SegmentationObject

	Args:
		is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
		is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False
	"""

	__slots__ = ["use_coco_data", "coco_root_dir"]
	_keys_ = __slots__
	_defaults_ = [False, None]
	_types_ = [bool, str]
	_disp_ = __slots__

	def __init__(
		self,
		is_training: Optional[bool] = None,
		is_evaluation: Optional[bool] = None,
	) -> None:
		super().__init__(is_training=is_training, is_evaluation=is_evaluation)

		voc_root_dir = os.path.join(self.root, "VOC2012")
		voc_list_dir = os.path.join(voc_root_dir, "list")

		coco_data_file = None
		if self.is_training:
			# use the PASCAL VOC 2012 train data with augmented data
			data_file = os.path.join(voc_list_dir, "train_aug.txt")
			# use coco data
			if self.use_coco_data and self.coco_root_dir is not None:
				coco_data_file = os.path.join(self.coco_root_dir, "train_2017.txt")
				assert os.path.isfile(
					coco_data_file
				), "COCO data file does not exist at: {}".format(self.coco_root_dir)
		else:
			data_file = os.path.join(voc_list_dir, "val.txt")

		# get image paths and mask paths
		self.images = []
		self.masks = []
		with open(data_file, "r") as lines:
			for line in lines:
				line_split = line.split(" ")
				rgb_img_loc = voc_root_dir + os.sep + line_split[0].strip()
				mask_img_loc = voc_root_dir + os.sep + line_split[1].strip()
				assert os.path.isfile(
					rgb_img_loc
				), "RGB file does not exist at: {}".format(rgb_img_loc)
				assert os.path.isfile(
					mask_img_loc
				), "Mask image does not exist at: {}".format(rgb_img_loc)
				self.images.append(rgb_img_loc)
				self.masks.append(mask_img_loc)

		# if you want to use Coarse data for training
		if self.is_training and coco_data_file is not None:
			with open(coco_data_file, "r") as lines:
				for line in lines:
					line_split = line.split(" ")
					rgb_img_loc = self.coco_root_dir + os.sep + line_split[0].rstrip()
					mask_img_loc = self.coco_root_dir + os.sep + line_split[1].rstrip()
					assert os.path.isfile(rgb_img_loc)
					assert os.path.isfile(mask_img_loc)
					self.images.append(rgb_img_loc)
					self.masks.append(mask_img_loc)
		self.ignore_label = 255
		self.bgrnd_idx = 0

		self.n_records = len(self.images)
		Opt.set_by_cfg_path(Entry.SegmentationModel, "n_classes", len(self.class_names()))

	def _training_transforms(self, size: tuple):
		random_gaussian_blur, photometric_distort, random_rotate, random_order =\
			Opt.get_argument_values(
				pth_tpl=Entry.Transform + ".{}",
				keys=(
					"RandomGaussianBlur.enable",
					"PhotometricDistort.enable",
					"RandomRotate.enable",
					"RandomOrder.enable",
				),
				defaults=(False, False, False, False)
			)

		first_aug = T.RandomShortSizeResize()
		aug_list = [
			T.RandomHorizontalFlip(),
			T.RandomCrop(size=size, ignore_idx=self.ignore_label),
			T.RandomGaussianBlur() if random_gaussian_blur else T.Identity(),
			T.PhotometricDistort() if photometric_distort else T.Identity(),
			T.RandomRotate() if random_rotate else T.Identity(),
		]

		if random_order:
			new_aug_list = [
				first_aug,
				T.RandomOrder(img_transforms=aug_list),
				T.ToTensor(),
			]
			return T.Compose(img_transforms=new_aug_list)
		else:
			aug_list.insert(0, first_aug)
			aug_list.append(T.ToTensor())
			return T.Compose(img_transforms=aug_list)

	def _validation_transforms(self, size: tuple):
		return T.Compose(img_transforms=[
			T.Resize(),
			T.ToTensor()
		])

	def _evaluation_transforms(self, size: tuple):
		aug_list = []
		if Opt.get_by_cfg_path(Entry.SegmentationModel, "resize_input_image_on_evaluation"):
			# we want to resize while maintaining aspect ratio. So, we pass img_size argument to resize function
			aug_list.append(T.Resize(size=min(size)))

		aug_list.append(T.ToTensor())
		return T.Compose(img_transforms=aug_list)

	def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
		crop_size_h, crop_size_w, img_index = batch_indexes_tup
		crop_size = (crop_size_h, crop_size_w)

		if self.is_training:
			_transform = self._training_transforms(size=crop_size)
		elif self.is_evaluation:
			_transform = self._evaluation_transforms(size=crop_size)
		else:
			_transform = self._validation_transforms(size=crop_size)

		img = self.read_image_pil(self.images[img_index])
		mask = self.read_mask_pil(self.masks[img_index])

		data = {"image": img}
		if not self.is_evaluation:
			data["mask"] = mask

		data = _transform(data)

		if self.is_evaluation:
			# for evaluation purposes, resize only the input and not mask
			data["mask"] = self.convert_mask_to_tensor(mask)

		data["label"] = data["mask"]
		del data["mask"]

		if self.is_evaluation:
			im_width, im_height = img.size
			img_name = self.images[img_index].split(os.sep)[-1].replace("jpg", "png")
			data["file_name"] = img_name
			data["im_width"] = im_width
			data["im_height"] = im_height

		return data

	def __len__(self):
		return len(self.images)

	def __repr__(self):
		from ...sampler.utils import SplUtil

		im_h, im_w = SplUtil.image_size_from_opts()

		if self.is_training:
			transforms_str = self._training_transforms(size=(im_h, im_w))
		elif self.is_evaluation:
			transforms_str = self._evaluation_transforms(size=(im_h, im_w))
		else:
			transforms_str = self._validation_transforms(size=(im_h, im_w))

		return super()._repr_by_line() + "\nTransform details:\n" + repr(transforms_str)

	# color palette
	@staticmethod
	def color_palette():
		color_codes = [
			[0, 0, 0],
			[128, 0, 0],
			[0, 128, 0],
			[128, 128, 0],
			[0, 0, 128],
			[128, 0, 128],
			[0, 128, 128],
			[128, 128, 128],
			[64, 0, 0],
			[192, 0, 0],
			[64, 128, 0],
			[192, 128, 0],
			[64, 0, 128],
			[192, 0, 128],
			[64, 128, 128],
			[192, 128, 128],
			[0, 64, 0],
			[128, 64, 0],
			[0, 192, 0],
			[128, 192, 0],
			[0, 64, 128],
		]

		color_codes = np.asarray(color_codes).flatten()
		return list(color_codes)

	# class names
	@staticmethod
	def class_names() -> List:
		return [
			"background",
			"aeroplane",
			"bicycle",
			"bird",
			"boat",
			"bottle",
			"bus",
			"car",
			"cat",
			"chair",
			"cow",
			"diningtable",
			"dog",
			"horse",
			"motorbike",
			"person",
			"potted_plant",
			"sheep",
			"sofa",
			"train",
			"tv_monitor",
		]
