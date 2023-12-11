from .. import BaseImageDataset
from utils.logger import Logger
from utils.type_utils import Opt
from utils.entry_utils import Entry

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from pycocotools.coco import COCO

from ... import transform as T


class COCODetection(BaseImageDataset):
	__slots__ = [
		"evaluation_mode", "evaluation_image_path", "evaluation_resize_input_image",
		"coco", "img_dir", "ids", "coco_id_to_contiguous_id", "contiguous_id_to_coco_id"
	]
	_keys_ = __slots__[:3]
	_defaults_ = ["single_image", None, False],  # single_image image_folder validation_set
	_types_ = [str, str, bool]
	_disp_ = __slots__

	def __init__(self, is_training=True, is_evaluation=False):
		super().__init__(is_training=is_training, is_evaluation=is_evaluation)

		split = "train" if is_training else "val"
		year = 2017
		ann_file = os.path.join(
			self.root, "annotations/instances_{}{}.json".format(split, year)
		)

		# disable printing, so that pycocotools print statements are not printed
		Logger.disable_printing()

		self.coco = COCO(ann_file)
		self.img_dir = os.path.join(self.root, "images/{}{}".format(split, year))
		self.ids = (
			list(self.coco.imgToAnns.keys())
			if is_training
			else list(self.coco.imgs.keys())
		)

		coco_categories = sorted(self.coco.getCatIds())
		# class No start from 1
		self.coco_id_to_contiguous_id = {
			coco_id: i + 1 for i, coco_id in enumerate(coco_categories)
		}
		self.contiguous_id_to_coco_id = {
			v: k for k, v in self.coco_id_to_contiguous_id.items()
		}
		# leave 0 as background class
		self.n_classes = len(self.contiguous_id_to_coco_id.keys()) + 1

		# enable printing
		Logger.enable_printing()

		Opt.set_by_cfg_path(Entry.DetectionModel, "n_classes", self.n_classes)
		self.n_records = len(self.ids)

	def _training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
		"""Training transforms should be implemented in sub-class"""
		raise NotImplementedError

	def _validation_transforms(self, size: tuple):
		"""Validation transforms should be implemented in sub-class"""
		raise NotImplementedError

	def _evaluation_transforms(self, size: tuple):
		"""Evaluation or Inference transforms (Resize (Optional) --> Tensor).

		.. note::
			Resizing the input to the same resolution as the detector's input is not enabled by default.
			It can be enabled by passing **--evaluation.detection.resize-input-images** flag.

		"""
		aug_list = []
		if self.evaluation_resize_input_image:
			aug_list.append(T.Resize(size=size))

		aug_list.append(T.ToTensor())
		return T.Compose(img_transforms=aug_list)

	def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
		crop_size_h, crop_size_w, img_index = batch_indexes_tup

		if self.is_training:
			transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
		elif self.is_evaluation:
			transform_fn = self._evaluation_transforms(size=(crop_size_h, crop_size_w))
		else:  # same for validation and evaluation
			transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

		image_id = self.ids[img_index]

		# read image
		image, img_name = self.get_image(image_id=image_id)
		im_width, im_height = image.size

		# get boxes([num_boxes, 4]) and labels([num_boxes])
		boxes, labels = self.get_boxes_and_labels(
			image_id=image_id, image_width=im_width, image_height=im_height
		)

		data = {"image": image, "box_labels": labels, "box_coordinates": boxes}

		if transform_fn is not None:
			data = transform_fn(data)

		output_data = {
			"image": {"image": data["image"]},
			"label": {
				"box_labels": data["box_labels"],  # [num_boxes, 4]
				"box_coordinates": data["box_coordinates"],  # [num_boxes]
				"image_id": torch.tensor(image_id),
				"image_width": torch.tensor(im_width),
				"image_height": torch.tensor(im_height),
			},
		}

		return output_data

	# for given image by id, get boxes and corresponding labels
	# boxes: [num_boxes, 4], labels: [num_boxes]
	def get_boxes_and_labels(self, image_id, image_width, image_height) -> Tuple[np.ndarray, np.ndarray]:
		ann_ids = self.coco.getAnnIds(imgIds=image_id)
		ann = self.coco.loadAnns(ann_ids)

		# filter crowd annotations
		ann = [obj for obj in ann if obj["iscrowd"] == 0]
		# get boxes and labels in the image
		boxes = np.array(
			[self._xywh2xyxy(obj["bbox"], image_width, image_height) for obj in ann],
			np.float32,
		).reshape((-1, 4))  # [num_boxes, 4]
		labels = np.array(
			[self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64
		).reshape((-1,))   # [num_boxes]

		# remove invalid boxes
		keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
		boxes = boxes[keep]
		labels = labels[keep]
		return boxes, labels

	# get the image tensor and image file name
	def get_image(self, image_id: int) -> Tuple:
		file_name = self.coco.loadImgs(image_id)[0]["file_name"]
		image_file = os.path.join(self.img_dir, file_name)
		image = self.read_image_pil(image_file)
		return image, file_name

	# convert form and clip by specified image size
	def _xywh2xyxy(self, box, image_width, image_height) -> List:
		x1, y1, w, h = box
		return [
			max(0, x1),
			max(0, y1),
			min(x1 + w, image_width),
			min(y1 + h, image_height),
		]

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def class_names() -> List:
		return [
			"background",
			"person",
			"bicycle",
			"car",
			"motorcycle",
			"airplane",
			"bus",
			"train",
			"truck",
			"boat",
			"traffic light",
			"fire hydrant",
			"stop sign",
			"parking meter",
			"bench",
			"bird",
			"cat",
			"dog",
			"horse",
			"sheep",
			"cow",
			"elephant",
			"bear",
			"zebra",
			"giraffe",
			"backpack",
			"umbrella",
			"handbag",
			"tie",
			"suitcase",
			"frisbee",
			"skis",
			"snowboard",
			"sports ball",
			"kite",
			"baseball bat",
			"baseball glove",
			"skateboard",
			"surfboard",
			"tennis racket",
			"bottle",
			"wine glass",
			"cup",
			"fork",
			"knife",
			"spoon",
			"bowl",
			"banana",
			"apple",
			"sandwich",
			"orange",
			"broccoli",
			"carrot",
			"hot dog",
			"pizza",
			"donut",
			"cake",
			"chair",
			"couch",
			"potted plant",
			"bed",
			"dining table",
			"toilet",
			"tv",
			"laptop",
			"mouse",
			"remote",
			"keyboard",
			"cell phone",
			"microwave",
			"oven",
			"toaster",
			"sink",
			"refrigerator",
			"book",
			"clock",
			"vase",
			"scissors",
			"teddy bear",
			"hair drier",
			"toothbrush",
		]
