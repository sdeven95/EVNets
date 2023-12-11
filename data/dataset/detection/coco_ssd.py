from .coco_base import COCODetection

import math
import torch
from ...transform import extensions as TE

from utils.type_utils import Opt
from utils.entry_utils import Entry

from typing import Optional, Tuple, Dict


@Entry.register_entry(Entry.Dataset)
class COCODetectionSSD(COCODetection):
	def __init__(self, is_training=True, is_evaluation=False):
		super().__init__(is_training=is_training, is_evaluation=is_evaluation)

		self.anchor = Entry.get_entity(Entry.Anchor)
		self.matcher = Entry.get_entity(Entry.Matcher)

		for aug in ("train", "val", "eval"):
			Opt.set(f"{Entry.Collate}.{aug}", "coco_ssd_collate_fn")

	def _training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
		"""Training data augmentation methods
		(SSDCroping --> PhotometricDistort --> RandomHorizontalFlip -> Resize --> ToTensor).
		"""
		aug_list = [
			TE.SSDCropping(),
			TE.PhotometricDistort(),
			TE.RandomHorizontalFlip(),
			TE.Resize(size=size),
			TE.BoxPercentCoords(),
			TE.ToTensor(),
		]

		return TE.Compose(img_transforms=aug_list)

	def _validation_transforms(self, size: tuple, *args, **kwargs):
		"""Implements validation transformation method (Resize --> ToTensor)."""
		aug_list = [
			TE.Resize(),
			TE.BoxPercentCoords(),
			TE.ToTensor(),
		]
		return TE.Compose(img_transforms=aug_list)

	def generate_anchors(self, height, width):
		"""Generate anchors **on-the-fly** based on the input resolution."""
		anchors = []
		for output_stride in self.anchor.output_strides:
			if output_stride == -1:
				fm_width = fm_height = 1
			else:
				fm_width = int(math.ceil(width / output_stride))
				fm_height = int(math.ceil(height / output_stride))
			fm_anchor = self.anchor(
				fm_height=fm_height, fm_width=fm_width, fm_output_stride=output_stride
			)
			anchors.append(fm_anchor)
		anchors = torch.cat(anchors, dim=0)
		return anchors

	def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
		crop_size_h, crop_size_w, img_index = batch_indexes_tup

		if self.is_training:
			transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
		else:
			# During evaluation, we use base class
			transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

		image_id = self.ids[img_index]

		image, image_file_name = self.get_image(image_id=image_id)
		im_width, im_height = image.size
		# boxes: [num_boxes, 4], labels: [num_boxes]
		boxes, labels = self.get_boxes_and_labels(
			image_id=image_id, image_width=im_width, image_height=im_height
		)

		data = {"image": image, "box_labels": labels, "box_coordinates": boxes}

		data = transform_fn(data)

		# convert to priors
		anchors = self.generate_anchors(height=crop_size_h, width=crop_size_w)

		# [num_priors] offset list for per prior, label list for per prior
		prior_locations, prior_labels = self.matcher(
			gt_boxes=data["box_coordinates"],
			gt_labels=data["box_labels"],
			anchors=anchors,
		)

		output_data = {
			"image": {
				"image": data["image"]
			},
			"label": {
				"prior_labels": prior_labels,
				"prior_locations": prior_locations,
				"image_id": torch.tensor(image_id),
				"image_width": torch.tensor(im_width),
				"image_height": torch.tensor(im_height),
			},
		}

		return output_data

	def __repr__(self):
		from ...sampler.utils import SplUtil

		im_h, im_w = SplUtil.image_size_from_opts()

		if self.is_training:
			transforms_str = self._training_transforms(size=(im_h, im_w))
		elif self.is_evaluation:
			transforms_str = self._evaluation_transforms(size=(im_h, im_w))
		else:
			transforms_str = self._validation_transforms(size=(im_h, im_w))

		return \
			super()._repr_by_line() \
			+ "\nTransform details:\n" + repr(transforms_str) \
			+ "\nAnchor details:\n" + repr(self.anchor) \
			+ "\nMatcher details:\n" + repr(self.matcher)


@Entry.register_entry(Entry.Collate)
def coco_ssd_collate_fn(batch):

	new_batch = {"image": dict(), "label": dict()}

	# new_batch = {
	# 	"image": {
	# 		image: [image1, image2, ...]
	# 	},
	# 	"label": {
	# 		box_labels: [image1_box_labels, image2_box_labels, ...],
	# 		box_coordinates: [image1_box_coordinates, image2_box_coordinates, ...],
	# 		image_id: [image1_image_id, image2_image_id, ...],
	# 		image_width: [image1_image_with, image2_image_width, ...],
	# 		image_height: [image1_image_height, image2_image_height, ...]
	# 	}
	# }
	for b_id, record in enumerate(batch):
		# prepare inputs
		if "image" in record["image"]:
			if "image" in new_batch["image"]:
				new_batch["image"]["image"].append(record["image"]["image"])
			else:
				new_batch["image"]["image"] = [record["image"]["image"]]

		# prepare outputs
		if "box_labels" in record["label"]:
			if "box_labels" in new_batch["label"]:
				new_batch["label"]["prior_labels"].append(record["label"]["prior_labels"])
			else:
				new_batch["label"]["prior_labels"] = [record["label"]["prior_labels"]]

		if "prior_locations" in record["label"]:
			if "prior_locations" in new_batch["label"]:
				new_batch["label"]["prior_locations"].append(record["label"]["prior_locations"])
			else:
				new_batch["label"]["prior_locations"] = [record["label"]["prior_locations"]]

		if "image_id" in record["label"]:
			if "image_id" in new_batch["label"]:
				new_batch["label"]["image_id"].append(record["label"]["image_id"])
			else:
				new_batch["label"]["image_id"] = [record["label"]["image_id"]]

		if "image_width" in record["label"]:
			if "image_width" in new_batch["label"]:
				new_batch["label"]["image_width"].append(record["label"]["image_width"])
			else:
				new_batch["label"]["image_width"] = [record["label"]["image_width"]]

		if "image_height" in record["label"]:
			if "image_height" in new_batch["label"]:
				new_batch["label"]["image_height"].append(
					record["label"]["image_height"]
				)
			else:
				new_batch["label"]["image_height"] = [record["label"]["image_height"]]

	# stack inputs
	# new_batch["image"]["image"] -> [N, C, H, W]
	new_batch["image"]["image"] = torch.stack(new_batch["image"]["image"], dim=0)

	# stack outputs [N, ...]
	if "prior_labels" in new_batch["label"]:
		new_batch["label"]["prior_labels"] = torch.stack(
			new_batch["label"]["prior_labels"], dim=0
		)

	if "prior_locations" in new_batch["label"]:
		new_batch["label"]["prior_locations"] = torch.stack(
			new_batch["label"]["prior_locations"], dim=0
		)

	if "image_id" in new_batch["label"]:
		new_batch["label"]["image_id"] = torch.stack(
			new_batch["label"]["image_id"], dim=0
		)

	if "image_width" in new_batch["label"]:
		new_batch["label"]["image_width"] = torch.stack(
			new_batch["label"]["image_width"], dim=0
		)

	if "image_height" in new_batch["label"]:
		new_batch["label"]["image_height"] = torch.stack(
			new_batch["label"]["image_height"], dim=0
		)

	return new_batch
