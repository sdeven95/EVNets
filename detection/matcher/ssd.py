from utils.entry_utils import Entry
from . import BaseMatcher

from ..utils import DttUtil

from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor


@Entry.register_entry(Entry.Matcher)
class SSDMatcher(BaseMatcher):
	"""
	This class assigns labels to anchor via `SSD matching process <https://arxiv.org/abs/1512.02325>`_

	Args:
		opts: command line arguments
		bg_class_id: Background class index

	Shape:
		- Input:
			- gt_boxes: Ground-truth boxes in corner form (xyxy format). Shape is :math:`(N, 4)` where :math:`N` is the number of boxes
			- gt_labels: Ground-truth box labels. Shape is :math:`(N)`
			- anchors: Anchor boxes in center form (c_x, c_y, w, h). Shape is :math:`(M, 4)` where :math:`M` is the number of anchors

		- Output:
			- matched_boxes of shape :math:`(M, 4)`
			- matched_box_labels of shape :math:`(M)`
	"""
	__slots__ = [
		"center_variance", "size_variance", "iou_threshold",
		"bg_class_id"
	]
	_keys_ = __slots__[:-1]
	_disp_ = __slots__
	_defaults_ = [0.1, 0.2, 0.5]
	_types_ = [float, float, float]

	def __init__(self, bg_class_id: int = 0, **kwargs):
		super().__init__(**kwargs)
		self.bg_class_id = bg_class_id

		assert 0.0 < self.center_variance < 1.0, f"The value of center variance should between 0 and 1. Got: {self.center_variance}"
		assert 0.0 < self.size_variance < 1.0, f"The value of size variance should between 0 and 1. Got: {self.size_variance}"
		assert 0.0 < self.iou_threshold < 1.0, f"The value of IOU threshold should between 0 and 1. Got: {self.iou_threshold}"

	def __call__(
		self,
		gt_boxes: Union[np.ndarray, Tensor],
		gt_labels: Union[np.ndarray, Tensor],
		anchors: Tensor,
	) -> Tuple[Tensor, Tensor]:
		if isinstance(gt_boxes, np.ndarray):
			gt_boxes = torch.from_numpy(gt_boxes)
		if isinstance(gt_labels, np.ndarray):
			gt_labels = torch.from_numpy(gt_labels)

		# convert box priors from center [c_x, c_y, w, h] to corner_form [left, top, right, bottom]
		anchors_xyxy = DttUtil.center_form_to_corner_form(boxes=anchors)

		# get group truth(matched) box and label for every prior
		# return [num_priors] group box list for per prior, label list for per prior
		matched_boxes_xyxy, matched_labels = DttUtil.assign_priors(
			gt_boxes,  # [num_boxes, 4] gt_boxes are in corner form [left, top, right, bottom]
			gt_labels,  # [num_boxes]
			anchors_xyxy,  # all priors( 6 * num_points_per_stride * num_strides) priors are in corner form [left, top, right, bottom]
			self.iou_threshold,
			background_id=self.bg_class_id,
		)

		# convert the matched boxes to center form [c_x, c_y, w, h]
		matched_boxes_cxcywh = DttUtil.corner_form_to_center_form(matched_boxes_xyxy)

		# convert to offset
		# Eq.(2) in paper https://arxiv.org/pdf/1512.02325.pdf
		# according to the group truth box and anchor, get four elements for computing location loss
		boxes_for_regression = DttUtil.convert_boxes_to_locations(
			gt_boxes=matched_boxes_cxcywh,  # center form
			prior_boxes=anchors,  # center form
			center_variance=self.center_variance,
			size_variance=self.size_variance,
		)

		return boxes_for_regression, matched_labels

	# for inference, decode boxes from predicted locations and corresponding anchors'
	# pred_locations: [N, 4] anchors: [N, 4]
	def convert_to_boxes(
		self, pred_locations: torch.Tensor, anchors: torch.Tensor
	) -> Tensor:
		"""
		Decodes boxes from predicted locations and anchors.
		"""

		# decode boxes in center form
		boxes = DttUtil.convert_locations_to_boxes(
			pred_locations=pred_locations,
			anchor_boxes=anchors,
			center_variance=self.center_variance,
			size_variance=self.size_variance,
		)
		# convert boxes from center form [c_x, c_y] to corner form [x, y]
		boxes = DttUtil.center_form_to_corner_form(boxes)
		return boxes
