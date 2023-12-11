import torch
from torch import Tensor

import math
from typing import Optional, Tuple


class DttUtil:

	@staticmethod
	# computer elements of location loss function according to group truth boxes(one or more) and specified default boxes(one or more)
	# help to computer target/true output(elements of location loss function) for training
	def convert_boxes_to_locations(
		gt_boxes: Tensor, prior_boxes: Tensor, center_variance: float, size_variance: float
		):
		"""
		This function implements Eq.(2) in the `SSD paper <https://arxiv.org/pdf/1512.02325.pdf>`_

		Args:
			gt_boxes (Tensor): Ground truth boxes in center form (cx, cy, w, h)
			prior_boxes (Tensor): Prior boxes in center form (cx, cy, w, h)
			center_variance (float): variance value for centers (c_x and c_y)
			size_variance (float): variance value for size (height and width)

		Returns:
			boxes tensor for training
		"""

		# T_cx = ((g_cx - d_cx) / d_w) / center_variance; Center variance is nothing but normalization
		# T_cy = ((g_cy - d_cy) / d_h) / center_variance
		# T_w = log(g_w/d_w) / size_variance and T_h = log(g_h/d_h) / size_variance

		# priors can have one dimension less
		if prior_boxes.dim() + 1 == gt_boxes.dim():
			prior_boxes = prior_boxes.unsqueeze(0)

		target_centers = (
			(gt_boxes[..., :2] - prior_boxes[..., :2]) / prior_boxes[..., 2:]
		) / center_variance
		target_size = torch.log(gt_boxes[..., 2:] / prior_boxes[..., 2:]) / size_variance
		return torch.cat((target_centers, target_size), dim=-1)

	@staticmethod
	# computer predict box according to predicted location elements
	# for predicting or interference
	def convert_locations_to_boxes(
		pred_locations: Tensor,
		anchor_boxes: Tensor,
		center_variance: float,
		size_variance: float,
		) -> Tensor:
		"""
			This is an inverse of convert_boxes_to_locations function (or Eq.(2) in `SSD paper <https://arxiv.org/pdf/1512.02325.pdf>`_
		Args:
			pred_locations (Tensor): predicted locations from detector
			anchor_boxes (Tensor): prior boxes in center form
			center_variance (float): variance value for centers (c_x and c_y)
			size_variance (float): variance value for size (height and width)

		Returns:
			predicted boxes' tensor in center form
		"""

		# priors can have one dimension less.
		if anchor_boxes.dim() + 1 == pred_locations.dim():
			anchor_boxes = anchor_boxes.unsqueeze(0)

		# T_w = log(g_w/d_w) / size_variance ==> g_w = exp(T_w * size_variance) * d_w
		# T_h = log(g_h/d_h) / size_variance ==> g_h = exp(T_h * size_variance) * d_h
		pred_size = (
			torch.exp(pred_locations[..., 2:] * size_variance) * anchor_boxes[..., 2:]
		)
		# T_cx = ((g_cx - d_cx) / d_w) / center_variance ==> g_cx = ((T_cx * center_variance) * d_w) + d_cx
		# T_cy = ((g_cy - d_cy) / d_w) / center_variance ==> g_cy = ((T_cy * center_variance) * d_h) + d_cy
		pred_center = (
			pred_locations[..., :2] * center_variance * anchor_boxes[..., 2:]
		) + anchor_boxes[..., :2]

		return torch.cat((pred_center, pred_size), dim=-1)

	@staticmethod
	# (cx, cy, w, h) -> (x_left, y_top, x_right, y_bott)
	def center_form_to_corner_form(boxes: Tensor) -> Tensor:
		return torch.cat(
			(
				boxes[..., :2] - boxes[..., 2:] * 0.5,
				boxes[..., :2] + boxes[..., 2:] * 0.5
			),
			dim=-1,
		)

	@staticmethod
	# (x_left, y_top, x_right, y_bott) -> (cx, cy, w, h)
	def corner_form_to_center_form(boxes: Tensor) -> Tensor:
		return torch.cat(
			(
				(boxes[..., :2] + boxes[..., 2:]) * 0.5,
				boxes[..., 2:] - boxes[..., :2]
			),
			dim=-1,
		)

	@staticmethod
	# box coordinate: [..., 4] 4-> left, top, right, bottom
	# intersection area / union section area
	def box_iou(
		boxes0: Tensor, boxes1: Tensor, eps: Optional[float] = 1e-5,
		) -> Tensor:
		"""
		Computes intersection-over-union between two boxes
		Args:
			boxes0 (Tensor): Boxes 0 of shape (N, 4)
			boxes1 (Tensor): Boxes 1 of shape (N or 1, 4)
			eps (Optional[float]): A small value is added to denominator for numerical stability

		Returns:
			iou (Tensor): IoU values between boxes0 and boxes1 and has shape (N)
		"""

		def area_of(left_top, right_bottom) -> torch.Tensor:
			"""
			Given two corners of the rectangle, compute the area
			Args:
				left_top (N, 2): left top corner.
				right_bottom (N, 2): right bottom corner.
			Returns:
				area (N): return the area.
			"""
			hw = torch.clamp(right_bottom - left_top, min=0.0)
			return hw[..., 0] * hw[..., 1]

		overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
		overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

		overlap_area = area_of(overlap_left_top, overlap_right_bottom)
		area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
		area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
		return overlap_area / (area0 + area1 - overlap_area + eps)

	@staticmethod
	# firstly make sure the specified group truth box(target) are assigned to the right prior box (max iou rule)
	# other each prior box choose a target group truth box which has the max iou with it
	# return: labels[num_priors] include label for per prior, background label is 0
	# 		  boxes[num_priors] include group truth box for per prior, the prior's (with background label) group truth box actually will not be used
	def assign_priors(
		gt_boxes: Tensor,  # group truth boxes for an image [num_boxes, 4]
		gt_labels: Tensor,  # labels of the group truth boxes [num_boxes]
		corner_form_priors: Tensor,
		iou_threshold: float,
		background_id: Optional[int] = 0,
		) -> Tuple[Tensor, Tensor]:
		"""
		Assign ground truth boxes and targets to priors (or anchors)

		Args:
			gt_boxes (Tensor): Ground-truth boxes tensor of shape (num_targets, 4)
			gt_labels (Tensor): Ground-truth labels of shape (num_targets)
			corner_form_priors (Tensor): Priors in corner form and has shape (num_priors, 4)
			iou_threshold (float): Overlap between priors and gt_boxes.
			background_id (int): Background class index. Default: 0

		Returns:
			boxes (Tensor): Boxes mapped to priors and has shape (num_priors, 4)
			labels (Tensor): Labels for mapped boxes and has shape (num_priors)
		"""

		# note: 0 as background_id, and the normal category index is bigger than 0, usually start from 1
		if gt_labels.nelement() == 0:
			# Images may not have any labels
			dev = corner_form_priors.device
			gt_boxes = torch.zeros((1, 4), dtype=torch.float32, device=dev)  # gt_boxes=[[0, 0, 0, 0]]
			gt_labels = torch.zeros(1, dtype=torch.int64, device=dev)  # gt_labels=[0]

		# gt_boxes: [1, num_targets, 4]  priors: [num_priors, 1, 4]
		# ious: [num_priors, num_targets]
		ious = DttUtil.box_iou(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))

		# first step, (1) for every prior, assign one target (max iou rule)
		# size: num_priors
		best_target_per_prior, best_target_per_prior_index = ious.max(1)

		# first step, (2) for every target, assign one prior (max iou rule)
		# size: num_targets
		best_prior_per_target, best_prior_per_target_index = ious.max(0)

		# second step, for every target, assign it to one prior
		# i.e. assign target to prior satisfied max( iou(target, every_prior))
		# so this operation has higher priority than (1) of first step
		# i.e. must care about target firstly, selecting the right prior for it
		# then if a prior is not selected by any target, result of (1) in first step is kept, but assigned iou < 1
		for target_index, prior_index in enumerate(best_prior_per_target_index):
			best_target_per_prior_index[prior_index] = target_index

		# this is original comment: 2.0 is used to make sure every target has a prior assigned
		# for selected priors, set its iou to 2, make sure it is assigned since normal iou between 0 and 1
		best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

		# note: it's possible more than one target assigned to one prior, but one prior only accept one target
		# the final result is: for selected prior by any target, their ious is equal to 2, for others, between 0 and 1(got from (1) of first step)

		# if none of group truth boxes exists, gt_boxes=[[0, 0, 0, 0]], gt_labels=[0], [0, 0, 0, 0] target is assigned to the first prior with iou = 2
		# and other prior automatically assign itself to target 0 with iou = 0 in (1) of first step
		# the final result is: every prior has target index 0

		# size: [num_priors]
		# every prior has a target index, but the selected prior by targets with iou=2 and unselected with 0 < iou< 1
		labels = gt_labels[best_target_per_prior_index]

		# set prior with iou < iou_threshold target index to background_id(usually equal to 0)
		# finally, we get target labels for every prior
		# every selected prior or iou > iou_threshold prior has right targeted index, other priors has background index
		labels[best_target_per_prior < iou_threshold] = background_id

		# get the group truth box for every prior
		boxes = gt_boxes[best_target_per_prior_index]
		return boxes, labels

	@staticmethod
	#  for [N, num_priors], keep positive and first 3 negative(by loss) positions
	#  maybe could be considered that: it has been done firstly: ([N, num_priors, num_classes], -1) => [N, num_priors]
	def hard_negative_mining(
		loss: Tensor, labels: Tensor, neg_pos_ratio: int
		) -> Tensor:
		"""
		This function is used to suppress the presence of a large number of negative predictions. For any example/image,
		it keeps all the positive predictions and cut the number of negative predictions to make sure the ratio
		between the negative examples and positive examples is no more than the given ratio for an image.
		Args:
			loss (Tensor): the loss for each example and has shape (N, num_priors). N=example/image size, num_priors=number of default bound boxes
			labels (Tensor): the labels and has shape (N, num_priors).
			neg_pos_ratio (int):  the ratio between the negative examples and positive examples. Usually, it is set as 3.

		"""
		# positive mask
		pos_mask = labels > 0  # positive mask [N, num_priors]

		num_pos = pos_mask.long().sum(dim=1, keepdim=True)  # number of positive priors  [N, 1]
		num_neg = num_pos * neg_pos_ratio  # number of kept priors [N, 1]

		loss[pos_mask] = -math.inf   # firstly ignore positive positions
		_, indexes = loss.sort(dim=1, descending=True)  # sort aligning columns to get indexes [N, num_priors]
		_, orders = indexes.sort(dim=1)  # sort indexes [N, num_priors]
		neg_mask = orders < num_neg   # negative mask, keep only k-first by loss

		return pos_mask | neg_mask   # positive and k-first negative are both hold up
