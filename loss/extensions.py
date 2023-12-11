from . import BaseLossExtension
from utils.type_utils import Opt
from utils.entry_utils import Entry

from utils.tensor_utils import Tsr
from utils.logger import Logger
from utils.checkpoint_utils import CheckPoint
from utils import Util

import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from typing import Dict, Union, Tuple

from detection.utils import DttUtil

from torch.nn import functional as F
from utils.type_utils import Par


@Entry.register_entry(Entry.Loss)
class CrossEntropy(BaseLossExtension):
	__slots__ = ["use_class_weights", "label_smoothing", "ignore_index"]
	_disp_ = __slots__

	_keys_ = __slots__
	_defaults_ = [False, 0.0, -1]
	_types_ = [bool, float, int]

	_args_ = __slots__

	def __init__(
			self,
			use_class_weights: bool = None,
			label_smoothing: float = None,
			ignore_index: int = None
	):
		super().__init__(**Par.purify(locals()))

	def forward(self, prediction, target, input_sample=None):
		weight = None
		if self.use_class_weights and self.training:
			n_classes = prediction.shape[1]
			weight = self._class_weights(target=target, n_classes=n_classes)

		return F.cross_entropy(
			input=prediction,
			target=target,
			weight=weight,
			ignore_index=self.ignore_index,
			label_smoothing=self.label_smoothing if self.training else 0.0
		)


@Entry.register_entry(Entry.Loss)
class BinaryCrossEntropy(BaseLossExtension):
	__slots__ = ["nothing"]

	_disp_ = __slots__

	_keys_ = __slots__
	_defaults_ = ["nothing"]
	_types_ = [str]

	def forward(self, prediction, target, input_sample=None):
		if target.dim() != prediction.dim():
			target = F.one_hot(target, num_classes=prediction[-1])

		return F.binary_cross_entropy_with_logits(
			input=prediction,
			target=target,
		)


@Entry.register_entry(Entry.Loss)
class SSDLoss(BaseLossExtension):
	__slots__ = [
		"neg_pos_ratio", "max_monitor_iter", "update_wt_freq",
		"label_smoothing", "unscaled_reg_loss", "unscaled_conf_loss", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [3, -1, 200, 0.0, 1e-7, 1e-7]
	_types_ = [int, int, int, float, float, float]

	def __init__(self):
		super().__init__()

		self.wt_loc = 1.0
		self.curr_iter = 0
		self.reset_unscaled_loss_values()

	def reset_unscaled_loss_values(self):
		# initialize with very small float values
		self.unscaled_conf_loss = 1e-7
		self.unscaled_reg_loss = 1e-7

	def forward(self, prediction: Dict, target: Dict):
		# prediction
		# confidence: (batch_size, num_priors, num_classes)
		# predicted_locations :(batch_size, num_priors, 4)
		confidence = prediction["scores"]
		predicted_locations = prediction["locations"]

		# target
		# prior_labels: (batch_size, num_priors)
		# prior_locations: (batch_size, num_priors, 4)
		prior_labels = target["prior_labels"]
		prior_locations = target["prior_locations"]

		num_classes = confidence.shape[-1]
		num_coordinates = predicted_locations.shape[-1]  # 4

		# only location with label > 0 used for location loss
		pos_mask = prior_labels > 0
		# [positive_num_priors, 4] filter predicted positive locations
		predicted_locations = predicted_locations[pos_mask].reshape(-1, num_coordinates)
		# [positive_num_priors, 4] filter ground truth positive locations
		prior_locations = prior_locations[pos_mask].reshape(-1, num_coordinates)
		# number of positive priors
		num_pos = max(1, prior_locations.shape[0])
		# location loss
		smooth_l1_loss = F.smooth_l1_loss(
			predicted_locations, prior_locations, reduction="sum"
		)

		# only positive priors and top k negative priors(class label = 0) to be used for classification loss
		# mask : (N, num_priors)
		with torch.no_grad():
			loss = -F.log_softmax(confidence, dim=2)[:, :, 0]  # back_ground class log_softmax
			mask = DttUtil.hard_negative_mining(loss, prior_labels, self.neg_pos_ratio)

		# filter confidence
		confidence = confidence[mask, :]
		label_smoothing = self.label_smoothing if self.training else 0.0
		# classification_loss
		classification_loss = F.cross_entropy(
			input=confidence.reshape(-1, num_classes),
			target=prior_labels[mask],
			reduction="sum",
			label_smoothing=label_smoothing,
		)

		# classification loss may dominate localization loss or vice-versa
		# therefore, to ensure that their contributions are equal towards total loss, we scale regression loss.
		# if classification loss contribution is less (or more), then scaling factor will be < 1 ( > 1)
		if self.curr_iter <= self.max_monitor_iter and self.training:
			self.unscaled_conf_loss += Tsr.tensor_to_numpy_or_float(
				classification_loss
			)
			self.unscaled_reg_loss += Tsr.tensor_to_numpy_or_float(
				smooth_l1_loss
			)

			# update location loss weight
			if (self.curr_iter + 1) % self.update_wt_freq == 0 or self.curr_iter == self.max_monitor_iter:
				before_update = round(Tsr.tensor_to_numpy_or_float(self.wt_loc), 4)
				self.wt_loc = self.unscaled_conf_loss / self.unscaled_reg_loss
				self.reset_unscaled_loss_values()

				after_update = round(Tsr.tensor_to_numpy_or_float(self.wt_loc), 4)
				Logger.log(
					f"Updating localization loss multiplier from {before_update} to {after_update}"
				)

			self.curr_iter += 1

		# adjust location loss by weight
		if self.training and self.wt_loc > 0.0:
			smooth_l1_loss = smooth_l1_loss * self.wt_loc

		return (smooth_l1_loss + classification_loss) / num_pos


@Entry.register_entry(Entry.Loss)
class SegCrossEntropy(BaseLossExtension):
	__slots__ = ["ignore_idx", "class_weights", "aux_weight", "label_smoothing"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [-1, False, 0.4, 0.0]
	_types_ = [int, bool, float, float]

	def _compute_loss(self, pre_mask, target_mask, weight=None):
		b, c, x_h, x_w = pre_mask.shape
		b, y_h, y_w = target_mask.shape

		# using label smoothing only for training
		label_smoothing = self.label_smoothing if self.training else 0.0

		if x_h != y_h or x_w != y_w:
			pre_mask = F.interpolate(
				pre_mask, size=(y_h, y_w), mode="bilinear", align_corners=True
			)

		return F.cross_entropy(
			input=pre_mask,
			target=target_mask,
			weight=weight,
			ignore_index=self.ignore_idx,
			label_smoothing=label_smoothing
		)

	def forward(
			self,
			prediction: Union[Tensor or Tuple[Tensor, Tensor]],
			target: Tensor
	) -> Tensor:
		aux_out = None
		if isinstance(prediction, Tuple) and len(prediction) == 2:
			mask, aux_out = prediction
		else:
			mask = prediction

		cls_wts = None
		if self.training:
			if self.class_weights:
				n_classes = mask.size(1) # Mask is of shape B x C x H x W
				cls_wts = self._class_weights(target=target, n_classes=n_classes)
			total_loss = self._compute_loss(pre_mask=mask, target_mask=target, weight=cls_wts)

			if aux_out is not None:
				loss_aux = self._compute_loss(pre_mask=aux_out, target_mask=target, weight=cls_wts)
				total_loss += loss_aux * self.aux_weight
			return total_loss
		else:
			return self._compute_loss(pre_mask=mask, target_mask=target, weight=None)


@Entry.register_entry(Entry.Loss)
class VanillaDistillationLoss(BaseLossExtension):
	__slots__ = [
		"teacher_model_name", "label_loss_name", "alpha", "tau",
		"adaptive_weight_balance", "steps", "weight_update_freq",
		"teacher_model_weights_path", "distillation_type",
	]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
		"resnet_50", "cross_entropy", 0.5, 1.0,
		False, 10000, 100,
		None, "soft"
	]
	_types_ = [
		str, str, float, float,
		bool, int, int,
		str, str
	]

	def __init__(self):
		super().__init__()
		self.is_distributed = Util.is_distributed()
		self.n_gpus = Opt.get(f"{Entry.Common}.Common.num_gpus")

		self.teacher_model = self.build_teacher_model()
		if not self.is_distributed and self.n_gpus > 0:
			self.teacher_model = torch.nn.DataParallel(self.teacher_model)
		self.teacher_model.eval()

		self.label_loss = self.build_label_loss_fn()

		self.weight_label = 1.0 if self.adaptive_weight_balance else self.alpha
		self.weight_dist = 1.0 if self.adaptive_weight_balance else 1.0 - self.alpha

		self.loss_acc_label = 0.0
		self.loss_acc_dist = 0.0
		self.step_counter = 0

		self.distillation_loss_fn = self.compute_soft_distillation_loss
		if self.distillation_type == "hard":
			self.distillation_loss_fn = self.compute_hard_distillation_loss

	def build_teacher_model(self) -> nn.Module:
		teacher_model = Entry.get_entity(Entry.ClassificationModel, entry_name=self.teacher_model_name)

		if self.teacher_model_weights_path is not None and os.path.isfile(self.teacher_model_weights_path):
			pretrained_wts_dict = torch.load(self.teacher_model_weights_path, map_location="cpu")
			CheckPoint.load_state_dict(model=teacher_model, state_dict=pretrained_wts_dict)
		else:
			raise RuntimeError(f"Pretrained weights are required for teacher model ({self.teacher_model_name}) in the distillation loss")

		return teacher_model

	def build_label_loss_fn(self) -> BaseLossExtension:
		return Entry.get_entity(Entry.Loss, entry_name=self.lablel_loss_name)

	def compute_soft_distillation_loss(self, input_sample, prediction) -> Tensor:
		"""
		Details about soft-distillation here: https://arxiv.org/abs/2012.12877
		"""

		with torch.no_grad():
			teacher_outputs = self.teacher_model(input_sample)

		multiplier = (self.tau * self.tau * 1.0) / prediction.numel()
		pred_dist = F.log_softmax(prediction / self.tau, dim=1)
		teach_dist = F.log_softmax(teacher_outputs / self.tau, dim=1)
		distillation_loss = (
			F.kl_div(pred_dist, teach_dist, reduction="sum", log_target=True)
			* multiplier
		)
		return distillation_loss

	def compute_hard_distillation_loss(self, input_sample, prediction) -> Tensor:
		"""
		Details about Distillation here: https://arxiv.org/abs/1503.02531
		"""
		with torch.no_grad():
			teacher_logits = self.teacher_model(input_sample)
			teacher_labels = teacher_logits.argmax(dim=-1)

		distillation_loss = self.label_loss(
			input_sample=input_sample, prediction=prediction, target=teacher_labels
		)
		return distillation_loss

	def compute_weights(self):
		prev_wt_dist = self.weight_dist
		self.weight_dist = round(
			self.loss_acc_label / (self.loss_acc_dist + self.eps), 3
		)
		# self.loss_acc_label = 0.0
		# self.loss_acc_dist = 0.0
		if self.is_master_node:
			Logger.log(
				"{} Contribution of distillation loss w.r.t label loss is updated".format(
					self.__class__.__name__
				)
			)
			print(
				"\t\t Dist. loss contribution: {} -> {}".format(
					prev_wt_dist, self.weight_dist
				)
			)

	def forward(self, prediction, target, input_sample=None) -> Tensor:
		distillation_loss = self.distillation_loss_fn(
			input_sample=input_sample, prediction=prediction
		)
		label_loss = self.label_loss(
			input_sample=input_sample, prediction=prediction, target=target
		)
		if self.adaptive_weight_balance and self.step_counter < self.steps:
			self.loss_acc_dist += Tsr.tensor_to_numpy_or_float(
				distillation_loss
			)
			self.loss_acc_label += Tsr.tensor_to_numpy_or_float(
				label_loss
			)

			# update the weights
			if (self.step_counter + 1) % self.weight_update_freq == 0:
				self.compute_weights()

			self.step_counter += 1
		elif self.adaptive_weight_balance and self.step_counter == self.steps:
			self.compute_weights()
			self.step_counter += 1

		total_loss = (self.weight_label * label_loss) + (
			self.weight_dist * distillation_loss
		)

		return total_loss
