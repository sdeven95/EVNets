from . import BaseSamplerDP, BaseSamplerDDP
from gbl import Constants
import random
import numpy as np
from .utils import SplUtil
from utils.logger import Logger
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Sampler)
class VariableBatchSampler(BaseSamplerDP):
	__slots__ = [
				"crop_size_width", "crop_size_height",
				"min_crop_size_width", "max_crop_size_width", "min_crop_size_height", "max_crop_size_height",
				"scale_inc", "scale_ep_intervals",
				"min_scale_inc_factor", "max_scale_inc_factor",
				"check_scale_div_factor", "max_img_scales",
				"num_repeats"
	]
	_keys_ = __slots__
	_defaults_ = [
		Constants.DEFAULT_IMAGE_WIDTH, Constants.DEFAULT_IMAGE_HEIGHT,
		160, 320, 160, 320,
		False, [40, ],
		1, 1,
		32, 5,
		1,
	]
	_types_ = [
		int, int,
		int, int, int, int,
		bool, (int, ),
		int, int,
		int, int,
		int,
	]
	_disp_ = __slots__

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.num_repeats = self.num_repeats if self.is_training else 1

		if self.is_training:
			self.img_batch_tuples = SplUtil.image_batch_pairs(**self.__dict__)
		else:
			self.img_batch_tuples = [(self.crop_size_height, self.crop_size_width, self.batch_size_gpu0)]

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			img_indices = list(np.repeat(self.img_indices, repeats=self.num_repeats))
			random.shuffle(img_indices)
		else:
			img_indices = self.img_indices

		start_index = 0
		n_samples = len(img_indices)
		while start_index < n_samples:
			crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

			end_index = min(start_index + batch_size, n_samples)
			batch_ids = img_indices[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if n_batch_samples != batch_size:
				batch_ids += img_indices[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:
				batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
				yield batch

	def update_scales(self):
		if self.epoch in self.scale_ep_intervals and self.scale_inc:
			self.min_crop_size_width += int(
				self.min_crop_size_width * self.min_scale_inc_factor
			)
			self.max_crop_size_width += int(
				self.max_crop_size_width * self.max_scale_inc_factor
			)

			self.min_crop_size_height += int(
				self.min_crop_size_height * self.min_scale_inc_factor
			)
			self.max_crop_size_height += int(
				self.max_crop_size_height * self.max_scale_inc_factor
			)

			self.img_batch_tuples = SplUtil.image_batch_pairs(**self._para_dict_)
			Logger.log("Scales updated in {}".format(self.__class__.__name__), only_master_node=False)
			Logger.log("New scales: {}".format(self.img_batch_tuples), only_master_node=False)

	def __len__(self):
		return len(self.img_indices) * self.num_repeats


@Entry.register_entry(Entry.Sampler)
class VariableBatchSamplerDDP(BaseSamplerDDP):
	__slots__ = [
				"crop_size_width", "crop_size_height",
				"min_crop_size_width", "max_crop_size_width", "min_crop_size_height", "max_crop_size_height",
				"scale_inc", "scale_ep_intervals",
				"min_scale_inc_factor", "max_scale_inc_factor",
				"check_scale_div_factor", "max_img_scales",
				"num_repeats"
	]
	_keys_ = __slots__
	_defaults_ = [
		DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
		160, 320, 160, 320,
		False, [40, ],
		1, 1,
		32, 5,
		1,
	],
	_types_ = [
		int, int,
		int, int, int, int,
		bool, (int, ),
		int, int,
		int, int,
		int,
	],
	_disp_ = __slots__

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.num_repeats = self.num_repeats if self.is_training else 1

		if self.is_training:
			self.img_batch_tuples = SplUtil.image_batch_pairs(**self.__dict__)
		else:
			self.img_batch_tuples = [(self.crop_size_height, self.crop_size_width, self.batch_size_gpu0)]

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			img_indices = list(np.repeat(self.img_indices, repeats=self.num_repeats))
			indices_rank_i = img_indices[self.rank: len(img_indices): self.num_replicas]
			random.shuffle(indices_rank_i)
		else:
			indices_rank_i = self.img_indices[self.rank: len(self.img_indices): self.num_replicas]

		start_index = 0
		n_samples_rank_i = len(indices_rank_i)
		while start_index < n_samples_rank_i:
			crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

			end_index = min(start_index + batch_size, n_samples_rank_i)
			batch_ids = indices_rank_i[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if n_batch_samples != batch_size:
				batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:
				batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
				yield batch

	def update_scales(self):
		if self.epoch in self.scale_ep_intervals and self.scale_inc:
			self.min_crop_size_width += int(
				self.min_crop_size_width * self.min_scale_inc_factor
			)
			self.max_crop_size_width += int(
				self.max_crop_size_width * self.max_scale_inc_factor
			)

			self.min_crop_size_height += int(
				self.min_crop_size_height * self.min_scale_inc_factor
			)
			self.max_crop_size_height += int(
				self.max_crop_size_height * self.max_scale_inc_factor
			)

			self.img_batch_tuples = SplUtil.image_batch_pairs(**self._para_dict_)
			Logger.log("Scales updated in {}".format(self.__class__.__name__), only_master_node=False)
			Logger.log("New scales: {}".format(self.img_batch_tuples), only_master_node=False)

	def __len__(self):
		return len(self.img_indices) * self.num_repeats
