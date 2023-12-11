import random
from typing import Optional
from gbl import Constants

from . import BaseSamplerDP, BaseSamplerDDP
from .utils import SplUtil

from utils.entry_utils import Entry


# build few random crop_sizes, when generate batch, randomly select a crop_size
@Entry.register_entry(Entry.Sampler)
class MultiScaleSampler(BaseSamplerDP):
	"""
	Multi-scale Batch Sampler for data parallel
	
	Args:
		n_data_samples (int): Number of samples in the dataset
		is_training (Optional[bool]): Training or validation mode. Default: False
	"""
	__slots__ = [
		"crop_size_width", "crop_size_height", 
		"min_crop_size_width", "max_crop_size_width", 
		"min_crop_size_height", "max_crop_size_height",
		"max_img_scales", "check_scale_div_factor", "scale_ep_intervals", "scale_inc_factor", "scale_inc"
	]
	_keys_ = __slots__
	_defaults_ = [
		Constants.DEFAULT_IMAGE_WIDTH, Constants.DEFAULT_IMAGE_HEIGHT,
		160, 320,
		160, 320,
		5, 32, [40], 0.25, False
	]
	_types_ = [
		int, int,
		int, int,
		int, int,
		int, int, (int,), float, bool
	]
	_disp_ = __slots__

	def __init__(
		self,
		n_data_samples: int = None,
		is_training: Optional[bool] = None,
	) -> None:
		super().__init__(
			n_data_samples=n_data_samples, is_training=is_training
		)
	
		if isinstance(self.scale_ep_intervals, int):
			self.scale_ep_intervals = [self.scale_ep_intervals]
	
		if is_training:
			self.img_batch_tuples = SplUtil.image_batch_pairs(**self.__dict__)
			# over-ride the batch-size !!!
			self.img_batch_tuples = [
				(h, w, self.batch_size_gpu0) for h, w, b in self.img_batch_tuples
			]
		else:
			self.img_batch_tuples = [(self.crop_size_height, self.crop_size_width, self.batch_size_gpu0)]

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			random.shuffle(self.img_indices)
			random.shuffle(self.img_batch_tuples)
	
		start_index = 0
		n_samples = len(self.img_indices)
		while start_index < n_samples:
			crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)
	
			end_index = min(start_index + batch_size, n_samples)
			batch_ids = self.img_indices[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if len(batch_ids) != batch_size:
				batch_ids += self.img_indices[: (batch_size - n_batch_samples)]
			start_index += batch_size
	
			if len(batch_ids) > 0:
				batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
				yield batch


@Entry.register_entry(Entry.Sampler)
class MultiScaleSamplerDDP(BaseSamplerDDP):
	"""
	Multi-scale Batch Sampler for distributed data parallel

	Args:
		n_data_samples (int): Number of samples in the dataset
		is_training (Optional[bool]): Training or validation mode. Default: False
	"""
	__slots__ = [
		"crop_size_width", "crop_size_height",
		"min_crop_size_width", "max_crop_size_width",
		"min_crop_size_height", "max_crop_size_height",
		"max_img_scales", "check_scale_div_factor", "scale_ep_intervals", "scale_inc_factor", "scale_inc"
	]
	_keys_ = __slots__
	_defaults_ = [
		Constants.DEFAULT_IMAGE_WIDTH, Constants.DEFAULT_IMAGE_HEIGHT,
		160, 320,
		160, 320,
		5, 32, [40], 0.25, False
	]
	_types_ = [
		int, int,
		int, int,
		int, int,
		int, int, (int,), float, bool
	]
	_disp_ = __slots__

	def __init__(
		self,
		n_data_samples: int = None,
		is_training: Optional[bool] = None,
	) -> None:
		super().__init__(
			n_data_samples=n_data_samples, is_training=is_training
		)

		if is_training:
			self.img_batch_tuples = SplUtil.image_batch_pairs(**self.__dict__)
			self.img_batch_tuples = [
				(h, w, self.batch_size_gpu0) for h, w, b in self.img_batch_tuples
			]
		else:
			self.img_batch_tuples = [
				(self.crop_size_height, self.crop_size_width, self.batch_size_gpu0)
			]

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			indices_rank_i = self.img_indices[
				self.rank: len(self.img_indices): self.num_replicas
			]
			random.shuffle(indices_rank_i)
		else:
			indices_rank_i = self.img_indices[
				self.rank: len(self.img_indices): self.num_replicas
			]

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
