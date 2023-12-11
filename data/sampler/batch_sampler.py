from . import BaseSamplerDP, BaseSamplerDDP
from gbl import Constants
import random
import numpy as np

from utils.entry_utils import Entry


@Entry.register_entry(Entry.Sampler)
class BatchSampler(BaseSamplerDP):
	__slots__ = ["crop_size_width", "crop_size_height", "num_repeats"]
	_keys_ = __slots__
	_defaults_ = [Constants.DEFAULT_IMAGE_WIDTH, Constants.DEFAULT_IMAGE_HEIGHT, 1]
	_types_ = [int, int, int]
	_disp_ = __slots__

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.num_repeats = self.num_repeats if self.is_training else 1

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			img_indices = list(np.repeat(self.img_indices, repeats=self.num_repeats))
			random.shuffle(img_indices)
		else:
			img_indices = self.img_indices

		start_index = 0
		batch_size = self.batch_size_gpu0
		n_samples = len(img_indices)
		while start_index < n_samples:
			end_index = min(start_index + batch_size, n_samples)
			batch_ids = img_indices[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if n_batch_samples != batch_size:
				batch_ids += img_indices[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:
				yield [(self.crop_size_width, self.crop_size_height, b_id) for b_id in batch_ids]

	def __len__(self):
		return len(self.img_indices) * self.num_repeats


@Entry.register_entry(Entry.Sampler)
class BatchSamplerDDP(BaseSamplerDDP):
	__slots__ = ["crop_size_width", "crop_size_height", "num_repeats"]
	_keys_ = __slots__
	_defaults_ = [Constants.DEFAULT_IMAGE_WIDTH, Constants.DEFAULT_IMAGE_HEIGHT, 1]
	_types_ = [int, int, int]
	_disp_ = __slots__

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.num_repeats = self.num_repeats if self.is_training else 1

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			img_indices = list(np.repeat(self.img_indices, repeats=self.num_repeats))
			# get indices for current replica
			indices_rank_i = img_indices[self.rank: len(img_indices): self.num_replicas]
			random.shuffle(indices_rank_i)
		else:
			indices_rank_i = self.img_indices[self.rank: len(self.img_indices): self.num_replicas]

		start_index = 0
		batch_size = self.batch_size_gpu0

		n_samples_rank_i = len(indices_rank_i)
		while start_index < n_samples_rank_i:
			end_index = min(start_index + batch_size, n_samples_rank_i)
			batch_ids = indices_rank_i[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if n_batch_samples != batch_size:
				batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:
				batch = [
					(self.crop_size_height, self.crop_size_width, b_id) for b_id in batch_ids
				]
				yield batch

	def __len__(self):
		return (len(self.img_indices) // self.num_replicas) * self.num_repeats
