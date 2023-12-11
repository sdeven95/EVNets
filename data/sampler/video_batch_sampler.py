import random
from typing import Optional

from . batch_sampler import BatchSampler, BatchSamplerDDP
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Sampler)
class VideoBatchSampler(BatchSampler):
	"""
	Batch sampler for videos

	Args:
		n_data_samples (int): Number of samples in the dataset
		is_training (Optional[bool]): Training or validation mode. Default: False
	"""
	__slots__ = ["num_frames_per_clip", "clips_per_video", "n_data_samples", "is_training"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [8, 1, None, False]
	_types_ = [int, int, int, bool]

	def __init__(
		self,
		n_data_samples: int = None,
		is_training: Optional[bool] = None,
	) -> None:
		super().__init__(
			n_data_samples=n_data_samples, is_training=is_training
		)

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			random.shuffle(self.img_indices)

		start_index = 0
		batch_size = self.batch_size_gpu0
		while start_index < self.n_samples:

			end_index = min(start_index + batch_size, self.n_samples)
			batch_ids = self.img_indices[start_index:end_index]
			start_index += batch_size

			if len(batch_ids) > 0:
				batch = [
					(
						self.crop_size_h,
						self.crop_size_w,
						b_id,
						self.num_frames_per_clip,
						self.clips_per_video,
					)
					for b_id in batch_ids
				]
				yield batch


@Entry.register_entry(Entry.Sampler)
class VideoBatchSamplerDDP(BatchSamplerDDP):
	"""
	Batch sampler for videos (DDP)

	Args:
		n_data_samples (int): Number of samples in the dataset
		is_training (Optional[bool]): Training or validation mode. Default: False
	"""
	__slots__ = ["num_frames_per_clip", "clips_per_video", "n_data_samples", "is_training"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [8, 1, None, False]
	_types_ = [int, int, int, bool]

	def __init__(
		self,
		n_data_samples: int = None,
		is_training: Optional[bool] = None,
	) -> None:
		super().__init__(
			n_data_samples=n_data_samples, is_training=is_training
		)

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
		batch_size = self.batch_size_gpu0
		while start_index < self.n_samples_per_replica:
			end_index = min(start_index + batch_size, self.n_samples_per_replica)
			batch_ids = indices_rank_i[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if n_batch_samples != batch_size:
				batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:
				batch = [
					(
						self.crop_size_h,
						self.crop_size_w,
						b_id,
						self.default_frames,
						self.clips_per_video,
					)
					for b_id in batch_ids
				]
				yield batch
