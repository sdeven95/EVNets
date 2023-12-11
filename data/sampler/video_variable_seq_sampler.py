import random
from typing import Optional

from .utils import SplUtil
from .variable_batch_sampler import VariableBatchSampler, VariableBatchSamplerDDP
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Sampler)
class VideoVariableSeqSampler(VariableBatchSampler):
	"""
	Extends `Variably-size multi-scale batch sampler <https://arxiv.org/abs/2110.02178?context=cs.LG>` for videos

	Args:
		n_data_samples (int): Number of samples in the dataset
		is_training (Optional[bool]): Training or validation mode. Default: False
	"""
	__slots__ = [
		"num_frames_per_clip", "random_video_clips",
		"min_clips_per_video", "max_clips_per_video", "clips_per_video", "min_frames_per_clip"
	]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
		8, False,
		1, 5, 1, 1
	],
	_types_ = [
		int, bool,
		int, int, int, int
	]

	def __init__(
		self,
		n_data_samples: int = None,
		is_training: Optional[bool] = None,
	) -> None:
		super().__init__(
			n_data_samples=n_data_samples, is_training=is_training
		)

		if not is_training:
			self.random_video_clips = False

		if is_training:
			# override img_batch_tuples
			self.img_batch_tuples = SplUtil.make_video_pairs(
				crop_size_height=self.crop_size_height,
				crop_size_width=self.crop_size_width,
				min_crop_size_height=self.min_crop_size_height,
				max_crop_size_height=self.max_crop_size_height,
				min_crop_size_width=self.min_crop_size_width,
				max_crop_size_width=self.max_crop_size_width,
				max_n_scales=self.max_n_scales,
				check_scale_div_factor=self.check_scale_div_factor,
				num_frames_per_clip=self.num_frames_per_clip,
			)
		else:
			self.img_batch_tuples = [
				(self.crop_size_h, self.crop_size_w, self.default_frames)
			]

	def __iter__(self):
		if self.shuffle:
			random.seed(self.epoch)
			random.shuffle(self.img_indices)
			random.shuffle(self.img_batch_tuples)

		start_index = 0
		while start_index < self.n_samples:
			if self.random_video_clips:
				# randomly sample number of clips and adjust frames per clip
				n_clips = max(
					1,
					random.randint(self.min_clips_per_video, self.max_clips_per_video),
				)
				batch_size = max(
					self.batch_size_gpu0,
					self.batch_size_gpu0 * (self.clips_per_video // n_clips),
				)
			else:
				n_clips = self.clips_per_video
				batch_size = self.batch_size_gpu0

			crop_h, crop_w, n_frames = random.choice(self.img_batch_tuples)
			end_index = min(start_index + batch_size, self.n_samples)
			batch_ids = self.img_indices[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if len(batch_ids) != batch_size:
				batch_ids += self.img_indices[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:

				batch = [
					(crop_h, crop_w, b_id, n_frames, n_clips) for b_id in batch_ids
				]
				yield batch


@Entry.register_entry(Entry.Sampler)
class VideoVariableSeqSamplerDDP(VariableBatchSamplerDDP):
	"""
	Extends `Variably-size multi-scale batch sampler <https://arxiv.org/abs/2110.02178?context=cs.LG>` for videos

	Args:
		n_data_samples (int): Number of samples in the dataset
		is_training (Optional[bool]): Training or validation mode. Default: False
	"""
	__slots__ = [
		"num_frames_per_clip", "random_video_clips",
		"min_clips_per_video", "max_clips_per_video", "clips_per_video", "min_frames_per_clip"
	]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
		8, False,
		1, 5, 1, 1
	],
	_types_ = [
		int, bool,
		int, int, int, int
	]

	def __init__(
		self,
		n_data_samples: int,
		is_training: Optional[bool] = False,
	) -> None:
		super().__init__(
			n_data_samples=n_data_samples, is_training=is_training
		)

		if not is_training:
			self.random_video_clips = False

		if is_training:
			# override img_batch_tuples
			self.img_batch_tuples = SplUtil.make_video_pairs(**self.__dict__)
		else:
			self.img_batch_tuples = [
				(self.crop_size_h, self.crop_size_w, self.default_frames)
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
		while start_index < self.n_samples_per_replica:
			if self.random_video_clips:
				# randomly sample number of clips and adjust batch size
				n_clips = max(
					1,
					random.randint(self.min_clips_per_video, self.max_clips_per_video),
				)
				batch_size = max(
					self.batch_size_gpu0,
					self.batch_size_gpu0 * (self.clips_per_video // n_clips),
				)
			else:
				n_clips = self.clips_per_video
				batch_size = self.batch_size_gpu0

			crop_h, crop_w, n_frames = random.choice(self.img_batch_tuples)

			end_index = min(start_index + batch_size, self.n_samples_per_replica)
			batch_ids = indices_rank_i[start_index:end_index]
			n_batch_samples = len(batch_ids)
			if n_batch_samples != batch_size:
				batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
			start_index += batch_size

			if len(batch_ids) > 0:
				batch = [
					(crop_h, crop_w, b_id, n_frames, n_clips) for b_id in batch_ids
				]
				yield batch
