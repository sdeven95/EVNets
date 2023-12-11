import torch.cuda
import torch.distributed as dist

from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry
from torch.utils.data.sampler import Sampler
import math


@Entry.register_entry(Entry.Sampler)
class SampleEfficientTraining(Cfg, Dsp):
	__slots__ = ["enable", "sample_confidence", "find_easy_samples_every_k_epochs", "min_sample_frequency", ]
	_cfg_path_ = Entry.Sampler
	_keys_ = __slots__
	_defaults_ = [False, 0.5, 5, 5]
	_types_ = [bool, float, int, int]
	_disp_ = __slots__


class BaseSampler(Cfg, Dsp, Sampler):
	__slots__ = [
		"train_batch_size", "val_batch_size", "eval_batch_size",
		"n_data_samples", "is_training",
		"batch_size_gpu0", "n_gpus", "rank", "num_replicas"
	]
	_cfg_path_ = Entry.Sampler
	_keys_ = __slots__[:-4]
	_defaults_ = [32, 32, 32, None, None],
	_types_ = [int, int, int, int, bool],

	def __iter__(self):
		raise NotImplementedError

	def set_epoch(self, epoch):
		self.epoch = epoch

	def update_indices(self, new_indices):
		self.img_indices = new_indices

	def update_scales(self):
		pass

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def __repr__(self):
		return super()._repr_by_line()


# note: delivery n_data_samples and is_training
class BaseSamplerDP(BaseSampler):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		n_gpus = max(1, torch.cuda.device_count())
		batch_size_gpu0 = self.train_batch_size if self.is_training else self.val_batch_size

		n_samples_per_gpu = int(math.ceil(self.n_data_samples * 1.0 / n_gpus))  # number of samples per gpu
		total_size = n_samples_per_gpu * n_gpus  # total_size >= n_data_samples

		indices = [idx for idx in range(self.n_data_samples)]  # all image indices in the dataset
		# This ensures that we can divide the batches evenly across GPUs
		# given total_size > n_data_samples, add some indices to the tail so the length reaches to total_size
		# such as: 2 gpus, let 5 samples per gpu, but only 8 samples, then indices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
		indices += indices[: (total_size - self.n_data_samples)]
		assert total_size == len(indices)  # now the length of indices equals to total size

		self.img_indices = indices
		self.n_samples = total_size
		self.batch_size_gpu0 = batch_size_gpu0
		self.n_gpus = n_gpus
		self.shuffle = self.is_training
		self.epoch = 0  # init epoch

		self.rank = None
		self.num_replicas = None

	def __iter__(self):
		raise NotImplementedError

	def __len__(self):
		return len(self.img_indices)


# base class for distributedDataParallel Sampler
# will truncate samplers so every replica will be given the same number of samplers
class BaseSamplerDDP(BaseSampler):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		batch_size_gpu0 = self.train_batch_size if self.is_training else self.val_batch_size

		if not dist.is_available():
			raise RuntimeError("Requires distributed package to be available")

		num_replicas = dist.get_world_size()  # 2
		rank = dist.get_rank()   # 0 1

		num_samples_per_replica = int(math.ceil(self.n_data_samples * 1.0 / num_replicas))  # number of samples per replica
		total_size = num_samples_per_replica * num_replicas

		img_indices = [idx for idx in range(self.n_data_samples)]
		img_indices += img_indices[: (total_size - self.n_data_samples)]
		assert len(img_indices) == total_size

		self.img_indices = img_indices
		self.n_samples_per_replica = num_samples_per_replica
		self.shuffle = self.is_training
		self.epoch = 0  # init epoch
		self.rank = rank  # rank
		self.batch_size_gpu0 = batch_size_gpu0
		self.num_replicas = num_replicas
		self.n_gpus = num_replicas
		self.skip_sample_indices = []

	def __iter__(self):
		raise NotImplementedError

	def __len__(self):
		return len(self.img_indices) // self.num_replicas


# obsoleted
# from .batch_sampler import BatchSampler, BatchSamplerDDP
# from .variable_batch_sampler import VariableBatchSampler, VariableBatchSamplerDDP
# from .multi_scale_sampler import MultiScaleSampler, MultiScaleSamplerDDP
# from .video_batch_sampler import VideoBatchSampler, VideoBatchSamplerDDP
# from .video_variable_seq_sampler import VideoVariableSeqSampler, VideoVariableSeqSamplerDDP


