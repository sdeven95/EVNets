from typing import Optional, List
from torch.utils.data import DataLoader

from .sampler import BaseSampler
from .dataset import BaseImageDataset

from utils.type_utils import Par


class EVNetsDataloader(DataLoader):
	"""This class extends PyTorch's Dataloader"""

	def __init__(
		self,
		dataset: BaseImageDataset,
		batch_size: int,
		batch_sampler: BaseSampler,
		num_workers: Optional[int] = 1,
		pin_memory: Optional[bool] = False,
		persistent_workers: Optional[bool] = False,
		collate_fn: Optional = None,
		prefetch_factor: Optional[int] = 2,
	):
		super(EVNetsDataloader, self).__init__(**Par.purify(locals()))

	def update_indices(self, new_indices: List):
		"""Update indices in the dataset class"""
		if hasattr(self.batch_sampler, "img_indices") and hasattr(
			self.batch_sampler, "update_indices"
		):
			self.batch_sampler.update_indices(new_indices)

	def samples_in_dataset(self):
		"""Number of samples in the dataset"""
		return len(self.batch_sampler.img_indices)

	def get_sample_indices(self) -> List:
		"""Sample IDs"""
		return self.batch_sampler.img_indices
