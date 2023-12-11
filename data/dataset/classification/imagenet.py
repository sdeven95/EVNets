import sys
import torch
from torchvision.datasets import ImageFolder

from data.dataset import BaseImageDataset
import data.transform as T

from utils import Util
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils import Logger


@Entry.register_entry(Entry.Dataset)
class ImagenetDataset(BaseImageDataset, ImageFolder):
	__slots__ = ["crop_ratio", ]
	_keys_ = __slots__
	_defaults_ = [0.875, ]
	_types_ = [float, ]
	_disp_ = __slots__

	def __init__(self, **kwargs):

		self.opts = sys.modules["_opts_"]

		BaseImageDataset.__init__(self, **kwargs)
		ImageFolder.__init__(self, root=self.root, transform=None, target_transform=None, is_valid_file=None)

		self.n_classes = len(list(self.class_to_idx.keys()))
		Opt.set_by_cfg_path(Entry.Common, "n_classes", self.n_classes)

		self.n_records = len(self.samples)
		for aug in ("train", "val", "eval"):
			Opt.set(f"{Entry.Collate}.{aug}", "imagenet_collate_fn")

	def _get_training_transforms(self, size):
		random_resize_crop, color_jitter, random_horizontal_flip, auto_augment, rand_augment, normalize, random_erasing =\
			Opt.get_argument_values(
				pth_tpl=Entry.Transform + ".{}",
				keys=(
					"RandomResizedCrop.enable",
					"ColorJitter.enable",
					"RandomHorizontalFlip.enable",
					"AutoAugment.enable",
					"RandAugment.enable",
					"Normalize.enable",
					"RandomErasing.enable",
				),
				defaults=(False, False, False, False, False, False, False)
			)

		assert not (auto_augment and rand_augment), \
			"AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"

		return T.Compose(img_transforms=[
			T.RandomCropResize(size=size) if random_resize_crop else T.Identity,
			T.ColorJitter() if color_jitter else T.Identity,
			T.RandomHorizontalFlip() if random_horizontal_flip else T.Identity,
			T.AutoAugment() if auto_augment else T.Identity,
			T.RandAugment() if rand_augment else T.Identity,
			T.ToTensor(),
			T.Normalize() if normalize else T.Identity,
			T.RandomErasing() if random_erasing else T.Identity,
		])

	def _get_validation_transforms(self, size):
		return T.Compose(img_transforms=[
			T.Resize(),
			T.CenterCrop(),
			T.ToTensor(),
		])

	def _get_evaluation_transforms(self, size):
		return self._get_validation_transforms(size)

	def __getitem__(self, index_tuple):
		"""
		: param index_tuple: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
		:return: dictionary containing input image, label, and sample_id.
		"""
		if "_opts_" not in sys.modules:
			sys.modules["_opts_"] = self.opts

		crop_size_h, crop_size_w, img_index = index_tuple
		transform_fn = \
			self._get_training_transforms(
				size=(crop_size_h, crop_size_w)
			) \
			if self.is_training \
			else self._get_validation_transforms(size=(crop_size_h, crop_size_w))

		img_path, target = self.samples[img_index]
		input_img = self.read_image_pil(img_path)

		if input_img is None:
			# Sometimes images are corrupt
			# Skip such images
			Logger.log("Img index {} is possibly corrupt.".format(img_index))
			input_tensor = torch.zeros(
				size=(3, crop_size_h, crop_size_w), dtype=torch.float
			)
			target = -1
			data = {"image": input_tensor}
		else:
			data = {"image": input_img}
			data = transform_fn(data)

		data["label"] = target
		data["sample_id"] = img_index

		return data

	def __len__(self) -> int:
		return len(self.samples)

	def __repr__(self) -> str:
		from ...sampler.utils import SplUtil

		im_h, im_w = SplUtil.image_size_from_opts()

		transforms_str = \
			self._training_transforms(size=[im_h, im_w]) \
			if self.is_training \
			else self._validation_transforms(size=[im_h, im_w])

		return super()._repr_by_line() + "\nTransform details:\n" + repr(transforms_str)

		# return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tn_classes={}\n\ttransform={}\n)".format(
		# 	self.__class__.__name__,
		# 	self.root,
		# 	self.is_training,
		# 	len(self.samples),
		# 	self.n_classes,
		# 	transforms_str,
		# )


@Entry.register_entry(Entry.Collate)
def imagenet_collate_fn(batch):
	batch_size = len(batch)
	img_size = [batch_size, *batch[0]["image"].shape]
	img_dtype = batch[0]["image"].dtype

	images = torch.zeros(size=img_size, dtype=img_dtype)
	# fill with -1, so that we can ignore corrupted images
	labels = torch.full(size=[batch_size], fill_value=-1, dtype=torch.long)
	sample_ids = torch.zeros(size=[batch_size], dtype=torch.long)
	valid_indexes = []
	# reorganize data
	for i, batch_i in enumerate(batch):
		label_i = batch_i.pop("label")
		images[i] = batch_i.pop("image")
		labels[i] = label_i  # label is an int
		sample_ids[i] = batch_i.pop("sample_id")  # sample id is an int
		if label_i != -1:  # filter corrupt image
			valid_indexes.append(i)

	# get data without corrupt images
	valid_indexes = torch.tensor(valid_indexes, dtype=torch.long)
	images = torch.index_select(images, dim=0, index=valid_indexes)
	labels = torch.index_select(labels, dim=0, index=valid_indexes)
	sample_ids = torch.index_select(sample_ids, dim=0, index=valid_indexes)

	# set memory format
	if Util.channel_last():
		images = images.to(memory_format=torch.channels_last)

	return {
		"image": images,
		"label": labels,
		"sample_id": sample_ids,
		"on_gpu": images.is_cuda,
	}
