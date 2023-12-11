import os
import pickle
import pathlib
import glob
import torch

from .. import BaseImageDataset
from ...video_transform import extensions as T

from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils.logger import Logger
from utils import Util

from typing import Optional, List, Union, Tuple


@Entry.register_entry(Entry.Dataset)
class KineticsDataset(BaseImageDataset):
	"""
	Dataset class for the Kinetics dataset

	Args:
		is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
		is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False
	"""

	__slots__ = ["metadata_file_train", "metadata_file_val"]
	_keys_ = __slots__
	_defaults_ = [None, None]
	_types_ = [str, str]
	_disp_ = __slots__

	@classmethod
	def add_arguments(cls):
		super().add_arguments()
		super()._add_arguments_(
			keys=KineticsDataset._args_,
			defaults=[None, None],
			types=[str, str]
		)

	def __init__(
		self,
		is_training: Optional[bool] = None,
		is_evaluation: Optional[bool] = None,
	) -> None:

		super().__init__(is_training=is_training, is_evaluation=is_evaluation)

		assert os.path.isdir(self.root), f"Directory does not exist: {self.root}"

		# get video reader
		self.video_reader = Entry.get_entity(Entry.VideoReader, is_training=is_training)
		metadata_file = self.metadata_file_train if is_training else self.metadata_file_val

		# firstly try to get samples from the metafile
		if metadata_file is not None:
			# internally, we take care that master node only downloads the file
			with open(metadata_file, "rb") as f:
				self.samples = pickle.load(f)
			assert isinstance(self.samples, List)
		# otherwise build structure from folders and save to the metadata file
		else:
			# each folder is a class
			class_names = sorted(
				(f.name for f in pathlib.Path(self.root).iterdir() if f.is_dir())
			)

			samples = []
			extensions = ["avi", "mp4"]
			for cls_idx in range(len(class_names)):
				cls_name = class_names[cls_idx]   # class name (dir name)
				class_folder = os.path.join(self.root, cls_name)
				# for each video file, get sample = { "label": idx, "video_path": ...}
				for video_path in glob.glob(f"{class_folder}/*"):
					file_extn = video_path.split(".")[-1]
					if (
						(file_extn in extensions)
						and os.path.isfile(video_path)
						and self.video_reader.check_video(filename=video_path)
					):
						samples.append({"label": cls_idx, "video_path": video_path})
			self.samples = samples
			results_loc = Opt.get(f"{Entry.Common}.Common.results_loc")

			# save meta data file
			if Util.is_master_node():
				stage = "train" if is_training else "val"
				metadata_file_loc = f"{results_loc}/kinetics_metadata_{stage}.pkl"

				with open(metadata_file_loc, "wb") as f:
					pickle.dump(self.samples, f)
				Logger.log(f"Metadata file saved at: {metadata_file_loc}")

		self.n_records = len(self.samples)

		# Cfg.set_by_cfg_path(Entry.VideoClassificationModelConfigure, "n_classes", len(self.class_names()))

	def _get_training_transforms(self, size: tuple or int):
		return T.Compose(video_transforms=[
			T.RandomCropResize(size=size),
			T.RandomHorizontalFlip(),
		])

	def _get_validation_transforms(self, size: Union[Tuple, List, int]):
		return T.Compose(video_transforms=[
			T.Resize(),
			T.CenterCrop(size=size),
		])

	def _evaluation_transforms(self, size: tuple):
		return self._get_validation_transforms(size=size)

	def __getitem__(self, batch_indexes_tup):
		(
			crop_size_h,
			crop_size_w,
			index,
			n_frames_to_sample,  # number of sampled frames for each clip
			clips_per_video,
		) = batch_indexes_tup
		if self.is_training:
			transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
		else:  # same for validation and evaluation
			transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

		try:
			info: dict = self.samples[index]
			target = info["label"]  # label index for video file

			# Default is Tensor of size [K, N, C, H, W].
			# If --dataset.kinetics.frame-stack-format="channel_first", then clip is of size [K, C, N, H, W]
			# here, K --> no. of clips, C --> Image channels, N --> Number of frames per clip, H --> Height, W --> Width
			input_video = self.video_reader.process_video(
				vid_filename=info["video_path"],
				n_frames_per_clip=n_frames_to_sample,
				clips_per_video=clips_per_video,
				video_transform_fn=transform_fn,
				is_training=self.is_training,
			)

			# if failed, return empty data
			if input_video is None:
				Logger.log("Corrupted video file: {}".format(info["video_path"]))
				input_video = self.video_reader.dummy_video(
					clips_per_video=clips_per_video,
					n_frames_to_sample=n_frames_to_sample,
					height=crop_size_h,
					width=crop_size_w,
				)

				data = {"image": input_video}
				target = Opt.get_by_cfg_path(Entry.Loss, "ignore_idx")
			else:
				data = {"image": input_video}

		# if failed, return empty data
		# [K, C, N, H, W] or # [K, N, C, H, W]
		# K --> number of clips, C --> Image channels, N --> Number of frames per clip, H --> Height, W --> Width
		except Exception as e:
			Logger.log("Unable to load index: {}. Error: {}".format(index, str(e)))
			input_video = self.video_reader.dummy_video(
				clips_per_video=clips_per_video,
				n_frames_to_sample=n_frames_to_sample,
				height=crop_size_h,
				width=crop_size_w,
			)

			target = Opt.get_by_cfg_path(Entry.Loss, "ignore_idx")
			data = {"image": input_video}

		# target is a 0-dimensional tensor
		# every clip has the same tensor
		data["label"] = torch.LongTensor(size=(input_video.shape[0],)).fill_(target)

		return data

	def __len__(self):
		return len(self.samples)

	def __repr__(self):
		from ...sampler.utils import SplUtil

		im_h, im_w = SplUtil.image_size_from_opts()

		transforms_str = \
			self._get_training_transforms(size=[im_h, im_w]) if self.is_training \
			else self._get_validation_transforms(size=[im_h, im_w])

		if hasattr(self.video_reader, "frame_transforms_str"):
			frame_transforms_str = self.video_reader.frame_transforms_str
		else:
			frame_transforms_str = None

		return \
			super()._repr_by_line() \
			+ "\nVideo transform details:\n" + repr(transforms_str) \
			+ "\nFrame transform details:\n" + frame_transforms_str


@Entry.register_entry(Entry.Collate)
def kinetics_collate_fn(batch: List):
	batch_size = len(batch)

	images = []
	labels = []
	for b in range(batch_size):
		images.append(batch[b]["image"])
		labels.append(batch[b]["label"])

	images = torch.cat(images, dim=0)  # [N, ...]
	labels = torch.cat(labels, dim=0)  # [N, ...]

	# check for contiguous
	if not images.is_contiguous():
		images = images.contiguous()

	if not labels.is_contiguous():
		labels = labels.contiguous()

	return {"image": images, "label": labels}


@Entry.register_entry(Entry.Collate)
def kinetics_collate_fn_train(batch: List):
	batch_size = len(batch)
	ignore_label = Opt.get_by_cfg_path(Entry.Loss, "ignore_idx")

	images = []
	labels = []
	for b in range(batch_size):
		b_label = batch[b]["label"]
		if ignore_label in b_label:
			continue
		images.append(batch[b]["image"])
		labels.append(b_label)

	images = torch.cat(images, dim=0)
	labels = torch.cat(labels, dim=0)

	# check for contiguous
	if not images.is_contiguous():
		images = images.contiguous()

	if not labels.is_contiguous():
		labels = labels.contiguous()

	return {"image": images, "label": labels}
